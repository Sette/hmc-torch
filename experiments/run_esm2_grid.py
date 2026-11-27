#!/usr/bin/env python3
"""Spec 29: small validation grid of hyperparameters for the three neural ESM-2 methods.

For each dataset and method (``ours`` = global MLP + BCE, ``chmcnn``, ``hmcnf``) the same
grid is run on seed 0, training on the train split only (no refit) and scoring the
validation AUPRC of the reconciled outputs; the best configuration is then run with the
usual protocol (3 seeds, refit on train + validation) and evaluated on test once.  Nothing
about the test split enters the selection.

Grid: hidden multiplier x lr x dropout.  Hidden multipliers are ``0.5, 1, 2`` on
``seq_FUN`` and ``0.5, 1`` on ``seq_GO`` (2x of the 9,000-unit layer does not fit the GPU).
HMCN-F multiplies its 384 units per level by the same factors.

Writes ``output/protein_lm_baselines/grid/<dataset>_<method>.json`` (existing files are
skipped) and ``grid/summary.md``.

Usage:
    PYTHONPATH=src python experiments/run_esm2_grid.py --datasets seq_FUN,seq_GO
    PYTHONPATH=src python experiments/run_esm2_grid.py --datasets seq_FUN --methods ours --max_epochs 2 --grid_seeds 0
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from hmc.features.protein import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_PROTEIN_MODEL,
    ProteinFeatureEncoder,
    orf_embedding_features,
)
from hmc.features.yeast_orf import (  # noqa: E402  pylint: disable=wrong-import-position
    read_orf_ids,
    read_proteome,
)
from run_esm2_baselines import (  # noqa: E402  pylint: disable=wrong-import-position
    OBJECTIVE,
    chmcnn_model_fn,
    hmcnf_model_fn,
    sd,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    hyperparams,
    load_dataset,
    train_mlp,
)

OUT_DIR = "output/protein_lm_baselines/grid"
LRS = (1e-4, 3e-4)
DROPOUTS = (0.5, 0.7)
MULTS = {"seq_FUN": (0.5, 1.0, 2.0), "seq_GO": (0.5, 1.0)}
METHODS = ("ours", "chmcnn", "hmcnf")
HMCNF_HIDDEN = 384


def configure(method, base, mult, lr, dropout):
    """Hyperparameters of one grid point."""
    hp = base | {"lr": lr, "dropout": dropout}
    if method == "hmcnf":
        hp["hmcnf_hidden"] = int(HMCNF_HIDDEN * mult)
    else:
        hp["hidden"] = int(base["hidden"] * mult)
    return hp


def model_fn_for(method, data, hp, device):
    """Model factory for ``method`` (``None`` = the default MLP of ``run_protocol``)."""
    if method == "chmcnn":
        return chmcnn_model_fn(data, hp, device)
    if method == "hmcnf":
        return hmcnf_model_fn(data, hp)
    return None


def run_method(name, method, data, base, args):
    """Grid on seed 0 (no refit), then the final protocol with the best configuration."""
    path = os.path.join(OUT_DIR, f"{name}_{method}.json")
    if os.path.exists(path):
        print(f"[skip] {name}/{method}: {path} exists")
        return
    kind = OBJECTIVE[method]
    t0 = time.time()
    grid = []
    for mult, lr, dropout in itertools.product(MULTS[name], LRS, DROPOUTS):
        hp = configure(method, base, mult, lr, dropout) | {"refit": False}
        run = train_mlp(data, kind, args.grid_seeds[0], args.device, hp,
                        model_fn=model_fn_for(method, data, hp, args.device))
        grid.append({"mult": mult, "lr": lr, "dropout": dropout, "val_auprc": run["val_auprc"],
                     "best_epoch": run["best_epoch"]})
        print(f"  {name}/{method} x{mult} lr {lr:g} d {dropout}: val AUPRC {run['val_auprc']:.4f} "
              f"(ep {run['best_epoch']}, {run['train_time_s']:.0f}s)", flush=True)
    best = max(grid, key=lambda g: g["val_auprc"])
    hp = configure(method, base, best["mult"], best["lr"], best["dropout"])
    runs = []
    for seed in args.seeds:
        run = train_mlp(data, kind, seed, args.device, hp,
                        model_fn=model_fn_for(method, data, hp, args.device))
        runs.append(run)
        ev = run["eval"]
        print(f"  {name}/{method} final seed {seed}: raw {ev['raw']['auprc']:.4f} rec {ev['rec']['auprc']:.4f}",
              flush=True)
    record = {"dataset": name, "method": method, "objective": kind, "grid": grid, "best": best,
              "hyperparams": hp, "runs": runs, "wall_time_s": time.time() - t0,
              "protocol": "grid: seed 0, train only, validation AUPRC (reconciled); final: 3 seeds, refit"}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=float)


def summarize(names):
    """Markdown table of the tuned results (test AUPRC, mean ± sd over seeds)."""
    lines = ["# ESM-2, tuned on validation: test AUPRC (micro), mean ± sd (ddof = 1)", "",
             "| dataset | method | best (mult, lr, dropout) | val AUPRC | test raw | test rec. |",
             "|---|---|---|---|---|---|"]
    for name in names:
        for method in METHODS:
            path = os.path.join(OUT_DIR, f"{name}_{method}.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            b = rec["best"]
            cells = [f"{np.mean(v):.4f} ± {sd(v):.4f}"
                     for v in ([r["eval"][m]["auprc"] for r in rec["runs"]] for m in ("raw", "rec"))]
            lines.append(f"| {name} | {method} | x{b['mult']}, {b['lr']:g}, {b['dropout']} | "
                         f"{b['val_auprc']:.4f} | {cells[0]} | {cells[1]} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="seq_FUN,seq_GO")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0,42,123")
    parser.add_argument("--grid_seeds", default="0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default=DEFAULT_PROTEIN_MODEL)
    parser.add_argument("--proteome", default="data/yeast/raw/orf_trans_all.fasta.gz")
    parser.add_argument("--ids_dir", default="data/yeast")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.grid_seeds = [int(s) for s in args.grid_seeds.split(",")]
    args.device = torch.device(args.device)

    proteome = read_proteome(args.proteome)
    encoder = ProteinFeatureEncoder(args.model)
    names = args.datasets.split(",")
    for name in names:
        todo = [m for m in args.methods.split(",")
                if not os.path.exists(os.path.join(OUT_DIR, f"{name}_{m}.json"))]
        if todo:
            data = load_dataset(name)
            ids = read_orf_ids(os.path.join(args.ids_dir, f"{name}_ids.tsv"))
            data["x"] = orf_embedding_features(data["x"], ids, encoder, proteome)
            base = hyperparams(name)
            if args.max_epochs:
                base["max_epochs"] = args.max_epochs
            print(f"\n=== {name}: grid for {todo}", flush=True)
            for method in todo:
                run_method(name, method, data, base, args)
    summarize(names)


if __name__ == "__main__":
    main()

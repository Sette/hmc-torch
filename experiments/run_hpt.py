#!/usr/bin/env python3
"""HPT (prompt tuning) combined with the R-matrix, under the paper's protocol.

Each run fine-tunes a masked LM with HPT's prompt (one ``[PRED]`` slot per
level, GAT-encoded label tokens, MLM auxiliary loss), trained with ZMLCE plus
one hierarchy objective:

* ``none``  -- ZMLCE only: HPT itself;
* ``mc``    -- ZMLCE on C-HMCNN's max-constraint outputs (MCLoss on logits);
* ``hinge`` -- ZMLCE + Eq. 2 over all ancestor pairs.

Training uses the training split only; the epoch (early stopping, patience 6)
and the threshold are chosen on validation; every model is scored raw and
after bottom-up reconciliation, at HPT's fixed threshold (logit 0) and at the
validation one, with micro/macro F1 (HPT's definition), AUPRC and GHV.

Stages:

* ``screen`` -- seed 0, at most 10 epochs, validation only (test untouched):
  {bert, specter2} x {none, mc, hinge} with bidirectional graph edges, plus
  HPT's own configuration (bert, none, one-way edges, ``hpt_ref``).
* ``final``  -- one configuration, 3 seeds, test scored once.  ``--auto``
  takes the screen's best configuration by validation micro-F1.

Usage:
    PYTHONPATH=src python experiments/run_hpt.py --dataset wos --stage screen
    PYTHONPATH=src python experiments/run_hpt.py --dataset wos --stage final --auto
    PYTHONPATH=src python experiments/run_hpt.py --dataset wos --stage final \\
        --backbone bert --loss none --graph down --seed 1
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, "src")

from hmc.datasets.dataset_manager import (  # noqa: E402  pylint: disable=wrong-import-position
    initialize_dataset_experiments,
)
from hmc.pipeline.hpt.main import (  # noqa: E402  pylint: disable=wrong-import-position
    BACKBONES,
    HPTConfig,
    evaluate_hpt,
    fit_hpt,
)

DATASETS = ("wos", "aapd", "arxiv", "eurlex")
SCREEN = [(b, loss, "both") for b in ("bert", "specter2") for loss in ("none", "mc", "hinge")]
SCREEN.append(("bert", "none", "down"))  # HPT as published
SEEDS = (0, 1, 2)


def run_name(
    backbone: str,
    loss: str,
    graph: str,
    seed: int,
    mlm: bool = True,
    selection: str = "rec_val_micro",
) -> str:
    suffix = "" if selection == "rec_val_micro" else f"_{selection}"
    return f"{backbone}_{loss}_{graph}{'' if mlm else '_nomlm'}{suffix}_seed{seed}"


def load_manager(dataset: str, limit: int | None):
    # ArXiv keeps its 50k-record cap (as in run_protocol.py); the rest are used whole.
    cap = 50_000 if dataset == "arxiv" else 0
    if limit:
        cap = limit * 5 if dataset in ("arxiv", "aapd") else cap
    return initialize_dataset_experiments(
        dataset,
        device="cpu",
        dataset_path="./data",
        is_global=True,
        arxiv_max_records=cap,
        model_cache_dir="./models",
        arxiv_load_features=False,
    )


def run_one(mgr, args, backbone, loss, graph, seed, out_path, score_test):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    lock = out_path + ".lock"
    if os.path.exists(out_path) or os.path.exists(lock):
        print(f"[skip] {out_path} exists or is being produced", flush=True)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(lock, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        cfg = HPTConfig(
            backbone=BACKBONES[backbone],
            loss=loss,
            mlm=not args.no_mlm,
            graph_direction=graph,
            max_epochs=args.max_epochs or (10 if args.stage == "screen" else 20),
            patience=args.patience,
            selection=args.selection,
            seed=seed,
            limit=args.limit,
        )
        print(f"== {args.dataset} {os.path.basename(out_path)}", flush=True)
        t0 = time.time()
        fit = fit_hpt(mgr, cfg, log=lambda msg: print(msg, flush=True), score_test=score_test)
        result = {
            "dataset": args.dataset,
            "stage": args.stage,
            "backbone": backbone,
            "loss": loss,
            "graph": graph,
            "mlm": cfg.mlm,
            "seed": seed,
            "config": asdict(cfg),
            "best_epoch": fit["best_epoch"],
            "best_val_f1": max(h["val_f1"] for h in fit["history"]),
            "best_selection_score": max(
                h["selection_score"] for h in fit["history"]
            ),
            "selection": cfg.selection,
            "history": fit["history"],
            "train_time_s": fit["train_time_s"],
            "wall_time_s": time.time() - t0,
            "n_train": fit["n"][0],
            "n_valid": fit["n"][1],
            "n_test": fit["n"][2],
        }
        if score_test:
            result.update(evaluate_hpt(fit))
            for mode in ("raw", "rec"):
                for thr in ("thr0", "val"):
                    r = result[mode][thr]
                    print(
                        f"test {mode}/{thr}: micro-F1 {r['f1']:.4f} macro-F1 "
                        f"{r['macro_f1']:.4f} (thr {r['threshold']})",
                        flush=True,
                    )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=float)
    finally:
        os.remove(lock)


def best_screen_config(out_dir: str, dataset: str) -> tuple[str, str, str]:
    """Screen configuration with the highest validation micro-F1."""
    runs = []
    for path in glob.glob(os.path.join(out_dir, "screen", dataset, "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        runs.append((r["best_val_f1"], r["backbone"], r["loss"], r["graph"]))
    if not runs:
        raise SystemExit(f"no screen results for {dataset}; run --stage screen first")
    _, backbone, loss, graph = max(runs)
    print(f"[auto] {dataset}: best screen config = {backbone}/{loss}/{graph}", flush=True)
    return backbone, loss, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--stage", default="screen", choices=["screen", "final"])
    parser.add_argument("--backbone", choices=sorted(BACKBONES))
    parser.add_argument("--loss", choices=["none", "mc", "hinge"])
    parser.add_argument("--graph", default="both", choices=["both", "down"])
    parser.add_argument("--seed", type=int, help="final stage: one seed (default: 0, 1, 2)")
    parser.add_argument("--auto", action="store_true", help="final stage: best screen config")
    parser.add_argument("--no_mlm", action="store_true", help="drop the MLM auxiliary loss")
    parser.add_argument(
        "--selection",
        default="rec_val_micro",
        choices=["rec_val_micro", "hpt_raw"],
        help=(
            "epoch-selection metric: our reconciled validation micro-F1 with a "
            "tuned threshold, or HPT's raw validation macro-F1 at logit 0"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="early-stopping patience in epochs (HPT uses 5)",
    )
    parser.add_argument("--max_epochs", type=int, help="default: 10 (screen) / 20 (final)")
    parser.add_argument("--limit", type=int, help="smoke test: first N documents per split")
    parser.add_argument("--out_dir", default="./output/hpt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.stage == "screen":
        configs = [(b, loss, g, 0) for b, loss, g in SCREEN]
        if args.backbone or args.loss:  # a single screen run
            configs = [(args.backbone or "bert", args.loss or "none", args.graph, 0)]
    else:
        if args.auto:
            backbone, loss, graph = best_screen_config(args.out_dir, args.dataset)
        elif args.backbone and args.loss:
            backbone, loss, graph = args.backbone, args.loss, args.graph
        else:
            parser.error("--stage final needs --auto or both --backbone and --loss")
        seeds = (args.seed,) if args.seed is not None else SEEDS
        configs = [(backbone, loss, graph, s) for s in seeds]

    mgr = load_manager(args.dataset, args.limit)
    out_dir = os.path.join(args.out_dir, args.stage, args.dataset)
    if args.limit:
        out_dir = os.path.join(args.out_dir, "smoke", args.stage, args.dataset)
    for backbone, loss, graph, seed in configs:
        name = run_name(backbone, loss, graph, seed, not args.no_mlm, args.selection)
        run_one(
            mgr, args, backbone, loss, graph, seed,
            os.path.join(out_dir, f"{name}.json"),
            score_test=args.stage == "final",
        )


if __name__ == "__main__":
    main()

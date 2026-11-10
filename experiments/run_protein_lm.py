#!/usr/bin/env python3
"""Do protein language-model embeddings beat the hand-crafted ``seq_`` features?

The ``seq_GO`` / ``seq_FUN`` rows are mapped back to yeast ORFs by
``recover_seq_ids.py``; this script embeds those proteins with a frozen ESM-2
(:class:`hmc.features.protein.ProteinFeatureEncoder`) and trains the global MLP
under exactly the protocol of ``run_protocol.py`` (early stopping and threshold
on validation, refit on train + validation, three seeds, BCE / hinge / MCLoss,
raw and reconciled).

Feature variants:
  * ``orig``     -- the ARFF features (control; reproduces run_protocol.py);
  * ``esm``      -- ESM-2 mean-pooled embeddings only;
  * ``esm+orig`` -- both, concatenated.
Rows without a unique ORF match get the training-mean embedding plus an
indicator column; their fraction is reported per split.

Writes ``output/protein_lm/<model>/<dataset>.json``.

Usage:
    PYTHONPATH=src python experiments/recover_seq_ids.py --dataset seq_GO
    PYTHONPATH=src python experiments/run_protein_lm.py --datasets seq_GO
"""

import argparse
import csv
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
)
from hmc.utils.datasets.scaling import (  # noqa: E402  pylint: disable=wrong-import-position
    standardize_splits,
)
from recover_seq_ids import (  # noqa: E402  pylint: disable=wrong-import-position
    read_proteome,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    OBJECTIVES,
    hyperparams,
    load_dataset,
    train_mlp,
)

SPLITS = ("train", "valid", "test")
VARIANTS = ("orig", "esm", "esm+orig")


def read_ids(path):
    """``{split: [orf or None per row]}`` from a ``recover_seq_ids.py`` table."""
    ids = {s: [] for s in SPLITS}
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            ids[rec["split"]].append(rec["orf"] if rec["unique"] == "1" else None)
    return ids


def esm_features(data, ids, encoder, proteome):
    """Per-split ESM matrices (standardized on train) with a missing-match column."""
    seq_of = {orf: seq for orf, _, seq in proteome}
    orfs = sorted({o for split in SPLITS for o in ids[split] if o})
    emb = dict(zip(orfs, encoder.encode([seq_of[o] for o in orfs])))
    dim = next(iter(emb.values())).shape[0]

    mats = []
    for split, x in zip(SPLITS, data["x"]):
        if len(ids[split]) != len(x):
            raise ValueError(f"{split}: {len(ids[split])} ids for {len(x)} rows")
        m = np.full((len(x), dim), np.nan, dtype=np.float32)
        for r, orf in enumerate(ids[split]):
            if orf:
                m[r] = emb[orf]
        mats.append(m)
    # standardize_splits imputes NaNs with the training mean before scaling.
    mats = standardize_splits(*mats)
    return [
        np.hstack([m, np.array([[o is None] for o in ids[s]], dtype=np.float32)])
        for m, s in zip(mats, SPLITS)
    ]


def run_dataset(name, args, encoder, proteome):
    model_tag = encoder.model_name.split("/")[-1]
    path = os.path.join(args.out_dir, model_tag, f"{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}: {path} exists")
        return
    t0 = time.time()
    data = load_dataset(name)
    ids = read_ids(os.path.join(args.ids_dir, f"{name}_ids.tsv"))
    esm = esm_features(data, ids, encoder, proteome)
    feats = {
        "orig": list(data["x"]),
        "esm": esm,
        "esm+orig": [np.hstack([e, o]) for e, o in zip(esm, data["x"])],
    }
    hp = hyperparams(name)
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    record = {
        "dataset": name,
        "protein_model": encoder.model_name,
        "unmatched_frac": {s: float(np.mean([o is None for o in ids[s]])) for s in SPLITS},
        "hyperparams": hp,
        "protocol": "train on train; early stopping and threshold on valid; report on test",
        "runs": [],
    }
    print(f"\n=== {name}: unmatched {record['unmatched_frac']}")
    for variant in args.variants:
        data["x"] = feats[variant]
        for seed in args.seeds:
            for kind in args.objectives:
                run = train_mlp(data, kind, seed, args.device, hp)
                run["features"] = variant
                run["n_features"] = int(data["x"][0].shape[1])
                record["runs"].append(run)
                ev = run["eval"]
                print(f"  {variant:<8} seed {seed:>3} {kind:<5} ep {run['best_epoch']:>3}  "
                      f"raw AUPRC {ev['raw']['auprc']:.4f} | rec AUPRC {ev['rec']['auprc']:.4f}  "
                      f"F1 {ev['rec']['f1']:.4f} ({run['train_time_s']:.0f}s)", flush=True)
    record["wall_time_s"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=float)

    print(f"--- {name} summary (rec AUPRC, mean ± sd over seeds)")
    for variant in args.variants:
        for kind in args.objectives:
            vals = [r["eval"]["rec"]["auprc"] for r in record["runs"]
                    if r["features"] == variant and r["objective"] == kind]
            print(f"  {variant:<8} {kind:<5} {np.mean(vals):.4f} ± {np.std(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="seq_GO")
    parser.add_argument("--model", default=DEFAULT_PROTEIN_MODEL)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,42,123")
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--proteome", default="data/yeast/raw/orf_trans_all.fasta.gz")
    parser.add_argument("--ids_dir", default="data/yeast")
    parser.add_argument("--out_dir", default="./output/protein_lm")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()
    args.variants = args.variants.split(",")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.objectives = args.objectives.split(",")
    args.device = torch.device(args.device)

    proteome = read_proteome(args.proteome)
    encoder = ProteinFeatureEncoder(args.model)
    for name in args.datasets.split(","):
        run_dataset(name, args, encoder, proteome)


if __name__ == "__main__":
    main()

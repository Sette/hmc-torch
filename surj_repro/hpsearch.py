#!/usr/bin/env python3
"""Choose Surj's hyperparameters on validation, once for every dataset (paper, Sec. 4.4).

"The hyper-parameters in our framework are the dimensions of the fully connected layers and
the learning rate.  These hyper-parameters are optimized with the validation set. [...] The
hyper-parameters are consistent across datasets."

Staged grid on the 8 FunCat datasets, seed 0, train split only, early stopping on
validation; the score of a configuration is its mean validation AP.  The test split is never
scored (``train.py --no_test``).  Each stage keeps the best value found so far:

  A. lr x batch          {1e-3, 3e-4, 1e-4} x {4, 32, 128}
  B. dropout             {0.3, 0.5, 0.7}
  C. hidden x D          {C-HMCNN's, 512, 1024} x {16, 64, 128}
  D. hierarchy constraint {none, MCM + MCLoss}

The winner is written to ``recipes.json`` as ``selected`` (and ``selected_onehot``, the same
recipe with one-hot labels, for the Table 5 ablation); the log goes to
``results/hpsearch/selection.json``.

Usage (from the repository root):  PYTHONPATH=src python surj_repro/hpsearch.py [--parallel 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import RECIPES_FILE  # noqa: E402

FUN = ["cellcycle_FUN", "derisi_FUN", "eisen_FUN", "expr_FUN", "gasch1_FUN", "gasch2_FUN",
       "seq_FUN", "spo_FUN"]
BASE = "L3_surj"
STAGES = [
    ("A", [{"lr": lr, "batch": b} for lr in ("1e-3", "3e-4", "1e-4") for b in (4, 32, 128)]),
    ("B", [{"dropout": d} for d in ("0.3", "0.5", "0.7")]),
    ("C", [{"hidden": h, "gae_dim": d} for h in ("chmcnn", 512, 1024) for d in (16, 64, 128)]),
    ("D", [{"mcm": "false", "mcloss": "false"}, {"mcm": "true", "mcloss": "true"}]),
]
OUT = os.path.join(HERE, "results", "hpsearch")


def config_name(cfg: dict) -> str:
    return "hp__" + "__".join(f"{k}-{v}" for k, v in sorted(cfg.items()))


def score(cfg: dict, parallel: int) -> float:
    """Mean validation AP over FunCat for ``cfg`` (runs whatever is missing)."""
    name = config_name(cfg)
    sets = [x for k, v in sorted(cfg.items()) for x in ("--set", f"{k}={v}")]

    def one(ds):
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--recipe", BASE, *sets,
               "--set", "epochs=select_norefit", "--datasets", ds, "--seeds", "0",
               "--no_test", "--name", os.path.join("hpsearch", name)]
        with open(os.path.join(OUT, "logs", f"{name}.{ds}.log"), "a", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, FUN))
    vals = []
    for path in sorted(glob.glob(os.path.join(OUT, name, "*", "seed0.json"))):
        with open(path, encoding="utf-8") as f:
            vals.append(json.load(f)["settings"]["full"]["val_ap"])
    if len(vals) != len(FUN):
        raise RuntimeError(f"{name}: {len(vals)} of {len(FUN)} datasets finished")
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)

    best: dict = {}
    log = []
    for stage, grid in STAGES:
        results = []
        for delta in grid:
            cfg = best | delta
            s = score(cfg, args.parallel)
            results.append((s, cfg))
            log.append({"stage": stage, "config": cfg, "mean_val_ap": s})
            print(f"{stage} {config_name(cfg)}: mean val AP {s:.4f}", flush=True)
        best = max(results, key=lambda t: t[0])[1]
        print(f"-> stage {stage} best: {best}", flush=True)
        with open(os.path.join(OUT, "selection.json"), "w", encoding="utf-8") as f:
            json.dump({"base": BASE, "log": log, "best": best}, f, indent=1)

    with open(RECIPES_FILE, encoding="utf-8") as f:
        recipes = json.load(f)
    chosen = recipes[BASE]["set"] | {k: _typed(v) for k, v in best.items()}
    recipes["selected"] = {"about": "Chosen by hpsearch.py on FunCat validation AP (seed 0).",
                           "set": chosen}
    recipes["selected_onehot"] = {"about": "`selected` with one-hot labels (Table 5 ablation).",
                                  "set": chosen | {"net": "onehot"}}
    with open(RECIPES_FILE, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2)
        f.write("\n")
    print(f"selected: {chosen}")


def _typed(v):
    if isinstance(v, int):
        return v
    if v in ("true", "false"):
        return v == "true"
    if v == "chmcnn":
        return v
    return float(v)


if __name__ == "__main__":
    main()

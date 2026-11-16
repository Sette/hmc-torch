#!/usr/bin/env python3
"""Compare the reconciliation gain under two epoch-selection rules (spec 23).

``output/protocol/`` selects the epoch on the reconciled validation scores for
every arm.  ``output/protocol_controls/select_raw/`` reruns BCE selecting on
the raw scores.  The symmetric comparison takes each arm at its own epoch: the
raw arm from the control run, the reconciled arm from the protocol run.

Writes ``output/protocol_controls/summary.json``.

Usage:
    python experiments/summarize_controls.py
"""

import json
import os

import numpy as np
from scipy import stats

BASE = "output/protocol"
CTRL = "output/protocol_controls/select_raw"
EURLEX = "output/protocol_controls/eurlex_200/eurlex.json"


def arm(path, mode, metric, obj="bce"):
    """Mean over seeds of one metric of one arm."""
    with open(path, encoding="utf-8") as fh:
        runs = [r for r in json.load(fh)["runs"] if r["objective"] == obj]
    return float(np.mean([r["eval"][mode][metric] for r in runs])), [r["best_epoch"] for r in runs]


def main():
    names = sorted(f[:-5] for f in os.listdir(CTRL) if f.endswith(".json"))
    out = {"datasets": {}, "tests": {}}
    for metric in ("auprc", "f1"):
        same, own = [], []
        for n in names:
            raw_rec_epoch, ep_rec = arm(f"{BASE}/{n}.json", "raw", metric)
            rec, _ = arm(f"{BASE}/{n}.json", "rec", metric)
            raw_own_epoch, ep_raw = arm(f"{CTRL}/{n}.json", "raw", metric)
            same.append(rec - raw_rec_epoch)
            own.append(rec - raw_own_epoch)
            out["datasets"].setdefault(n, {})[metric] = {
                "delta_same_epoch": same[-1], "delta_own_epoch": own[-1],
                "epochs_rec": ep_rec, "epochs_raw": ep_raw,
            }
        for label, d in (("same_epoch", same), ("own_epoch", own)):
            d = np.array(d)
            out["tests"][f"{metric}_{label}"] = {
                "n": len(d), "positive": int((d > 0).sum()), "median": float(np.median(d)),
                "mean": float(d.mean()),
                "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(d) > 5 else None,
            }
            print(f"{metric:5s} {label:10s} n={len(d)} wins={int((d > 0).sum())} "
                  f"median={np.median(d):+.4f} mean={d.mean():+.4f}")
    if os.path.exists(EURLEX):
        out["eurlex_200"] = {}
        for obj in ("bce", "hinge", "mc"):
            for mode in ("raw", "rec"):
                f1, ep = arm(EURLEX, mode, "f1", obj)
                ap, _ = arm(EURLEX, mode, "auprc", obj)
                out["eurlex_200"][f"{obj}_{mode}"] = {"f1": f1, "auprc": ap, "epochs": ep}
            print("eurlex_200", obj, out["eurlex_200"][f"{obj}_rec"])
    with open("output/protocol_controls/summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()

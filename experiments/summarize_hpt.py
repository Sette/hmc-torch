#!/usr/bin/env python3
"""Aggregate HPT final runs (one JSON per seed) into a summary JSON.

Example:
    python experiments/summarize_hpt.py --dir output/hpt/final/wos \
        --backbone bert --loss none --graph down --selection hpt_raw
"""

import argparse
import glob
import json
import os
import statistics


def _stats(values: list[float]) -> dict:
    """Mean and sample standard deviation (0 for a single run)."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="./output/hpt/final/wos")
    parser.add_argument("--backbone", default="bert")
    parser.add_argument("--loss", default="none")
    parser.add_argument("--graph", default="down")
    parser.add_argument("--selection", default="hpt_raw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)
        if run.get("backbone") != args.backbone:
            continue
        if run.get("loss") != args.loss or run.get("graph") != args.graph:
            continue
        selection = run.get("config", {}).get("selection", "rec_val_micro")
        if selection != args.selection:
            continue
        runs.append((path, run))

    if not runs:
        raise SystemExit(
            f"no {args.backbone}/{args.loss}/{args.graph}/{args.selection} runs in {args.dir}"
        )

    summary = {
        "dataset": runs[0][1]["dataset"],
        "backbone": args.backbone,
        "loss": args.loss,
        "graph": args.graph,
        "selection": args.selection,
        "n_seeds": len(runs),
        "seeds": [run["seed"] for _, run in runs],
        "best_epochs": {str(run["seed"]): run["best_epoch"] for _, run in runs},
        "train_time_s": _stats([run["train_time_s"] for _, run in runs]),
        "splits": {
            "train": runs[0][1]["n_train"],
            "valid": runs[0][1]["n_valid"],
            "test": runs[0][1]["n_test"],
        },
        "metrics": {},
        "sources": [path for path, _ in runs],
    }
    for mode in ("raw", "rec"):
        for thr in ("thr0", "val"):
            summary["metrics"][f"{mode}/{thr}"] = {
                "f1": _stats([run[mode][thr]["f1"] for _, run in runs]),
                "macro_f1": _stats([run[mode][thr]["macro_f1"] for _, run in runs]),
                "auprc": _stats([run[mode]["auprc"] for _, run in runs]),
                "ghv_rate": _stats([run[mode]["ghv"]["rate"] for _, run in runs]),
            }

    out = args.out or os.path.join(args.dir, "summary.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=float)
    print(f"wrote {out} ({len(runs)} seeds)")
    for key, value in summary["metrics"].items():
        print(
            f"  {key}: f1 {value['f1']['mean']:.4f}±{value['f1']['std']:.4f} "
            f"macro {value['macro_f1']['mean']:.4f} auprc {value['auprc']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()

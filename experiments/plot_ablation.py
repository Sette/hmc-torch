#!/usr/bin/env python3
"""Plot ablation results: grouped bar chart of 4 configs × 3 datasets.

Reads output/ablation/results.json and generates
docs/hmc-paper/figures/fig_ablation.pdf
"""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_results(path: str) -> dict:
    """Load ablation results JSON."""
    with open(path) as f:
        return json.load(f)


def plot_ablation(results: dict, out_path: str) -> None:
    """Generate grouped bar chart comparing 4 ablation modes across datasets."""
    mode_order = ["bce_only", "consistency_loss", "reconciliation", "both"]
    mode_labels = ["BCE only", "Consistency\nloss", "Reconciliation", "Both"]
    colors = ["#b0bec5", "#ffb74d", "#4fc3f7", "#1b5e20"]

    datasets = list(results.keys())
    ds_labels: list[str] = []
    for ds in datasets:
        if "arxiv" in ds.lower():
            ds_labels.append("ArXiv\n(shallow tree)")
        elif "fun" in ds.lower():
            ds_labels.append("cellcycle_FUN\n(deep tree)")
        elif "go" in ds.lower():
            ds_labels.append("cellcycle_GO\n(DAG)")

    n_modes = len(mode_order)
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mode, label, color) in enumerate(
        zip(mode_order, mode_labels, colors)
    ):
        values: list[float] = []
        for ds in datasets:
            ds_results = {r["mode"]: r for r in results[ds]}
            values.append(ds_results[mode]["f1"])
        offset = (i - n_modes / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)

        # Annotate with value
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Micro-F1")
    ax.set_title("R-Matrix Ablation: Training vs Inference Contributions")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, max(
        max(r["f1"] for r in results[ds]) for ds in datasets
    ) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Add delta annotations between key modes
    for j, ds in enumerate(datasets):
        ds_map = {r["mode"]: r for r in results[ds]}
        rec_f1 = ds_map["reconciliation"]["f1"]
        bce_f1 = ds_map["bce_only"]["f1"]
        both_f1 = ds_map["both"]["f1"]
        delta_rec = rec_f1 - bce_f1
        delta_both = both_f1 - rec_f1
        y_base = max(bce_f1, rec_f1, both_f1) + 0.04
        ax.annotate(f"Δrec={delta_rec:+.3f}\nΔMC={delta_both:+.3f}",
                    xy=(j, y_base), ha="center", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                              alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    results_path = "./output/ablation/results.json"
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run experiments/run_ablation.py first.")
        return

    results = load_results(results_path)
    out = "docs/hmc-paper/figures/fig_ablation.pdf"
    plot_ablation(results, out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation plot: grouped bars showing 4 configs across 3 datasets.

Panel 1: ArXiv (shallow tree)
Panel 2: cellcycle_FUN (deep tree)
Panel 3: cellcycle_GO (DAG, sparse)

Highlights reconciliation as the dominant mechanism.
Output: docs/hmc-paper/figures/fig_ablation.pdf
"""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

MODE_ORDER = ["bce_only", "consistency_loss", "reconciliation", "both"]
MODE_LABELS = ["BCE\nonly", "Consistency\nloss", "Reconciliation", "Both\n(full R)"]
COLORS = ["#d9d9d9", "#f4a582", "#0571b0", "#7fbf7b"]

DATASET_CONFIG = {
    "arxiv":          {"label": "ArXiv\n(shallow tree, D=2)",    "ylim": (0.71, 0.74)},
    "cellcycle_FUN":  {"label": "cellcycle_FUN\n(deep tree, D=6)",  "ylim": (0.26, 0.29)},
    "cellcycle_GO":   {"label": "cellcycle_GO\n(DAG, D=13, sparse)", "ylim": (0.38, 0.41)},
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_ablation(results: dict, out_path: str) -> None:
    datasets = [d for d in ["arxiv", "cellcycle_FUN", "cellcycle_GO"] if d in results]
    n_ds = len(datasets)
    n_modes = len(MODE_ORDER)

    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4.5), sharey=False)

    for ax_idx, ds_name in enumerate(datasets):
        ax = axes[ax_idx] if n_ds > 1 else axes
        ds_results = {r["mode"]: r for r in results[ds_name]}
        cfg = DATASET_CONFIG.get(ds_name, {})

        x = np.arange(n_modes)
        values = []
        for mode in MODE_ORDER:
            val = ds_results[mode]["f1"] if mode in ds_results else 0.0
            values.append(val)

        bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.6)

        # Value on top
        for bar, val in zip(bars, values):
            offset = 0.0006 if ds_name == "arxiv" else 0.0008
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        # Highlight best and worst
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("#0571b0")
        bars[best_idx].set_linewidth(2.0)

        # Delta arrow: reconciliation vs bce_only
        bce_val = values[0]
        rec_val = values[2]
        delta = rec_val - bce_val
        if delta != 0:
            y_mid = (bce_val + rec_val) / 2
            ax.annotate(
                f"$\\Delta$={delta:+.4f}", xy=(2.5, y_mid),
                fontsize=8, ha="center", va="center",
                color="#0571b0" if delta > 0 else "#d73027",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(MODE_LABELS, fontsize=8)
        ax.set_title(cfg.get("label", ds_name), fontsize=10)
        ax.set_ylabel("Micro-F1")
        ax.set_ylim(*cfg.get("ylim", (0, 1)))
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("R-Matrix Decomposition: Where Does the Benefit Come From?",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.subplots_adjust(top=0.82, wspace=0.25)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    path = "./output/ablation/results.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found."); return
    results = load_results(path)
    plot_ablation(results, "docs/hmc-paper/figures/fig_ablation.pdf")


if __name__ == "__main__":
    main()

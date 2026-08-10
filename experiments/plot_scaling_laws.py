#!/usr/bin/env python3
"""Plot scaling laws: how HMC performance varies with hierarchy properties.

Reads output/scaling_laws/results.json and generates 4 plots:
  1. Micro-F1 vs n_classes (log scale), colored by modality
  2. Micro-F1 vs max_depth
  3. Train time vs n_classes × n_train
  4. n_train/n_classes ratio histogram per modality

Output: docs/hmc-paper/figures/fig_scaling_laws.pdf
"""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

MODALITY_COLORS = {
    "text": "#1b5e20",
    "tabular_tree": "#e65100",
    "tabular_dag": "#0d47a1",
    "tabular_other": "#6a1b9a",
}
MODALITY_MARKERS = {
    "text": "o",
    "tabular_tree": "s",
    "tabular_dag": "D",
    "tabular_other": "^",
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def plot_scaling_laws(results: list[dict], out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Group by modality
    modalities: dict[str, list[dict]] = {}
    for r in results:
        mod = r.get("modality", "unknown")
        modalities.setdefault(mod, []).append(r)

    # ---- 1. F1 vs n_nodes ----
    ax = axes[0, 0]
    for mod, pts in modalities.items():
        xs = [p["n_nodes"] for p in pts]
        ys = [p["f1"] for p in pts]
        labels = [p["dataset"] for p in pts]
        ax.scatter(xs, ys, c=MODALITY_COLORS.get(mod, "#999"),
                   marker=MODALITY_MARKERS.get(mod, "o"),
                   label=mod, s=60, edgecolors="white", linewidth=0.5)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Number of classes (log scale)")
    ax.set_ylabel("Micro-F1")
    ax.set_title("F1 vs Hierarchy Size")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # ---- 2. F1 vs max_depth ----
    ax = axes[0, 1]
    for mod, pts in modalities.items():
        xs = [p["max_depth"] for p in pts]
        ys = [p["f1"] for p in pts]
        ax.scatter(xs, ys, c=MODALITY_COLORS.get(mod, "#999"),
                   marker=MODALITY_MARKERS.get(mod, "o"),
                   label=mod, s=60, edgecolors="white", linewidth=0.5)
        for x, y, lbl in zip(xs, ys, [p["dataset"] for p in pts]):
            ax.annotate(lbl, (x, y), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Max hierarchy depth")
    ax.set_ylabel("Micro-F1")
    ax.set_title("F1 vs Hierarchy Depth")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- 3. Time vs n_classes * n_train ----
    ax = axes[1, 0]
    for mod, pts in modalities.items():
        xs = [p["n_nodes"] * p["n_train"] for p in pts]
        ys = [p["time_s"] for p in pts]
        ax.scatter(xs, ys, c=MODALITY_COLORS.get(mod, "#999"),
                   marker=MODALITY_MARKERS.get(mod, "o"),
                   label=mod, s=60, edgecolors="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_classes × n_train (log scale)")
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title("Training Time vs Problem Size")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- 4. n_train / n_nodes histogram ----
    ax = axes[1, 1]
    ratios_by_mod: dict[str, list[float]] = {}
    for mod, pts in modalities.items():
        ratios_by_mod[mod] = [p["ratio"] for p in pts]
    all_ratios = [r for v in ratios_by_mod.values() for r in v]
    bins = np.logspace(np.log10(max(min(all_ratios), 0.1)),
                       np.log10(max(all_ratios)), 15)
    for mod, ratios in ratios_by_mod.items():
        ax.hist(ratios, bins=bins, alpha=0.5,
                color=MODALITY_COLORS.get(mod, "#999"), label=mod)
    ax.set_xscale("log")
    ax.set_xlabel("n_train / n_classes (log scale)")
    ax.set_ylabel("Number of datasets")
    ax.set_title("Data-per-Class Ratio by Modality")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.suptitle("HMC Scaling Laws", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    results_path = "./output/scaling_laws/results.json"
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. "
              f"Run experiments/run_scaling_laws.py first.")
        return
    results = load_results(results_path)
    if not results:
        print("No results found.")
        return
    out = "docs/hmc-paper/figures/fig_scaling_laws.pdf"
    plot_scaling_laws(results, out)


if __name__ == "__main__":
    main()

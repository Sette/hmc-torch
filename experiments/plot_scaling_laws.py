#!/usr/bin/env python3
"""Scaling laws for HMC: 3-panel figure for paper.

Panel 1: Micro-F1 vs number of classes (log), colored by modality.
Panel 2: Micro-F1 vs hierarchy depth.
Panel 3: Training time vs problem size C×N (log-log) with power-law fit.

Output: docs/hmc-paper/figures/fig_scaling_laws.pdf
"""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")

# Style
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

MODALITY_STYLE = {
    "text":          {"color": "#1b7837", "marker": "o", "label": "Text (SPECTER2)"},
    "tabular_tree":  {"color": "#e66101", "marker": "s", "label": "Tabular tree (FunCat)"},
    "tabular_dag":   {"color": "#2166ac", "marker": "D", "label": "Tabular DAG (GO)"},
    "tabular_other": {"color": "#762a83", "marker": "^", "label": "Other tabular"},
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("results", [])


def plot_scaling_laws(results: list[dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    mods: dict[str, list[dict]] = {}
    for r in results:
        mods.setdefault(r.get("modality", "unknown"), []).append(r)

    # ---- Panel 1: F1 vs n_nodes (log) ----
    ax = axes[0]
    for mod, pts in mods.items():
        s = MODALITY_STYLE.get(mod, MODALITY_STYLE["text"])
        xs = [p["n_nodes"] for p in pts]
        ys = [p["f1"] for p in pts]
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], label=s["label"],
                   s=50, edgecolors="white", linewidth=0.4, zorder=5)
        # Annotate only key outliers
        for p in pts:
            if p["f1"] > 0.70 or p["n_nodes"] > 3000 or p["dataset"] in ("diatoms_others",):
                ax.annotate(p["dataset"], (p["n_nodes"], p["f1"]),
                            fontsize=5, alpha=0.8, xytext=(4, 4),
                            textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Number of classes $C$")
    ax.set_ylabel("Micro-F1")
    ax.set_title("(a) F1 vs Hierarchy Size")
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.02, 0.88)

    # ---- Panel 2: F1 vs max_depth ----
    ax = axes[1]
    for mod, pts in mods.items():
        s = MODALITY_STYLE.get(mod, MODALITY_STYLE["text"])
        xs = [p["max_depth"] for p in pts]
        ys = [p["f1"] for p in pts]
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], label=s["label"],
                   s=50, edgecolors="white", linewidth=0.4, zorder=5)
    ax.set_xlabel("Max hierarchy depth $D$")
    ax.set_ylabel("Micro-F1")
    ax.set_title("(b) F1 vs Hierarchy Depth")
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.02, 0.88)

    # ---- Panel 3: Time vs C×N (log-log) with power-law fit ----
    ax = axes[2]
    for mod, pts in mods.items():
        s = MODALITY_STYLE.get(mod, MODALITY_STYLE["text"])
        xs = [p["n_nodes"] * p["n_train"] for p in pts]
        ys = [p["time_s"] for p in pts]
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], label=s["label"],
                   s=50, edgecolors="white", linewidth=0.4, zorder=5)

    # Power-law fit
    log_x = np.log([p["n_nodes"] * p["n_train"] for p in results])
    log_y = np.log([p["time_s"] for p in results])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_fit = np.logspace(min(log_x), max(log_x), 100)
    y_fit = np.exp(intercept) * x_fit ** slope
    ax.plot(x_fit, y_fit, "k--", linewidth=1.2, alpha=0.6,
            label=f"$t \\propto (C \\times N)^{{{slope:.2f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size $C \\times N$")
    ax.set_ylabel("Training time (s)")
    ax.set_title("(c) Training Time vs Problem Size")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    path = "./output/scaling_laws/results.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found."); return
    results = load_results(path)
    if not results:
        print("No results."); return
    plot_scaling_laws(results, "docs/hmc-paper/figures/fig_scaling_laws.pdf")


if __name__ == "__main__":
    main()

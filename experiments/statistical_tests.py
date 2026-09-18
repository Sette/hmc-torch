#!/usr/bin/env python3
"""Friedman + Nemenyi tests behind the FunCat comparison (paper Table 4).

The three literature columns (WWW'22, C-HMCNN, HMCN-F) come from the
published numbers; our GBDT and Global columns are read from the per-dataset
``output/experiments/<dataset>/<method>/metrics.json`` files written by
``run_experiment_matrix.py`` (GBDT) and ``run_sota_comparison.py`` (Global).
Falls back to the paper's table when those files are missing.

Usage:
    python experiments/statistical_tests.py
"""

import json
import pathlib
import sys

import numpy as np
from scipy.stats import friedmanchisquare

sys.path.insert(0, "src")

# Literature columns, as reported in the paper's FunCat table (AUPRC).
LITERATURE = {
    "cellcycle_FUN": {"WWW22": 0.269, "C-HMCNN": 0.255, "HMCN-F": 0.252},
    "derisi_FUN": {"WWW22": 0.231, "C-HMCNN": 0.195, "HMCN-F": 0.193},
    "eisen_FUN": {"WWW22": 0.392, "C-HMCNN": 0.306, "HMCN-F": 0.298},
    "expr_FUN": {"WWW22": 0.382, "C-HMCNN": 0.302, "HMCN-F": 0.301},
    "gasch1_FUN": {"WWW22": 0.369, "C-HMCNN": 0.286, "HMCN-F": 0.284},
    "gasch2_FUN": {"WWW22": 0.273, "C-HMCNN": 0.258, "HMCN-F": 0.254},
    "seq_FUN": {"WWW22": 0.341, "C-HMCNN": 0.292, "HMCN-F": 0.291},
    "spo_FUN": {"WWW22": 0.241, "C-HMCNN": 0.215, "HMCN-F": 0.211},
}
# Paper table, used when a run is not on disk.
FALLBACK = {
    "cellcycle_FUN": {"GBDT": 0.241, "Global": 0.239},
    "derisi_FUN": {"GBDT": 0.169, "Global": 0.191},
    "eisen_FUN": {"GBDT": 0.285, "Global": 0.278},
    "expr_FUN": {"GBDT": 0.297, "Global": 0.297},
    "gasch1_FUN": {"GBDT": 0.274, "Global": 0.276},
    "gasch2_FUN": {"GBDT": 0.228, "Global": 0.239},
    "seq_FUN": {"GBDT": 0.307, "Global": 0.282},
    "spo_FUN": {"GBDT": 0.202, "Global": 0.198},
}
METHODS = ("WWW22", "C-HMCNN", "HMCN-F", "GBDT", "Global")
# Nemenyi critical value for k=5, alpha=0.05 (Demsar 2006, Table 5).
Q_ALPHA_005 = 2.728


def _measured(dataset: str) -> dict:
    """Read our AUPRC values for one dataset, falling back to the paper table."""
    values = dict(FALLBACK[dataset])
    for method, folder in (("GBDT", "tabular_gbdt"), ("Global", "global")):
        path = pathlib.Path("output/experiments") / dataset / folder / "metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            auprc = metrics.get("AUPRC", metrics.get("auprc"))
            if auprc is not None:
                values[method] = float(auprc)
    return values


def main() -> None:
    """Run the Friedman test and the Nemenyi post-hoc comparisons."""
    datasets = sorted(LITERATURE)
    table = np.array(
        [[{**LITERATURE[ds], **_measured(ds)}[m] for m in METHODS] for ds in datasets]
    )
    n_datasets, n_methods = table.shape
    print(f"{n_datasets} datasets x {n_methods} methods (AUPRC)")
    for ds, row in zip(datasets, table):
        print(f"  {ds:<15} " + "  ".join(f"{m}={v:.3f}" for m, v in zip(METHODS, row)))

    chi2, p_value = friedmanchisquare(*[table[:, i] for i in range(n_methods)])
    print(f"\nFriedman: chi2={chi2:.4f}, p={p_value:.2e}")

    # Ranks: 1 = best on that dataset
    ranks = np.array(
        [n_methods - np.argsort(np.argsort(row)) for row in table], dtype=float
    )
    average_ranks = ranks.mean(axis=0)
    cd = Q_ALPHA_005 * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    print(f"Average ranks: " + ", ".join(f"{m}={r:.2f}" for m, r in zip(METHODS, average_ranks)))
    print(f"Critical difference (Nemenyi, alpha=0.05) = {cd:.3f}\n")

    print("Pairs beyond the critical difference:")
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(average_ranks[i] - average_ranks[j])
            if diff >= cd:
                print(f"  {METHODS[i]} > {METHODS[j]}  (delta rank = {diff:.2f})")


if __name__ == "__main__":
    main()

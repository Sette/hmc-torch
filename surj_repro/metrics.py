"""Metrics on saved test scores.

``micro_ap`` over the evaluated columns is C-HMCNN's metric and the only one reported as a
result.  The others are hypotheses about how a published number could have been computed:
keeping the root columns, the trapezoidal area under the micro PR curve (``auc`` of
``precision_recall_curve``, both imported by C-HMCNN's ``main.py``), and the weighted, macro
and per-sample averages of average precision.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def micro_ap(y: np.ndarray, s: np.ndarray, mask: np.ndarray) -> float:
    return float(average_precision_score(y[:, mask], s[:, mask], average="micro"))


def trapezoid_pr_auc(y: np.ndarray, s: np.ndarray, mask: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y[:, mask].ravel(), s[:, mask].ravel())
    return float(auc(r, p))


def _ap(y, s, average):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y, s, average=average))


def hypotheses(y: np.ndarray, s: np.ndarray, mask: np.ndarray) -> dict:
    """Every candidate metric; columns/rows without a positive are dropped where undefined."""
    cols = mask & (y.sum(0) > 0)
    rows = y[:, mask].sum(1) > 0
    return {
        "micro_ap": micro_ap(y, s, mask),
        "micro_ap_with_roots": micro_ap(y, s, np.ones_like(mask)),
        "micro_trapezoid": trapezoid_pr_auc(y, s, mask),
        "weighted_ap": _ap(y[:, cols], s[:, cols], "weighted"),
        "macro_ap": _ap(y[:, cols], s[:, cols], "macro"),
        "samples_ap": _ap(y[rows][:, mask], s[rows][:, mask], "samples"),
    }

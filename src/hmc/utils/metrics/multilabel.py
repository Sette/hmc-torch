"""Thresholded multi-label metrics over a label mask (micro/macro F1, AUPRC).

``macro_f1`` follows HPT's evaluation (``HPT/eval.py``, after HiAGM): the mean
per-label F1 over *every* evaluated label, with labels that are neither gold
nor predicted in the split counting as 0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def micro_prf(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float, float]:
    """Micro precision, recall and F1 of ``s >= thr`` against ``y``."""
    pred = s >= thr
    tp = float(np.logical_and(pred, y == 1).sum())
    fp = float(np.logical_and(pred, y == 0).sum())
    fn = float(np.logical_and(~pred, y == 1).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_f1(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    """Mean per-label F1 over all columns; 0 for a label with no tp+fp+fn."""
    pred = s >= thr
    gold = y == 1
    tp = np.logical_and(pred, gold).sum(0).astype(np.float64)
    fp = np.logical_and(pred, ~gold).sum(0)
    fn = np.logical_and(~pred, gold).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean()) if f1.size else 0.0


def micro_auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Micro-averaged average precision (0 when there is no positive)."""
    return float(average_precision_score(y, s, average="micro")) if y.sum() else 0.0


def pick_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Threshold of :data:`THRESHOLDS` maximising validation micro-F1."""
    f1s = [micro_prf(y_val, s_val, t)[2] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1s))])

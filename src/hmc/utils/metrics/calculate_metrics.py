"""
Utility functions for computing classification metrics.

This module provides :func:`calculate_metrics`, a convenience wrapper around
scikit-learn's ``precision_recall_fscore_support`` and
``average_precision_score`` that returns a unified metrics dictionary.
"""

from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def calculate_metrics(y_true, y_pred, y_pred_binary) -> dict:
    """
    Compute micro-averaged precision, recall, F1-score, and average precision.

    Args:
        y_true: Ground-truth binary label matrix.
        y_pred: Continuous prediction scores (used for average precision).
        y_pred_binary: Binarised predictions (used for precision/recall/F1).

    Returns:
        dict: Keys ``"precision"``, ``"recall"``, ``"f1score"``, and
        ``"average_precision_score"``.
    """
    score = precision_recall_fscore_support(
        y_true,
        y_pred_binary,
        average="micro",
        zero_division=0,
    )

    avg_score = average_precision_score(
        y_true,
        y_pred,
        average="micro",
    )

    return {
        "precision": score[0],
        "recall": score[1],
        "f1score": score[2],
        "average_precision_score": avg_score,
    }

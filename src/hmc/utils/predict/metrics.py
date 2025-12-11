import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def find_best_thresholds(
    local_inputs, local_outputs, active_levels, grid=None, average="micro"
):
    """
    For each active level, finds the threshold that maximizes the F1-score.

    Args:
        local_inputs: dict[level] -> tensor (binary targets for that level, concatenated)
        local_outputs: dict[level] -> tensor (model outputs for that level, concatenated)
        active_levels: list of levels (ints) to optimize
        grid: iterable, list of thresholds to test (default: np.arange(0, 1.05, 0.05))
        average: averaging method for sklearn.metrics (default: "micro")
    Returns:
        dict: the best threshold for each level
    """
    if grid is None:
        grid = np.arange(0.0, 1.05, 0.05)
    best_thresholds = {}
    for idx in active_levels:
        y_true = local_inputs[idx]
        y_score = local_outputs[idx]
        best_score = 0.0
        best_th = 0.5
        for th in grid:
            y_pred = (y_score > th).astype(int)
            f1 = precision_recall_fscore_support(
                y_true, y_pred, average=average, zero_division=0
            )[2]
            if f1 > best_score:
                best_score = f1
                best_th = th
        best_thresholds[idx] = best_th
    return best_thresholds

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import os.path

import pandas as pd
from sklearn.metrics import classification_report, f1_score


def custom_optimizers(n):
    return ["SGD"] * n


def custom_thresholds(n):
    return [0.5] * n


def custom_dropouts(n):
    return [0.1] * n


def custom_lrs(n):
    return [0.001] * n


def create_report_metrics(y_pred, y_true, target_names):
    rerport = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
        target_names=target_names,
    )

    # Converter o dicionário em DataFrame
    df_report = pd.DataFrame(rerport).transpose()

    return df_report


def create_reports(results, y_true, labels, max_depth):
    fscore = [[] for _ in range(max_depth)]
    reports = {}
    for i in range(max_depth):
        level_name = f"level{i + 1}"
        y_test_bin = [label[level_name].tolist() for label in y_true]
        fscore[i].append(f1_score(results[i], y_test_bin, average="weighted"))
        reports[i] = create_report_metrics(
            results[i], y_test_bin, list(labels[level_name].keys())
        )

    return reports, fscore


def generete_md(binary_predictions, df_test, labels, path="."):
    for idx, binary_label in enumerate(binary_predictions, start=1):
        level_name = f"level{idx}"

        y_test_bin = [label[level_name] for label in df_test.labels]

        rerport = classification_report(
            y_test_bin,
            binary_label,
            target_names=list(labels[level_name].keys()),
            output_dict=True,
            zero_division=0,
        )
        # Converter o dicionário em DataFrame
        df_report = pd.DataFrame(rerport).transpose()

        markdown = df_report.to_markdown()

        file_path = os.path.join(path, f"report-{idx}.md")

        # Escrever o markdown em um arquivo
        with open(file_path, "w") as f:
            f.write(markdown)


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


def find_best_threshold(local_inputs, local_outputs, grid=None, average="micro"):
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

    y_true = local_inputs
    y_score = local_outputs
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
    best_threshold = best_th
    return best_threshold

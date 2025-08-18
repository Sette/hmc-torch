import os.path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

# def custom_thresholds(n):
#    start = 0.5
#    step = 1
#    return [start + i * step for i in range(n)]


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

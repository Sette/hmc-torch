from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def calculate_metrics(y_true, y_pred, y_pred_binary):
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

    metrics = {
        "precision": score[0],
        "recall": score[1],
        "f1score": score[2],
        "average_precision_score": avg_score,
    }

    return metrics

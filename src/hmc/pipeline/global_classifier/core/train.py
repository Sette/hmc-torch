"""
Train a global classifier
"""

import logging
import time

import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from tqdm import tqdm

from hmc.models.global_classifier.constraint.model import (
    get_constr_out,
)
from hmc.utils.dataset.labels import global_to_local_predictions
from hmc.utils.path.files import create_dir
from hmc.utils.path.output import (
    save_dict_to_json,
)
from hmc.utils.train.job import (
    find_global_best_threshold,
    log_system_info,
)


def train_step(args):
    """
    Train a global classifier
    """
    args.model = args.model.to(args.device)
    args.to_eval = args.to_eval.to(args.device)

    optimizer = torch.optim.Adam(
        args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    start_train = time.perf_counter()
    for _ in range(args.epochs):
        args.model.train()
        for _, (x, labels) in tqdm(enumerate(args.train_loader)):
            x = x.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            output = args.model(x.float())

            # MCLoss
            constr_output = get_constr_out(output, args.r_matrix)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, args.r_matrix)
            train_output = (1 - labels) * constr_output.double() + labels * train_output

            loss = criterion(train_output[:, args.to_eval], labels[:, args.to_eval])
            loss.backward()
            optimizer.step()
    args.usage = log_system_info(args.device)
    end_train = time.perf_counter()
    args.total_time = end_train - start_train
    print("Tempo de treino: %f segundos", args.total_time)
    return test_step(args)


def test_step(args):
    """
    Test a global classifier
    """

    args.model.eval()
    for i, (x, y) in enumerate(args.test_loader):
        args.model.eval()

        x = x.to(args.device)
        y = y.to(args.device)

        constrained_output = args.model(x.float())
        predicted = constrained_output.data > 0.5
        predicted = predicted.to("cpu")
        constrained_output = constrained_output.to("cpu")
        y = y.to("cpu")
        args.to_eval = args.to_eval.to("cpu")

        if i == 0:
            predicted_test = predicted
            constr_test = constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim=0)

    args.best_threshold, _ = find_global_best_threshold(
        constr_test.data[:, args.to_eval],
        y_test[:, args.to_eval],
        args,
    )

    args.y_pred_local_binary = global_to_local_predictions(
        constr_test.data > args.best_threshold,
        args.hmc_dataset.dataset_values["local_nodes_idx"],
        args.hmc_dataset.dataset_values["nodes_idx"],
    )

    args.y_test_local_binary = global_to_local_predictions(
        y_test,
        args.hmc_dataset.dataset_values["local_nodes_idx"],
        args.hmc_dataset.dataset_values["nodes_idx"],
    )

    return get_local_scores(args)


def get_local_scores(args):
    """
    Get local scores
    """
    local_test_score = {
        level: {"f1score": None, "precision": None, "recall": None}
        for level in range(len(args.y_test_local_binary))
    }

    for level, (y_test_local, y_pred_local) in enumerate(
        zip(args.y_test_local_binary, args.y_pred_local_binary)
    ):
        score = precision_recall_fscore_support(
            y_test_local,
            y_pred_local,
            average="micro",
            zero_division=0,
        )
        local_test_score[level]["precision"] = score[0]  # Precision
        local_test_score[level]["recall"] = score[1]  # Recall
        local_test_score[level]["f1score"] = score[2]  # F1-score
        logging.info("Local evaluation score:")
        logging.info(
            "Level %d Precision: %.4f, Recall: %.4f, F1-score: %.4f",
            level,
            score[0],
            score[1],
            score[2],
        )

    score = precision_recall_fscore_support(
        args.y_test[:, args.to_eval],
        args.constr_test.data[:, args.to_eval] > args.best_threshold,
        average="micro",
        zero_division=0,
    )

    local_test_score["global"] = {"f1score": None, "precision": None, "recall": None}
    local_test_score["global"]["precision"] = score[0]  # Precision
    local_test_score["global"]["recall"] = score[1]  # Recall
    local_test_score["global"]["f1score"] = score[2]  # F1-score
    local_test_score["global"]["best_threshold"] = args.best_threshold
    local_test_score["global"]["avg_precision"] = average_precision_score(
        args.y_test[:, args.to_eval],
        args.constr_test.data[:, args.to_eval],
        average="micro",
    )

    local_test_score["global"]["usage"] = args.usage
    local_test_score["global"]["training_time_seconds"] = args.total_time

    logging.info(
        "Global evaluation score with best threshold %.3f", args.best_threshold
    )
    logging.info(
        "Precision: %.4f, Recall: %.4f, F1-score: %.4f", score[0], score[1], score[2]
    )

    create_dir(args.results_path)

    save_dict_to_json(
        local_test_score,
        f"{args.results_path}/test-scores.json",
    )

    logging.info(
        "Average precision score: %.4f", local_test_score["global"]["avg_precision"]
    )

    args.score = local_test_score["global"]

    return args

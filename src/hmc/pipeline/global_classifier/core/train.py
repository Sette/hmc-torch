"""
Train a global classifier
"""

import logging
import time
from dataclasses import dataclass

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


@dataclass
class EvaluationDataDTO:
    """Data Transfer Object to carry PyTorch evaluation tensors."""

    y_test_local_binary: list
    y_pred_local_binary: list
    y_test: torch.Tensor
    constr_test_data: torch.Tensor
    to_eval: torch.Tensor


def train_step(args):
    """
    Train and evaluate a global classifier.
    """
    model = args.model.to(args.device)
    to_eval = args.to_eval.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    start_train = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for _, (x, labels) in tqdm(enumerate(args.train_loader)):
            x = x.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            output = model(x.float())

            # MCLoss
            constr_output = get_constr_out(output, args.r_matrix)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, args.r_matrix)
            train_output = (1 - labels) * constr_output.double() + labels * train_output

            loss = criterion(train_output[:, to_eval], labels[:, to_eval])
            loss.backward()
            optimizer.step()

    usage = log_system_info(args.device)
    end_train = time.perf_counter()
    total_time = end_train - start_train
    print("Tempo de treino: %f segundos", total_time)

    constr_test = None
    y_test = None
    for i, (x, y) in enumerate(args.test_loader):
        model.eval()

        x = x.to(args.device)
        y = y.to(args.device)

        constrained_output = model(x.float())
        cpu_constrained_output = constrained_output.to("cpu")
        y = y.to("cpu")
        to_eval = to_eval.to("cpu")

        if i == 0:
            constr_test = cpu_constrained_output
            y_test = y
        else:
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim=0)

    best_threshold = 0.5
    y_test = y_test.to("cpu")
    constr_test = constr_test.to("cpu")
    if args.best_threshold:
        best_threshold, _ = find_global_best_threshold(
            y_test[:, to_eval],
            constr_test.data[:, to_eval],
            args,
            mode="global",
        )

    y_pred_local_binary = global_to_local_predictions(
        constr_test.data > best_threshold,
        args.hmc_dataset.local_nodes_idx,
        args.hmc_dataset.nodes_idx,
    )

    y_test_local_binary = global_to_local_predictions(
        y_test,
        args.hmc_dataset.local_nodes_idx,
        args.hmc_dataset.nodes_idx,
    )

    local_test_score = {
        level: {"f1score": None, "precision": None, "recall": None}
        for level in range(len(y_test_local_binary))
    }
    for level, (y_test_local, y_pred_local) in enumerate(
        zip(y_test_local_binary, y_pred_local_binary)
    ):
        score = precision_recall_fscore_support(
            y_test_local,
            y_pred_local,
            average="micro",
            zero_division=0,
        )
        local_test_score[level]["precision"] = score[0]
        local_test_score[level]["recall"] = score[1]
        local_test_score[level]["f1score"] = score[2]
        logging.info("Local evaluation score:")
        logging.info(
            "Level %d Precision: %.4f, Recall: %.4f, F1-score: %.4f",
            level,
            score[0],
            score[1],
            score[2],
        )

    score = precision_recall_fscore_support(
        y_test[:, to_eval],
        constr_test.data[:, to_eval] > best_threshold,
        average="micro",
        zero_division=0,
    )

    local_test_score["global"] = {}
    local_test_score["global"]["precision"] = float(score[0])
    local_test_score["global"]["recall"] = float(score[1])
    local_test_score["global"]["f1score"] = float(score[2])
    local_test_score["global"]["best_threshold"] = float(best_threshold)
    local_test_score["global"]["avg_precision"] = float(
        average_precision_score(
            y_test[:, to_eval], constr_test.data[:, to_eval], average="micro"
        )
    )
    local_test_score["global"]["usage"] = usage
    local_test_score["global"]["training_time_seconds"] = total_time

    logging.info("Global evaluation score with best threshold %.3f", best_threshold)
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

"""
Train a global classifier
"""

import logging
import time
import argparse
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any, Dict, Union
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


# 2. Data Transfer Object (DTO) - Only for moving tensors safely
@dataclass
class EvaluationDataDTO:
    """
    Data Transfer Object to carry PyTorch evaluation tensors.
    """

    y_test_local_binary: list[torch.Tensor]
    y_pred_local_binary: list[torch.Tensor]
    y_test: torch.Tensor
    constr_test_data: torch.Tensor
    to_eval: torch.Tensor


def train_step(
    args,
):
    """
    Train a global classifier
    """
    args.model = args.model.to(args.device)
    args.to_eval = args.to_eval.to(args.device)

    optimizer = torch.optim.Adam(
        args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCELoss()

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


def test_step(
    args,
    eval_data: EvaluationDataDTO,
):
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
            args.predicted_test = predicted
            args.constr_test = constrained_output
            args.y_test = y
        else:
            args.predicted_test = torch.cat((args.predicted_test, predicted), dim=0)
            args.constr_test = torch.cat((args.constr_test, constrained_output), dim=0)
            args.y_test = torch.cat((args.y_test, y), dim=0)

    args.best_threshold, _ = find_global_best_threshold(
        args.constr_test.data[:, args.to_eval],
        args.y_test[:, args.to_eval],
        args,
    )

    args.y_pred_local_binary = global_to_local_predictions(
        args.constr_test.data > args.best_threshold,
        args.hmc_dataset.dataset_values["local_nodes_idx"],
        args.hmc_dataset.dataset_values["nodes_idx"],
    )

    args.y_test_local_binary = global_to_local_predictions(
        args.y_test,
        args.hmc_dataset.dataset_values["local_nodes_idx"],
        args.hmc_dataset.dataset_values["nodes_idx"],
    )

    return get_local_scores(args)


def get_local_scores(
    config: argparse.Namespace,
    eval_data: EvaluationDataDTO,
):
    """
    Get local scores
    """
    local_test_score: Dict[Union[int, str], Any] = {
        level: {
            "f1score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        for level in range(len(eval_data.y_test_local_binary))
    }

    for level, (y_test_local, y_pred_local) in enumerate(
        zip(eval_data.y_test_local_binary, eval_data.y_pred_local_binary)
    ):
        score = precision_recall_fscore_support(
            y_test_local,
            y_pred_local,
            average="micro",
            zero_division=0,
        )

        # PREVENÇÃO: Envolver em float() para evitar erro de serialização com numpy.float64
        local_test_score[level]["precision"] = float(score[0])  # Precision
        local_test_score[level]["recall"] = float(score[1])  # Recall
        local_test_score[level]["f1score"] = float(score[2])  # F1-score

        logging.info("Local evaluation score:")
        logging.info(
            "Level %d Precision: %.4f, Recall: %.4f, F1-score: %.4f",
            level,
            score[0],
            score[1],
            score[2],
        )

    y_true_global = eval_data.y_test[:, eval_data.to_eval].cpu().numpy()
    y_pred_global = (
        (eval_data.constr_test.data[:, eval_data.to_eval] > config.best_threshold)
        .cpu()
        .numpy()
    )

    score = precision_recall_fscore_support(
        y_true_global,
        y_pred_global,
        average="micro",
        zero_division=0,
    )

    # score = precision_recall_fscore_support(
    #     args.y_test[:, args.to_eval.to("cpu")],
    #     args.constr_test.data[:, args.to_eval.to("cpu")] > args.best_threshold,
    #     average="micro",
    #     zero_division=0,
    # )

    local_test_score["global"] = {}

    local_test_score["global"]["precision"] = float(score[0])  # Precision
    local_test_score["global"]["recall"] = float(score[1])  # Recall
    local_test_score["global"]["f1score"] = float(score[2])  # F1-score
    local_test_score["global"]["best_threshold"] = float(args.best_threshold)

    local_test_score["global"]["avg_precision"] = float(
        average_precision_score(
            y_true_global,
            y_pred_global,
            average="micro",
        )
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

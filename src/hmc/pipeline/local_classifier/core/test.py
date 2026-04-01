"""
This module contains the test step functions HMC local classifier.
"""

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from hmc.utils.dataset.labels import local_to_global_predictions
from hmc.utils.metrics.calculate_metrics import calculate_metrics
from hmc.utils.path.output import save_dict_to_json


def find_local_best_threshold(
    local_outputs,
    local_inputs,
    args,
):
    """
    Find the best threshold for local predictions.
    Args:
        local_outputs: Array of local predictions.
        local_inputs: Array of local targets.
        args: Object containing dataset information.
    Returns:
        Tuple of best threshold and best scores.
    """
    if args.best_threshold:
        logging.info("find best theshold")
        best_thresholds = {level: 0 for _, level in enumerate(args.active_levels)}
        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            level: {
                "precision": 0,
                "recall": 0,
                "f1score": 0,
                "average_precision_score": 0,
            }
            for _, level in enumerate(args.active_levels)
        }
        logging.info("Evaluating %d active levels...", len(args.active_levels))
        for level in args.active_levels:
            y_pred = local_outputs[level].to("cpu").numpy()
            y_true = local_inputs[level].to("cpu").int().numpy()
            for actual_threshold in thresholds:
                y_pred_binary = y_pred > actual_threshold
                metrics = calculate_metrics(y_true, y_pred, y_pred_binary)

                if metrics["f1score"] > best_scores[level]["f1score"]:
                    best_thresholds[level] = actual_threshold
                    best_scores[level] = metrics

        logging.info("Best thresholds per level:")
        for idx in args.active_levels:
            logging.info(
                "Level %d: threshold=%.2f, precision=%.4f,"
                + "recall=%.4f, f1-score=%.4f avg score=%.4f",
                idx,
                best_thresholds[idx],
                best_scores[idx]["precision"],
                best_scores[idx]["recall"],
                best_scores[idx]["f1score"],
                best_scores[idx]["average_precision_score"],
            )
    else:
        best_thresholds = {level: 0.5 for _, level in enumerate(args.active_levels)}
    return best_thresholds, best_scores


def find_best_threshold_global(
    all_y_pred,
    y_true_global_original,
    args,
):
    """
    Find the best threshold for global predictions.
    Args:
        all_y_pred: Array of global predictions.
        y_true_global_original: Array of global targets.
        args: Object containing dataset information.
    Returns:
        Tuple of best threshold and best scores.
    """
    # Concat global targets
    best_threshold = 0.5
    best_scores = {
        "precision": 0,
        "recall": 0,
        "f1score": 0,
        "average_precision_score": 0,
    }
    if args.best_threshold:
        logging.info("finding best threshold")

        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            y_pred_global, y_pred_global_binary = local_to_global_predictions(
                all_y_pred,
                args.hmc_dataset.local_nodes_idx,
                args.hmc_dataset.nodes_idx,
                threshold=actual_threshold,
            )
            metrics = calculate_metrics(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global[:, args.hmc_dataset.to_eval],
                y_pred_global_binary[:, args.hmc_dataset.to_eval],
            )
            if metrics["f1score"] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1score": metrics["f1score"],
                }

        thresholds = np.linspace(best_threshold - 0.01, best_threshold, 10)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
            "average_precision_score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            y_pred_global, y_pred_global_binary = local_to_global_predictions(
                all_y_pred,
                args.hmc_dataset.local_nodes_idx,
                args.hmc_dataset.nodes_idx,
                threshold=actual_threshold,
            )
            metrics = calculate_metrics(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global[:, args.hmc_dataset.to_eval],
                y_pred_global_binary[:, args.hmc_dataset.to_eval],
            )
            if metrics["f1score"] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1score": metrics["f1score"],
                    "average_precision_score": metrics["average_precision_score"],
                }

        logging.info("Best threshold: %.2f", best_threshold)
        logging.info("Best scores: %s", best_scores)

    return best_threshold, best_scores


def test_step(args):
    """
    Evaluates the model on the test dataset for each active level and \
        saves the results.
    Args:
        args: An object containing the following attributes:
            - model: The trained model to evaluate.
            - test_loader: DataLoader providing test data batches \
                (inputs, targets, global_targets).
            - device: The device (CPU or CUDA) to run computations on.
            - active_levels: Iterable of indices indicating which \
                levels to evaluate.
            - dataset_name: Name of the dataset (used for saving results).
            - hmc_dataset: Dataset object containing hierarchical information \
                (optional, for global evaluation).
    Returns:
        None. The function saves the evaluation results as a JSON file in \
            'results/train' directory.
    Side Effects:
        - Logs evaluation progress and results.
        - Saves local test scores (precision, recall, f-score, support) for \
            each active level to a JSON file.
    """
    args.model.to(args.device)
    args.model.eval()

    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    for level in args.active_levels:
        args.model.levels[str(level)].load_state_dict(
            torch.load(
                os.path.join(args.results_path, f"best_model_level_{level}.pth"),
                weights_only=True,
            )
        )

    y_true_global = []
    all_y_pred = []
    with torch.no_grad():
        for inputs, targets, global_targets in args.test_loader:
            inputs = inputs.to(args.device)
            targets = [target.to(args.device).float() for target in targets]
            global_targets = global_targets.to("cpu")
            outputs = args.model(inputs.float())

            for index in args.active_levels:
                all_y_pred.append(outputs[index].to("cpu").numpy())
                local_inputs[index].append(targets[index].to("cpu"))
                local_outputs[index].append(outputs[index].to("cpu"))
            y_true_global.append(global_targets)
        # Concat all outputs and targets by level
    local_inputs = {
        level: torch.cat(local_input, dim=0)
        for level, local_input in local_inputs.items()
    }
    local_outputs = {
        key: torch.cat(outputs, dim=0) for key, outputs in local_outputs.items()
    }
    # Get local scores

    local_best_thresholds, local_score = find_local_best_threshold(
        local_outputs,
        local_inputs,
        args,
    )

    y_true_global = torch.cat(y_true_global, dim=0).numpy()
    global_best_threshold, global_score = find_best_threshold_global(
        all_y_pred,
        y_true_global,
        args,
    )
    local_score["global"] = global_score
    local_score["usage"] = args.usage
    local_score["training_time_seconds"] = args.training_time_seconds

    args.score = local_score["f1score"]  # F1-score

    local_score["metadata"] = {
        "dataset": args.dataset_name,
        "job_id": args.job_id,
        "method": args.method,
        "epochs_to_evaluate": args.epochs_to_evaluate,
        "threshold": local_best_thresholds,
        "global_threshold": global_best_threshold,
        "batch_size": args.batch_size,
        "max_depth": args.max_depth,
        "epochs": args.epochs,
        "active_levels": args.active_levels,
        "lr_values": args.lr_values,
        "weight_decay_values": args.weight_decay_values,
        "dropout_values": args.dropout_values,
        "hidden_dims": args.hidden_dims,
        "num_layers_values": args.num_layers_values,
        "seed": args.seed,
    }

    save_dict_to_json(
        local_score,
        f"{args.results_path}/{args.job_id}.json",
    )

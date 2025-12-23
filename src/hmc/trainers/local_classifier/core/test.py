import logging
import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

from hmc.utils.dataset.labels import local_to_global_predictions
from hmc.utils.path.output import (
    save_dict_to_json,
)

from tqdm import tqdm


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
    with torch.no_grad():
        for inputs, targets, global_targets in args.test_loader:
            inputs = inputs.to(args.device)
            targets = [target.to(args.device).float() for target in targets]
            global_targets = global_targets.to("cpu")
            outputs = args.model(inputs.float())

            for index in args.active_levels:
                output = outputs[index].to("cpu")
                target = targets[index].to("cpu")
                local_inputs[index].append(target)
                local_outputs[index].append(output)
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
    local_test_score = {
        level: {"f1score": None, "precision": None, "recall": None}
        for _, level in enumerate(args.active_levels)
    }

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

                precision = score[0]
                recall = score[1]
                f1_score = score[2]

                if f1_score > best_scores[level]["f1score"]:
                    best_thresholds[level] = actual_threshold
                    best_scores[level] = {
                        "precision": precision,
                        "recall": recall,
                        "f1score": f1_score,
                        "average_precision_score": avg_score,
                    }

        logging.info("Best thresholds per level:")
        for idx in args.active_levels:
            logging.info(f"Level {idx}: threshold={best_thresholds[idx]:.2f}, ")
    else:
        best_thresholds = {level: 0.5 for _, level in enumerate(args.active_levels)}

    all_y_pred = []
    logging.info("Evaluating %d active levels...", len(args.active_levels))
    for idx in args.active_levels:
        y_pred = local_outputs[idx].to("cpu").numpy()
        all_y_pred.append(y_pred)
        y_pred_binary = y_pred > best_thresholds[idx]

        score = precision_recall_fscore_support(
            local_inputs[idx],
            y_pred_binary,
            average="micro",
            zero_division=0,
        )

        avg_score = average_precision_score(
            local_inputs[idx],
            y_pred,
            average="micro",
        )

        # score = average_precision_score(
        #     local_inputs[idx], y_pred_binary, average="micro"
        # )
        local_test_score[idx]["precision"] = score[0]  # Precision
        local_test_score[idx]["recall"] = score[1]  # Recall
        local_test_score[idx]["f1score"] = score[2]  # F1-score
        local_test_score[idx]["avg_precision_score"] = avg_score

    # Save the trained model
    # torch.save(
    #     args.model.state_dict(),
    #     f"results/train/{args.dataset_name}-{job_id}-state_dict.pt",
    # )
    # args.model.save(f"results/train/{args.dataset_name}-{job_id}.pt")

    # Concat global targets
    y_true_global_original = torch.cat(y_true_global, dim=0).numpy()
    best_threshold = 0.5
    if args.best_threshold:
        logging.info("finding best threshold")

        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
            "average_precision_score": 0,
        }

        for actual_threshold in thresholds:
            y_pred_global, y_pred_global_binary = local_to_global_predictions(
                all_y_pred,
                args.hmc_dataset.local_nodes_idx,
                args.hmc_dataset.nodes_idx,
                threshold=actual_threshold,
            )
            score = precision_recall_fscore_support(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global_binary[:, args.hmc_dataset.to_eval],
                average="micro",
                zero_division=0,
            )
            logging.info("Global evaluation score:")
            logging.info(
                "Precision: %.4f, Recall: %.4f, F1-score: %.4f",
                score[0],
                score[1],
                score[2],
            )

            avg_score = average_precision_score(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global[:, args.hmc_dataset.to_eval],
                average="micro",
            )

            if score[2] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": score[0],
                    "recall": score[1],
                    "f1score": score[2],
                    "average_precision_score": avg_score,
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
            score = precision_recall_fscore_support(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global_binary[:, args.hmc_dataset.to_eval],
                average="micro",
                zero_division=0,
            )

            logging.info("Global evaluation score:")
            logging.info(
                "Precision: %.4f, Recall: %.4f, F1-score: %.4f",
                score[0],
                score[1],
                score[2],
            )

            avg_score = average_precision_score(
                y_true_global_original[:, args.hmc_dataset.to_eval],
                y_pred_global[:, args.hmc_dataset.to_eval],
                average="micro",
            )

            if score[2] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": score[0],
                    "recall": score[1],
                    "f1score": score[2],
                    "average_precision_score": avg_score,
                }

    y_pred_global, y_pred_global_binary = local_to_global_predictions(
        all_y_pred,
        args.hmc_dataset.local_nodes_idx,
        args.hmc_dataset.nodes_idx,
        threshold=best_threshold,
    )

    score = precision_recall_fscore_support(
        y_true_global_original[:, args.hmc_dataset.to_eval],
        y_pred_global_binary[:, args.hmc_dataset.to_eval],
        average="micro",
        zero_division=0,
    )
    logging.info("Global evaluation score with best threshold %.3f", best_threshold)
    logging.info(
        "Precision: %.4f, Recall: %.4f, F1-score: %.4f", score[0], score[1], score[2]
    )

    avg_score = average_precision_score(
        y_true_global_original[:, args.hmc_dataset.to_eval],
        y_pred_global[:, args.hmc_dataset.to_eval],
        average="micro",
    )

    logging.info("Average precision score: %.4f", avg_score)

    local_test_score["global"] = {
        "precision": score[0],
        "recall": score[1],
        "f1score": score[2],
        "avgscore": avg_score,
    }

    args.score = score[2]  # F1-score

    local_test_score["metadata"] = {
        "dataset": args.dataset_name,
        "job_id": args.job_id,
        "method": args.method,
        "epochs_to_evaluate": args.epochs_to_evaluate,
        "threshold": best_thresholds,
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
    # logging.info("Local test score: %s", str(local_test_score))

    save_dict_to_json(
        local_test_score,
        f"{args.results_path}/{args.job_id}.json",
    )

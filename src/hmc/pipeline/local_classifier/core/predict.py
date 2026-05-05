"""
This module contains the test step functions HMC local classifier.
"""

import os

import torch

from hmc.utils.path.output import save_dict_to_json
from hmc.utils.train.job import find_global_best_threshold, find_local_best_threshold


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
        local_inputs,
        local_outputs,
        args,
    )

    y_true_global = torch.cat(y_true_global, dim=0).numpy()
    global_best_threshold, global_score = find_global_best_threshold(
        all_y_pred,
        y_true_global,
        args,
    )
    local_score["global"] = global_score
    local_score["usage"] = args.usage
    local_score["training_time_seconds"] = args.training_time_seconds

    args.score = local_score["f1score"]  # F1-score

    local_score["metadata"] = {
        "dataset": args.dataset.dataset_name,
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

import logging
import os
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
)

from hmc.utils.output import (
    save_dict_to_json,
)

from hmc.utils.labels import local_to_global_predictions


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

    args.model.eval()

    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    for level in args.active_levels:
        args.model.levels[str(level)].load_state_dict(
            torch.load(os.path.join(args.results_path, f"best_model_level_{level}.pth"))
        )

    threshold = 0.2
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
    all_y_pred_binary = []
    all_y_pred = []
    logging.info("Evaluating %d active levels...", len(args.active_levels))
    for idx in args.active_levels:
        y_pred = local_outputs[idx].to("cpu").numpy()
        all_y_pred.append(y_pred)
        y_pred_binary = y_pred > threshold

        all_y_pred_binary.append(y_pred_binary)
        # y_pred_binary = (local_outputs[idx] > threshold).astype(int)

        score = precision_recall_fscore_support(
            local_inputs[idx], y_pred_binary, average="micro"
        )

        # score = average_precision_score(
        #     local_inputs[idx], y_pred_binary, average="micro"
        # )
        local_test_score[idx]["precision"] = score[0]  # Precision
        local_test_score[idx]["recall"] = score[1]  # Recall
        local_test_score[idx]["f1score"] = score[2]  # F1-score

    local_test_score["metadados"] = {
        "dataset": args.dataset_name,
        "job_id": args.job_id,
        "method": args.method,
        "early_metric": args.epochs_to_evaluate,
    }
    logging.info("Local test score: %s", str(local_test_score))

    save_dict_to_json(
        local_test_score,
        f"{args.results_path}/{args.job_id}.json",
    )

    # Save the trained model
    # torch.save(
    #     args.model.state_dict(),
    #     f"results/train/{args.dataset_name}-{job_id}-state_dict.pt",
    # )
    # args.model.save(f"results/train/{args.dataset_name}-{job_id}.pt")

    # Concat global targets
    y_true_global_original = torch.cat(y_true_global, dim=0).numpy()

    y_pred_global_binary = local_to_global_predictions(
        all_y_pred_binary,
        args.hmc_dataset.train.local_nodes_idx,
        args.hmc_dataset.train.nodes_idx,
    )

    score = precision_recall_fscore_support(
        y_true_global_original[:, args.hmc_dataset.train.to_eval],
        y_pred_global_binary[:, args.hmc_dataset.train.to_eval],
        average="micro",
        zero_division=0,
    )
    logging.info("Global evaluation score:")
    logging.info(
        "Precision: %.4f, Recall: %.4f, F1-score: %.4f", score[0], score[1], score[2]
    )

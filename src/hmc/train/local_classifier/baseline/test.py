import logging

import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

from hmc.train.utils import (
    create_job_id_name,
    save_dict_to_json,
)
from hmc.utils.dir import create_dir


def test_step(args):
    """
    Evaluates the model on the test dataset for each active level and saves the results.
    Args:
        args: An object containing the following attributes:
            - model: The trained model to evaluate.
            - test_loader: DataLoader providing test data batches (inputs, targets, global_targets).
            - device: The device (CPU or CUDA) to run computations on.
            - active_levels: Iterable of indices indicating which levels to evaluate.
            - dataset_name: Name of the dataset (used for saving results).
            - hmc_dataset: Dataset object containing hierarchical information (optional, for global evaluation).
    Returns:
        None. The function saves the evaluation results as a JSON file in the 'results/train' directory.
    Side Effects:
        - Logs evaluation progress and results.
        - Saves local test scores (precision, recall, f-score, support) for each active level to a JSON file.
    """

    args.model.eval()
    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    threshold = 0.2

    Y_true_global = []
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
        Y_true_global.append(global_targets)
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

    logging.info("Evaluating %d active levels...", len(args.active_levels))
    for idx in args.active_levels:
        args.model.levels[str(idx)].load_state_dict(
            torch.load(f"best_model_baseline_level_{idx}.pth")
        )
        y_pred_binary = local_outputs[idx].data > threshold

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

    logging.info("Local test score: %s", str(local_test_score))

    job_id = create_job_id_name(prefix="test")

    create_dir("results/train")

    save_dict_to_json(
        local_test_score,
        f"results/train/{args.dataset_name}-{job_id}.json",
    )

    # Save the trained model
    # torch.save(
    #     args.model.state_dict(),
    #     f"results/train/{args.dataset_name}-{job_id}-state_dict.pt",
    # )
    # args.model.save(f"results/train/{args.dataset_name}-{job_id}.pt")

    # Concat global targets
    # Y_true_global_original = torch.cat(Y_true_global, dim=0).numpy()

    # Y_pred_global = local_to_global_predictions(
    #     local_outputs,
    #     args.hmc_dataset.train.local_nodes_idx,
    #     args.hmc_dataset.train.nodes_idx,
    # )

    # Y_true_global_converted = local_to_global_predictions(
    #     local_inputs, train.local_nodes_idx, train.nodes_idx
    # )

    # score = average_precision_score(
    #     Y_true_global_original[:, to_eval], Y_pred_global[:, to_eval], average="micro"
    # )

    # logging.info(score)

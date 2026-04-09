"""
This module contains the validation step for the HMC local classifier.
"""

import logging

import torch

from hmc.utils.metrics.calculate_metrics import calculate_metrics
from hmc.utils.train.early_stopping import (
    check_early_stopping_normalized,
)
from hmc.utils.train.losses import compute_loss


def validate_step(args):
    """
    Performs a validation step for a hierarchical multi-level classifier model.
    Args:
        args: An object containing all necessary arguments and attributes, including:
            - model: The model to evaluate, with attribute `levels` for each depth.
            - val_loader: DataLoader for the validation dataset.
            - criterions: List of loss functions, one per level.
            - device: Device to run computations on.
            - max_depth: Maximum number of levels in the hierarchy.
            - active_levels: List of indices for currently active levels.
            - best_val_loss: List of best validation losses per level.
            - best_model: List to store the best model state_dict per level.
            - patience_counters: List of patience counters for early \
                stopping per level.
            - early_stopping_patience: Number of epochs to wait before \
                early stopping.
            - level_active: List indicating if a level is still active.
    Returns:
        tuple:
            - local_val_losses (list of float): Average validation loss per level.
            - local_val_precision (list of float): Average precision score per level.
    Side Effects:
        - Updates `args.best_val_loss`, `args.best_model`, and \
            `args.patience_counters` for improved levels.
        - Freezes parameters of levels that triggered early stopping by \
            setting `requires_grad` to False.
        - Logs progress and early stopping events.
    """

    args.model.eval()

    args.result_path = (
        f"{args.output_path}/train/{args.method}-{args.dataset_name}/{args.job_id}"
    )

    threshold = 0.5

    # Get local scores
    args.local_val_scores = [0.0] * args.max_depth
    args.local_val_losses = [0.0] * args.max_depth

    with torch.no_grad():
        for batch in args.val_loader:
            loss_dict = compute_loss(
                batch,
                args,
                step="valid",
            )

    for level in args.active_levels:
        if args.level_active[level]:
            y_pred = loss_dict["local_outputs"][level].to("cpu").numpy()
            y_true = loss_dict["local_inputs"][level].to("cpu").int().numpy()
            y_pred_binary = y_pred > threshold

            metrics = calculate_metrics(y_true, y_pred, y_pred_binary)

            # local_val_score[idx] = score
            logging.info(
                "Level %d: precision=%.4f, recall=%.4f, f1-score=%.4f avg score=%.4f",
                level,
                metrics["precision"],
                metrics["recall"],
                metrics["f1score"],
                metrics["average_precision_score"],
            )

            if args.early_metric == "f1-score":
                args.local_val_scores[level] = metrics["f1score"]
            elif args.early_metric == "avg-score":
                args.local_val_scores[level] = metrics["average_precision_score"]

    args.local_val_losses = [
        loss / len(args.val_loader) for loss in args.local_val_losses
    ]
    check_early_stopping_normalized(args, args.active_levels)

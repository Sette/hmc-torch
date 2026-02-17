import logging
import os

import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from hmc.utils.train.early_stopping import (
    check_early_stopping_tabat,
)
from hmc.utils.train.losses import calculate_local_loss

def compute_loss(model, batch, criterion, device):
    x = batch[0].float().to(device)
    inputs = batch[1]

    logits, attn_weights = model(x)
    local_losses = {}
    local_outputs = {}
    local_inputs = {}

    loss = 0.0
    for level_idx, logits in logits.items():
        # targets multi-label em float (0/1)
        y = inputs[level_idx].to(device).float()

        loss_level = criterion(logits, y)
        local_outputs[level_idx] = logits
        local_inputs[level_idx] = y
        local_losses[level_idx] = loss_level
        loss = loss + loss_level


    return loss, attn_weights, local_losses, local_inputs, local_outputs

def valid_local_tabat(args):
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

    args.result_path = "%s/train/%s-%s/%s" % (
        args.output_path,
        args.method,
        args.dataset_name,
        args.job_id,
    )

    local_inputs = {
        level: torch.tensor([]) for _, level in enumerate(args.active_levels)
    }
    local_outputs = {
        level: torch.tensor([]) for _, level in enumerate(args.active_levels)
    }

    threshold = 0.5

    # Get local scores
    args.local_val_scores = [0.0] * args.max_depth
    args.local_val_losses = [0.0] * args.max_depth

    with torch.no_grad():
        for batch in args.val_loader:
            loss, attn_weights, local_losses, local_inputs, local_outputs = compute_loss(args.model, batch, args.criterion, args.device)


    for level in args.active_levels:
        if args.level_active[level]:
            y_pred = local_outputs[level].to("cpu").numpy()
            y_true = local_inputs[level].to("cpu").int().numpy()
            y_pred_binary = y_pred > threshold

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

            # local_val_score[idx] = score
            logging.info(
                "Level %d: precision=%.4f, recall=%.4f, f1-score=%.4f avg score=%.4f",
                level,
                score[0],
                score[1],
                score[2],
                avg_score,
            )

            if args.early_metric == "f1-score":
                args.local_val_scores[level] = score[2]
            elif args.early_metric == "avg-score":
                args.local_val_scores[level] = avg_score

    args.local_val_losses = [
        loss / len(args.val_loader) for loss in args.local_val_losses
    ]
    check_early_stopping_tabat(args, args.active_levels)
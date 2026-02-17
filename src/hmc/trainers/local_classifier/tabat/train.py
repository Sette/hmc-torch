"""
Local classifier training module for hierarchical multi-class classification.

This module provides the core training functionality for HMC (Hierarchical
Multi-Class) local classifier models. It implements a progressive training
approach where levels of the hierarchy are activated incrementally during
the training process, with support for early stopping and validation
monitoring at each level.

The training process includes:
- Progressive level activation with warm-up epochs
- Individual optimizer management for each hierarchy level
- Per-level loss computation and early stopping
- Periodic validation evaluation
- Comprehensive logging and monitoring

Functions:
    train_step: Main training loop for hierarchical multi-class local classifier.
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim

from hmc.trainers.local_classifier.core.valid import valid_step
from hmc.utils.dataset.labels import show_local_losses
from hmc.utils.train.job import (
    create_job_id_name,
    end_timer,
    start_timer,
)
from hmc.utils.train.losses import calculate_hierarchical_local_loss

def compute_loss(model, batch, criterion, device):
    x = batch[0].float().to(device)
    inputs = batch[1]

    logits, attn_weights = model(x)
    local_losses = {}
    local_outputs = {}

    loss = 0.0
    for level_idx, logits in logits.items():
        # targets multi-label em float (0/1)
        y = inputs[level_idx].to(device).float()
        loss_level = criterion(logits, y)
        local_outputs[level_idx] = logits
        local_losses[level_idx] = loss_level
        loss = loss + loss_level


    return loss, attn_weights, local_losses, local_outputs



def train_local_tabat(args):
    """
    Executes the training loop for a hierarchical multi-class (HMC) local \
        classifier model.
    This function performs the following steps:
    - Moves the model and loss criterion to the specified device.
    - Initializes early stopping parameters and tracking variables for \
        each level of the hierarchy.
    - Sets up optimizers for each model level with individual learning rates \
        and weight decays.
    - Iterates over the specified number of epochs, performing:
        - Training over batches: forward pass, loss computation for active \
            levels, and gradient accumulation.
        - Backward pass and optimizer step for each level.
        - Logging of training losses.
        - Periodic evaluation on the validation set, including loss\
            and precision reporting.
        - Early stopping if all levels have triggered it.
    Args:
        args: An object containing all necessary training parameters and \
            objects, including:
            - model: The hierarchical model with per-level submodules.
            - criterion_list: List of loss functions for each level.
            - device: Device to run computations on.
            - hmc_dataset: Dataset object with max_depth attribute.
            - active_levels: List of currently active levels for training.
            - max_depth: Maximum depth of the hierarchy.
            - lr_values: List of learning rates for each level.
            - weight_decay_values: List of weight decay values for each level.
            - epochs: Number of training epochs.
            - train_loader: DataLoader for training data.
            - epochs_to_evaluate: Frequency of validation evaluation.
            - Additional attributes used for logging and early stopping.
    """

    args.model = args.model.to(args.device)
    args.criterion_list = [criterion.to(args.device) for criterion in args.criterion_list]

    args.early_stopping_patience = args.patience
    args.early_stopping_patience_score = args.patience_score
    # if args.early_metric == "f1-score":
    #     args.early_stopping_patience = 20
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    args.patience_counters_score = [0] * args.hmc_dataset.max_depth
    args.level_active = [level in args.active_levels for level in range(args.max_depth)]
    logging.info("Active levels: %s", args.active_levels)
    logging.info("Level active: %s", args.level_active)

    args.best_val_loss = [float("inf")] * args.max_depth
    args.best_val_score = [0.0] * args.max_depth
    args.best_model = [None] * args.max_depth
    args.job_id = create_job_id_name(prefix="test")
    logging.info("Best val loss created %s", args.best_val_loss)

    # Loss multi-label por nível
    args.criterion = nn.BCEWithLogitsLoss()

    # Pesos por nível (pode ajustar depois)
    args.level_weights = {k: 1.0 for k in args.model.levels.keys()}

    # Peso global para FunCat vs GO (ex.: dar mais peso a GO)
    args.lambda_funcat = 1.0

    args.optimizer = optim.Adam(args.model.parameters(), lr=1e-3)

    args.model.train()

    # args.r = args.hmc_dataset.R.to(args.device)
    if args.warmup:
        args.level_active = [False] * len(args.level_active)
        args.level_active[0] = True
        next_level = 1
        logging.info(
            "Using %s with %d warm-up epochs", args.parent_conditioning, args.n_warmup_epochs
        )
    else:
        next_level = len(args.active_levels)

    start = start_timer()
    for epoch in range(1, args.epochs + 1):
        args.epoch = epoch
        logging.info(
            "Level active: %s",
            [level for level, level_bool in enumerate(args.level_active) if level_bool],
        )

        for batch in args.train_loader:
            
            args.optimizer.zero_grad()

            loss, attn_w, local_losses, local_outputs = compute_loss(args.model, batch, args.criterion, args.device)

            loss.backward()
            args.optimizer.step()


        logging.info("Epoch %d/%d", epoch, args.epochs)

        if epoch % args.epochs_to_evaluate == 0:
            args.train_methods["valid_step"](args)
            if not any(args.level_active):
                logging.info("All levels have triggered early stopping.")
                args.total_time = end_timer(start)
                break

        if epoch % args.n_warmup_epochs == 0 and next_level < args.max_depth:
            if next_level < args.max_depth:
                args.level_active[next_level] = True
                logging.info("Activating level %d", next_level)
                next_level += 1
                args.n_warmup_epochs += args.n_warmup_epochs_increment
    args.total_time = end_timer(start)

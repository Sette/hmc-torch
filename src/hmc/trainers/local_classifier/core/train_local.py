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

from hmc.trainers.local_classifier.core.valid_local import valid_step
from hmc.utils.dataset.labels import show_local_losses
from hmc.utils.train.job import (
    create_job_id_name,
    end_timer,
    start_timer,
)
from hmc.utils.train.losses import calculate_local_loss


def train_step(args):
    """
    Executes the training loop for a hierarchical multi-class (HMC) local \
        classifier model.
    This function performs the following steps:
    - Moves the model and loss criterions to the specified device.
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
            - criterions: List of loss functions for each level.
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
    args.criterions = [criterion.to(args.device) for criterion in args.criterions]

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

    args.optimizers = [
        torch.optim.Adam(
            args.model.levels[str(level)].parameters(),
            lr=args.lr_values[level],
            weight_decay=args.weight_decay_values[level],
        )
        for level in range(args.hmc_dataset.max_depth)
    ]

    args.model.train()

    args.r = args.hmc_dataset.r.to(args.device)
    if (
        args.warmup
        or args.model_regularization == "soft"
        or args.model_regularization == "residual"
    ):
        args.level_active = [False] * len(args.level_active)
        args.level_active[0] = True
        next_level = 1
        logging.info(
            "Using soft regularization with %d warm-up epochs", args.n_warmup_epochs
        )
    else:
        next_level = len(args.active_levels)

    start = start_timer()
    for epoch in range(1, args.epochs + 1):
        args.epoch = epoch
        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
        logging.info(
            "Level active: %s",
            [level for level, level_bool in enumerate(args.level_active) if level_bool],
        )

        for inputs, targets, _ in args.train_loader:
            inputs = inputs.to(args.device)
            targets = [target.to(args.device) for target in targets]
            outputs = args.model(inputs.float())

            for optimizer in args.optimizers:
                optimizer.zero_grad()

            total_loss = 0.0

            for level in args.active_levels:
                if args.level_active[level]:
                    args.current_level = level
                    loss = calculate_local_loss(
                        outputs[level],
                        targets[level],
                        args,
                    )

                    local_train_losses[level] += loss.item()
                    total_loss += loss

            total_loss.backward()

            for level, optimizer in enumerate(args.optimizers):
                if args.level_active[level]:
                    optimizer.step()

        for level, local_train_loss in enumerate(local_train_losses):
            if args.level_active[level]:
                local_train_losses[level] = local_train_loss / len(args.train_loader)

        logging.info("Epoch %d/%d", epoch, args.epochs)
        show_local_losses(local_train_losses, dataset="Train")

        if epoch % args.epochs_to_evaluate == 0:
            valid_step(args)
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

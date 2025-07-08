import logging

import torch

from hmc.train.local_classifier.baseline.valid import valid_step
from hmc.train.utils import (
    show_global_loss,
    show_local_losses,
    show_local_score,
)

from hmc.model.local_classifier.constrained.utils import get_constr_out

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def train_step(args):
    """
    Executes the training loop for a hierarchical multi-class (HMC) local classifier model.
    This function performs the following steps:
    - Moves the model and loss criterions to the specified device.
    - Initializes early stopping parameters and tracking variables for each level of the hierarchy.
    - Sets up optimizers for each model level with individual learning rates and weight decays.
    - Iterates over the specified number of epochs, performing:
        - Training over batches: forward pass, loss computation for active levels, and gradient accumulation.
        - Backward pass and optimizer step for each level.
        - Logging of training losses.
        - Periodic evaluation on the validation set, including loss and precision reporting.
        - Early stopping if all levels have triggered it.
    Args:
        args: An object containing all necessary training parameters and objects, including:
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

    args.early_stopping_patience = 10
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    # args.level_active = [True] * args.hmc_dataset.max_depth
    args.level_active = [level in args.active_levels for level in range(args.max_depth)]
    logging.info("Active levels: %s", args.active_levels)
    logging.info("Level active: %s", args.level_active)

    args.best_val_loss = [float("inf")] * args.max_depth
    args.best_val_score = [0.0] * args.max_depth
    args.best_model = [None] * args.max_depth
    logging.info("Best val loss created %s", args.best_val_loss)

    # optimizers = [
    #     torch.optim.Adam(
    #         model.parameters(),
    #         lr=args.lr_values[int(idx)],
    #         weight_decay=args.weight_decay_values[int(idx)],
    #     )
    #     for idx, model in args.model.levels.items()
    # ]
    # args.optimizers = optimizers

    args.optimizers = torch.optim.Adam(
        args.model.parameters(),
        lr=args.lr_values[0],
        weight_decay=args.weight_decay_values[0],
    )

    for epoch in range(1, args.epochs + 1):
        args.model.train()
        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
        # args.active_levels = [i for i, active in enumerate(args.level_active) if active]
        # logging.info("Active levels: %s", args.active_levels)
        logging.info(
            "Level active: %s",
            [level for level, level_bool in enumerate(args.level_active) if level_bool],
        )

        for inputs, targets, _ in args.train_loader:

            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            # Zerar os gradientes antes de cada batch
            args.optimizers.zero_grad()
            # for optimizer in args.optimizers:
            #     optimizer.zero_grad()
            for index in args.active_levels:
                if args.level_active[index]:
                    output = outputs[index].double()
                    target = targets[index].double()

                    # MCLoss
                    if index == 0:
                        loss = args.criterions[index](output, target)
                    else:
                        R = args.hmc_dataset.all_matrix_r[index].to(args.device)
                        constr_output = get_constr_out(output, R)
                        train_output = target * output.double()
                        train_output = get_constr_out(train_output, R)
                        train_output = (
                            1 - target
                        ) * constr_output.double() + target * train_output
                        loss = args.criterions[index](train_output, target)
                    local_train_losses[index] += loss

        # Backward pass (cálculo dos gradientes)
        for i, total_loss in enumerate(local_train_losses):
            if i in args.active_levels and args.level_active[i]:
                total_loss.backward()
        args.optimizers.step()
        # for optimizer in args.optimizers:
        #    optimizer.step()

        local_train_losses = [
            loss / len(args.train_loader) for loss in local_train_losses
        ]
        non_zero_losses = [loss for loss in local_train_losses if loss > 0]
        global_train_loss = (
            sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0
        )

        logging.info("Epoch %d/%d", epoch, args.epochs)
        show_local_losses(local_train_losses, dataset="Train")
        show_global_loss(global_train_loss, dataset="Train")

        if epoch % args.epochs_to_evaluate == 0:
            local_val_losses, local_val_score = valid_step(args)
            show_local_losses(local_val_losses, dataset="Val")
            # show_local_score(local_val_score, dataset="Val")

            if not any(args.level_active):
                logging.info("All levels have triggered early stopping.")
                break

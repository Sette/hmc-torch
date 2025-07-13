import logging
import torch

from hmc.train.local_classifier.constrained.valid import valid_step
from hmc.train.utils import (
    show_global_loss,
    show_local_losses,
    show_local_score,
)

import networkx as nx
import numpy as np

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

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    R = np.zeros(args.hmc_dataset.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        args.hmc_dataset.A
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(args.hmc_dataset.A)):
        # here we need to use the function nx.descendants() \
        # because in the directed graph the edges have source \
        # from the descendant and point towards the ancestor
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(args.device)

    class_indices_per_level = {
        lvl: torch.tensor(
            [
                args.hmc_dataset.nodes_idx[n.replace("/", ".")]
                for n in args.hmc_dataset.levels[lvl]
            ],
            device=args.device,
        )
        for lvl in args.hmc_dataset.levels.keys()
    }

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
        logging.info("Active levels: %s", args.active_levels)

        for inputs, targets, _ in args.train_loader:

            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            # Zerar os gradientes antes de cada batch
            args.optimizers.zero_grad()
            for index in args.active_levels:
                if args.level_active[index]:
                    output = outputs[index].double()
                    target = targets[index].double()
                    child_indices = class_indices_per_level[
                        index
                    ]  # [n_classes_nivel_atual]
                    # MCLoss
                    if index == 0:
                        loss = args.criterions[index](output, target)
                    else:
                        # Índices globais dos pais para cada amostra
                        parent_target = targets[index - 1]
                        parent_indices = class_indices_per_level[index - 1]
                        parent_index_each_sample = parent_target.argmax(dim=1)
                        parent_global_idxs = parent_indices[parent_index_each_sample]
                        # Constrói a máscara usando R_global (shape: [1, n, n])
                        mask = torch.stack(
                            [
                                R[0, child_indices, parent_global_idxs[b]]
                                for b in range(output.shape[0])
                            ],
                            dim=0,
                        )  # [batch, n_classes_nivel_atual]
                        masked_output = output + (1 - mask) * (-1e9)
                        loss = args.criterions[index](
                            torch.sigmoid(masked_output), target
                        )
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
        # show_local_losses(local_train_losses, dataset="Train")
        # show_global_loss(global_train_loss, dataset="Train")

        if epoch % args.epochs_to_evaluate == 0:
            logging.info("Validating at epoch %d", epoch)
            valid_step(args)
            # show_local_losses(local_val_losses, dataset="Val")
            # show_local_score(local_val_score, dataset="Val")

            if not any(args.level_active):
                logging.info("All levels have triggered early stopping.")
                break

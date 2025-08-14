import logging

import torch

from hmc.train.local_classifier.core.valid import valid_step
from hmc.train.utils import (
    show_global_loss,
    show_local_losses,
)

from hmc.train.utils import (
    create_job_id_name,
)


def calculate_local_loss(outputs, targets, args, level):

    output = outputs[level]  # Preferencialmente float32
    target = targets[level]

    if args.model_regularization == "mask" and level != 0:
        child_indices = args.class_indices_per_level[level]  # [n_classes_nivel_atual]
        # MCLoss
        # Índices globais dos pais para cada amostra
        parent_target = targets[level - 1]
        parent_indices = args.class_indices_per_level[level - 1]
        parent_index_each_sample = parent_target.argmax(dim=1)
        parent_global_idxs = parent_indices[parent_index_each_sample]
        # Constrói a máscara usando R_global (shape: [1, n, n])
        mask = torch.stack(
            [
                args.R[0, child_indices, parent_global_idxs[b]]
                for b in range(output.shape[0])
            ],
            dim=0,
        )  # [batch, n_classes_nivel_atual]
        masked_output = output + (1 - mask) * (-1e9)
        loss = args.criterions[level](torch.sigmoid(masked_output), target)
    elif args.model_regularization == "soft" and level != 0:
        loss = args.criterions[level](output.double(), target)
        lambda_hier = 0.1
        global_dict = args.hmc_dataset.nodes_idx
        # classes_local_to_global: mapeia idx local para global correto
        local_dict = args.hmc_dataset.local_nodes_idx[level]
        classes_local_to_global = [
            int(global_dict[node.replace("/", ".")])
            for node, i in sorted(local_dict.items(), key=lambda x: x[1])
        ]
        # Constrói vetor de probabilidades globais "espalhado"
        probs_expanded = torch.zeros(
            (output.size(0), args.R.size(0)), device=output.device
        )
        probs_expanded[:, classes_local_to_global] = output
        # Penalidade global
        probs_i = probs_expanded.unsqueeze(2)  # (batch, n_total, 1)
        probs_j = probs_expanded.unsqueeze(1)  # (batch, 1, n_total)
        diff = probs_j - probs_i
        # Máscara hierárquica global (sem diagonal)
        R_mask = args.R - torch.eye(args.R.size(0), device=output.device)
        penalty = torch.clamp(diff * R_mask, min=0).sum() / (
            output.size(0) * R_mask.sum()
        )
        loss = loss + lambda_hier * penalty
    else:
        loss = args.criterions[level](output.double(), target)

    return loss


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
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    # args.level_active = [True] * args.hmc_dataset.max_depth
    args.level_active = [level in args.active_levels for level in range(args.max_depth)]
    logging.info("Active levels: %s", args.active_levels)
    logging.info("Level active: %s", args.level_active)

    args.best_val_loss = [float("inf")] * args.max_depth
    args.best_val_score = [0.0] * args.max_depth
    args.best_model = [None] * args.max_depth
    args.job_id = create_job_id_name(prefix="test")
    logging.info("Best val loss created %s", args.best_val_loss)

    args.optimizer = torch.optim.Adam(
        args.model.parameters(),
        lr=args.lr_values[0],
        weight_decay=args.weight_decay_values[0],
    )
    args.model.train()

    if args.model_regularization == "mask" or args.model_regularization == "soft":
        args.R = args.hmc_dataset.R.to(args.device)
        args.R = args.R.squeeze(0)
        print(args.R.shape)
        args.class_indices_per_level = {
            lvl: torch.tensor(
                [
                    args.hmc_dataset.nodes_idx[n.replace("/", ".")]
                    for n in args.hmc_dataset.levels[lvl]
                ],
                device=args.device,
            )
            for lvl in args.hmc_dataset.levels.keys()
        }

    n_warmup_epochs = 1  # defina quantas épocas quer pré-treinar o nível 0

    for epoch in range(1, args.epochs + 1):
        args.model.train()
        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
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
            args.optimizer.zero_grad()

            total_loss = 0.0

            # Se ainda estamos no warm-up, só treine o nível 0
            if epoch <= n_warmup_epochs:
                index = 0
                output = outputs[index].double()
                target = targets[index].double()
                loss = args.criterions[index](output, target)
                local_train_losses[index] += loss
            else:
                for level in args.active_levels:
                    if args.level_active[level]:
                        loss = calculate_local_loss(
                            outputs,
                            targets,
                            args,
                            level,
                        )
                        local_train_losses[
                            level
                        ] += loss.item()  # Acumula média por batch
                        total_loss += loss  # Soma da loss para backward

                # Após terminar loop dos níveis, execute backward
                total_loss.backward()
                args.optimizer.step()

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
            valid_step(args)
            # show_local_losses(local_val_losses, dataset="Val")
            # show_local_score(local_val_score, dataset="Val")

            if not any(args.level_active):
                logging.info("All levels have triggered early stopping.")
                break

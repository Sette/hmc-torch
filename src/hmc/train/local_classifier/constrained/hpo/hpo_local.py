import logging
import sys

import optuna
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from hmc.model.local_classifier.constrained.model import ConstrainedHMCLocalModel
from hmc.train.utils import (
    create_job_id_name,
    save_dict_to_json,
    show_global_loss,
    show_local_losses,
)
from hmc.model.local_classifier.constrained.model import get_constr_out

from hmc.utils.dir import create_dir


def optimize_hyperparameters_per_level(args):
    """
    Optimize hyperparameters for each active level of a hierarchical multi-class (HMC) local classifier using Optuna.
    This function performs hyperparameter optimization for each specified level in \
        a hierarchical classification model.
    For each level, it runs an Optuna study to find the best combination of hyperparameters (hidden dimension, \
        learning rate, dropout, number of layers, and weight decay) that minimizes the validation loss. \
    The best hyperparameters for each level are saved to a JSON file.
    Args:
        args: An object containing all necessary arguments and configurations, including:
            - levels_size (list): Number of classes per level.
            - input_dims (dict): Input dimensions per dataset.
            - data (str): Dataset identifier.
            - device (torch.device): Device to run the model on.
            - criterions (list): List of loss functions, one per level.
            - patience (int, optional): Number of epochs to wait for improvement before early stopping.
            - hmc_dataset: Dataset object with attribute max_depth.
            - epochs (int): Number of training epochs per trial.
            - epochs_to_evaluate (int): Frequency (in epochs) to evaluate on validation set.
            - train_loader (DataLoader): DataLoader for training data.
            - val_optimizer (callable): Function to evaluate validation loss and precision.
            - n_trials (int): Number of Optuna trials per level.
            - active_levels (list or None): List of levels to optimize. If None, all levels are optimized.
            - max_depth (int): Maximum depth (number of levels) in the hierarchy.
            - dataset_name (str): Name of the dataset.
    Returns:
        dict: A dictionary mapping each optimized level to its best hyperparameters, e.g.,
            {
                level_0: {
                    "hidden_dim": ...,
                    "lr": ...,
                    "dropout": ...,
                    "num_layers": ...,
                    "weight_decay": ...
                },
                ...
    Side Effects:
        - Saves the best hyperparameters per level to a JSON file in 'results/hpo/'.
        - Logs progress and results to the logging system and stdout.
    Raises:
        optuna.TrialPruned: If a trial is pruned by Optuna's early stopping mechanism.
    """

    def objective(trial, level):
        """
        Objective function for Optuna hyperparameter optimization of a hierarchical multi-class local classifier.
        This function defines the training and validation loop for a single Optuna trial, optimizing hyperparameters
        such as hidden dimension size, learning rate, dropout, number of layers, and weight decay for a specific level
        in a hierarchical classification model. It performs model training, validation, and early stopping based on
        validation loss, and reports results to Optuna for pruning and optimization.
        Args:
            trial (optuna.trial.Trial): The Optuna trial object used for suggesting hyperparameters \
                and reporting results.
            level (int): The hierarchical level for which the model is being optimized.
        Returns:
            float: The best validation loss achieved during training for the current trial.
        Raises:
            optuna.TrialPruned: If Optuna determines that the trial should be pruned early based on \
                intermediate results.
        """

        logging.info("Tentativa número: %d", trial.number)
        hidden_dim = trial.suggest_int("hidden_dim_level_%d" % level, 64, 512, log=True)
        lr_by_level = trial.suggest_float("lr_level_%d" % level, 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout_level_%d" % level, 0.3, 0.8, log=True)
        num_layers = trial.suggest_int("num_layers_level_%d" % level, 1, 3, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay_level_%d" % level, 1e-6, 1e-2, log=True
        )

        active_levels_train = [level]

        params = {
            "levels_size": args.levels_size[level],
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "active_levels": active_levels_train,
            "all_matrix_r": args.hmc_dataset.all_matrix_r,
        }

        args.model = ConstrainedHMCLocalModel(**params).to(args.device)

        optimizer = torch.optim.Adam(
            args.model.parameters(),
            lr=lr_by_level,
            weight_decay=weight_decay,
        )

        args.model = args.model.to(args.device)
        args.criterions = [criterion.to(args.device) for criterion in args.criterions]

        patience = args.patience if args.patience is not None else 3
        patience_counter = patience
        level_active = [False] * args.hmc_dataset.max_depth
        level_active[level] = True

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        # best_val_precision = 0.0

        logging.info("Levels to evaluate: %s", args.active_levels)
        logging.info("Best val loss created %f", best_val_loss)

        for epoch in range(1, args.epochs + 1):
            args.model.train()
            local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
            for inputs, targets, _ in args.train_loader:

                inputs, targets = inputs.to(args.device), [
                    target.to(args.device) for target in targets
                ]
                output = args.model(inputs.float())

                # Zerar os gradientes antes de cada batch
                optimizer.zero_grad()
                target = targets[level].float()

                loss = args.criterions[level](output[str(level)], target)
                local_train_losses[level] += loss

            # Backward pass (cálculo dos gradientes)
            for total_loss in local_train_losses:
                if total_loss > 0:
                    total_loss.backward()
            optimizer.step()

            local_train_losses = [
                loss / len(args.train_loader) for loss in local_train_losses
            ]
            non_zero_losses = [loss for loss in local_train_losses if loss > 0]
            global_train_loss = (
                sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0
            )

            logging.info("Trial %d - Epoch %d/%d", trial.number, epoch, args.epochs)
            show_local_losses(local_train_losses, dataset=f"Train-{trial.number}")
            show_global_loss(global_train_loss, dataset=f"Train-{trial.number}")

            if epoch % args.epochs_to_evaluate == 0:
                local_val_loss, local_val_f1 = val_optimizer(args)
                if (
                    round(local_val_f1, 4) > best_val_f1
                    and local_val_loss < best_val_loss
                ):
                    best_val_f1 = round(local_val_f1, 4)
                    best_val_loss = local_val_loss
                    patience_counter = patience
                else:
                    if (
                        round(local_val_f1, 4) < best_val_f1
                        or local_val_loss > best_val_loss
                    ):
                        patience_counter -= 1

                if patience_counter == 0:
                    logging.info(
                        "Early stopping triggered for trial %d at epoch %d.",
                        trial.number,
                        epoch,
                    )
                    break

                # Reporta o valor de validação para Optuna
                trial.report(local_val_f1, step=epoch)

                logging.info("Local loss %d: %f", trial.number, local_val_loss)
                logging.info("Local F1 %d: %f", trial.number, local_val_f1)

                # Early stopping (pruning)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return best_val_f1

    best_params_per_level = {}

    create_dir("results/hpo")
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if args.active_levels is None:
        args.active_levels = [i for i in range(args.max_depth)]
        logging.info("Active levels: %s", args.active_levels)
    else:
        args.active_levels = [int(x) for x in args.active_levels]
        logging.info("Active levels: %s", args.active_levels)

    for level in args.active_levels:
        args.level = level
        logging.info("\n🔍 Optimizing hyperparameters for level %d...\n", level)

        study = optuna.create_study()

        study.optimize(
            lambda trial: objective(trial, args.level),
            n_trials=args.n_trials,
        )

        level_parameters = {
            "hidden_dim": study.best_params[f"hidden_dim_level_{level}"],
            "lr": study.best_params[f"lr_level_{level}"],
            "dropout": study.best_params[f"dropout_level_{level}"],
            "num_layers": study.best_params[f"num_layers_level_{level}"],
            "weight_decay": study.best_params[f"weight_decay_level_{level}"],
        }

        best_params_per_level[level] = level_parameters

        logging.info(
            "✅ Best hyperparameters for level %s: %s", level, study.best_params
        )

    job_id = create_job_id_name(prefix="hpo")

    save_dict_to_json(
        best_params_per_level,
        f"results/hpo/best_params_{args.dataset_name}-{job_id}.json",
    )

    return best_params_per_level


def val_optimizer(args):
    """
    Evaluates the model on the validation set and computes the average loss and average precision score.
    Args:
        args: An object containing the following attributes:
            - model: The PyTorch model to evaluate.
            - val_loader: DataLoader for the validation dataset.
            - device: The device (CPU or CUDA) to use for computation.
            - criterions: A list or dict of loss functions for each level.
            - level: The current level to evaluate.
    Returns:
        tuple:
            - local_val_loss (float): The average validation loss over all batches.
            - local_val_precision (float): The average precision score (micro-averaged) over the validation set.
    """

    args.model.eval()
    local_val_loss = 0.0
    output_val = 0.0
    y_val = 0.0
    local_val_precision = 0.0
    local_val_f1 = 0.0
    theshold = 0.3

    with torch.no_grad():
        for level, (inputs, targets, _) in enumerate(args.val_loader):
            if level == args.level:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to(args.device), [
                        target.to(args.device) for target in targets
                    ]
                outputs = args.model(inputs.float())
                output = outputs[str(args.level)]
                target = targets[args.level].float()

                if args.constrained and level != 0:
                    constr_output = get_constr_out(
                        output, args.hmc_dataset.all_matrix_r[level].to(args.device)
                    )
                    train_output = target * output.double()
                    train_output = get_constr_out(
                        train_output,
                        args.hmc_dataset.all_matrix_r[level].to(args.device),
                    )
                    train_output = (
                        1 - target
                    ) * constr_output.double() + target * train_output
                else:
                    train_output = output

                local_val_loss += args.criterions[args.level](output, target)

                if level == 0:
                    output_val = output.to("cpu")
                    y_val = target.to("cpu")
                else:
                    output_val = torch.cat((output_val, output.to("cpu")), dim=0)
                    y_val = torch.cat((y_val, target.to("cpu")), dim=0)

    local_val_precision = average_precision_score(y_val, output_val, average="micro")
    score = precision_recall_fscore_support(
        y_val.numpy(),
        output_val.data > theshold,
        average="micro",
    )
    local_val_f1 = score[2]
    logging.info(
        "Validation Loss: %.4f, Validation Precision: %.4f, Validation F1: %.4f",
        local_val_loss,
        local_val_precision,
        local_val_f1,
    )

    local_val_loss = local_val_loss / len(args.val_loader)
    # logging.info(f"Levels to evaluate: {args.active_levels}")

    return local_val_loss, local_val_f1

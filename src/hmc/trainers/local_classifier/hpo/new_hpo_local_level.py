import logging
import sys

import optuna
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from hmc.models.local_classifier.baseline import HMCLocalModel
from hmc.models.local_classifier.constraint import HMCLocalModelConstraint
from hmc.utils.dataset.labels import (
    show_local_losses,
)
from hmc.utils.path.dir import create_dir
from hmc.utils.path.output import save_dict_to_json
from hmc.utils.train.early_stopping import (
    check_early_stopping_normalized,
)
from hmc.utils.train.job import create_job_id_name

from hmc.utils.train.losses import calculate_local_loss


def optimize_hyperparameters(args):
    """
    Optimize hyperparameters for each active level of a hierarchical \
        multi-class (HMC) local classifier using Optuna.
    This function performs hyperparameter optimization for each specified level in \
        a hierarchical classification model.
    For each level, it runs an Optuna study to find the best \
        combination of hyperparameters (hidden dimension, \
        learning rate, dropout, number of layers, and weight decay) \
            that minimizes the validation loss. \
    The best hyperparameters for each level are saved to a JSON file.
    Args:
        args: An object containing all necessary arguments \
            and configurations, including:
            - levels_size (list): Number of classes per level.
            - input_dims (dict): Input dimensions per dataset.
            - data (str): Dataset identifier.
            - device (torch.device): Device to run the model on.
            - criterions (list): List of loss functions, one per level.
            - patience (int, optional): Number of epochs to wait for \
                improvement before early stopping.
            - hmc_dataset: Dataset object with attribute max_depth.
            - epochs (int): Number of training epochs per trial.
            - epochs_to_evaluate (int): Frequency (in epochs) to evaluate\
                on validation set.
            - train_loader (DataLoader): DataLoader for training data.
            - val_optimizer (callable): Function to evaluate validation \
                loss and precision.
            - n_trials (int): Number of Optuna trials per level.
            - active_levels (list or None): List of levels to optimize.\
                If None, all levels are optimized.
            - max_depth (int): Maximum depth (number of levels) in the hierarchy.
            - dataset_name (str): Name of the dataset.
    Returns:
        dict: A dictionary mapping each optimized level to its best\
            hyperparameters, e.g.,
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

    def objective(trial):
        """
        Objective function for Optuna hyperparameter optimization of a hierarchical\
            multi-class local classifier.
        This function defines the training and validation loop for a \
            single Optuna trial, optimizing hyperparameters
        such as hidden dimension size, learning rate, dropout, number of layers, \
            and weight decay for a specific level
        in a hierarchical classification model. It performs model training, validation,\
            and early stopping based on
        validation loss, and reports results to Optuna for pruning and optimization.
        Args:
            trial (optuna.trial.Trial): The Optuna trial object used for suggesting \
                hyperparameters and reporting results.
            level (int): The hierarchical level for which the model is\
                being optimized.
        Returns:
            float: The best validation loss achieved during training for the\
                current trial.
        Raises:
            optuna.TrialPruned: If Optuna determines that the trial should be \
                pruned early based on intermediate results.
        """

        logging.info("Trial number: %d", trial.number)
        dropouts = {}
        hidden_dims_all = {}
        num_layers_values = {}
        weight_decays = {}
        lrs = {}
        for level in args.active_levels:
            dropout = trial.suggest_float(
                f"dropout_level_{level}",
                0.3,
                0.8,
                log=True,
            )
            weight_decay = trial.suggest_float(
                f"weight_decay_level_{level}",
                1e-6,
                1e-2,
                log=True,
            )
            lr = trial.suggest_float(
                f"lr_level_{level}",
                1e-6,
                1e-2,
                log=True,
            )
            num_layers = trial.suggest_int(
                f"num_layers_level_{level}",
                2,
                5,
                log=True,
            )

            dropouts[level] = dropout
            weight_decays[level] = weight_decay
            lrs[level] = lr
            num_layers_values[level] = num_layers

            for i in range(num_layers):
                if i == 0:
                    dim = trial.suggest_int(
                        f"hidden_dim_level_{level}_layer_{i}",
                        args.input_size,
                        args.input_size * 4,
                        log=True,
                    )
                else:
                    dim = trial.suggest_int(
                        f"hidden_dim_level_{level}_layer_{i}",
                        args.levels_size[level],
                        args.levels_size[level] * 4,
                        log=True,
                    )

                hidden_dims_all[level].append(dim)

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_size,
            "hidden_dims": hidden_dims_all,
            "num_layers": num_layers_values,
            "dropouts": dropouts,
            "active_levels": [level],
            "results_path": args.results_path,
            "residual": args.model_regularization == "residual",
        }

        args.model = HMCLocalModel(**params).to(args.device)

    args.job_id = create_job_id_name(prefix="hpo")

    args.results_path = (
        f"{args.output_path}/hpo/{args.method}/{args.dataset_name}/{args.job_id}"
    )

    args.best_params_per_level = {}

    args.input_size = args.input_dims[args.data]

    create_dir(args.results_path)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        objective,
        n_trials=args.n_trials,
    )

    logging.info("Best hyperparameters: %s", study.best_params)


def val_optimizer(args):
    """
    Evaluates the model on the validation set and computes the average \
        loss and average precision score.
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
            - local_val_precision (float): The average precision score \
                (micro-averaged) over the validation set.
    """

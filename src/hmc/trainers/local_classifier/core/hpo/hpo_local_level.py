import logging
import sys
import optuna
import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from hmc.models.local_classifier.baseline.model import HMCLocalModel
from hmc.models.local_classifier.constrained.model import ConstrainedHMCLocalModel
from hmc.utils.job import create_job_id_name


from hmc.utils.labels import (
    show_local_losses,
)

from hmc.utils.early_stopping import (
    check_early_stopping_normalized,
)


from hmc.utils.output import save_dict_to_json

from hmc.utils.dir import create_dir

import numpy as np
import random



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

    def objective(trial, level):
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

        logging.info("Tentativa número: %d", trial.number)
        dropout = trial.suggest_float(f"dropout_level_{level}", 0.3, 0.8, log=True)
        weight_decay = trial.suggest_float(f"weight_decay_level_{level}", 1e-6, 1e-2, log=True)
        lr = trial.suggest_float(f"lr_level_{level}", 1e-6, 1e-2, log=True)
        num_layers = trial.suggest_int(f"num_layers_level_{level}", 1, 5, log=True)
        dropouts = {level: dropout}
        hidden_dims = []
        
        for i in range(num_layers):
            input_size = args.input_size
            if args.model_regularization == "resitual" and level > 0:
                input_size += args.levels_size[level - 1]
            if i == 0:
                dim = trial.suggest_int(
                    f"hidden_dim_level_{level}_layer_{i}",
                    input_size,
                    input_size*3,
                    log=True)
            else:
                dim = trial.suggest_int(
                    f"hidden_dim_level_{level}_layer_{i}",
                    args.levels_size[level],
                    args.levels_size[level]*3,
                    log=True)
            hidden_dims.append(dim)
            
        hidden_dims_all = {level: hidden_dims}
        
        hpo_levels = [i for i in range(level+1)]
        
        print(f"Active levels: {hpo_levels}")
        
        for i in hpo_levels:
            if i != level:
                num_layers_local = args.best_params_per_level[i]["num_layers"]
                hidden_dims_all[i] = [
                    args.best_params_per_level[i]["hidden_dims"][layer]
                    for layer in range(num_layers_local)
                ]
                dropouts[i] = args.best_params_per_level[i]["dropout"]
        
        args.level_active = [
            i == level for i in range(args.max_depth)
        ]

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_size,
            "hidden_dims": hidden_dims_all,
            "num_layers": num_layers,
            "dropouts": dropouts,
            "active_levels": hpo_levels,
            "results_path": args.results_path,
            "resitual": args.model_regularization == "resitual",
        }

        if args.method == "local_constrained":
            args.model = ConstrainedHMCLocalModel(**params).to(args.device)
        else:
            args.model = HMCLocalModel(**params).to(args.device)

        optimizer = torch.optim.Adam(
            args.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        args.optimizer = optimizer

        args.model = args.model.to(args.device)
        for criterion in args.criterions:
            criterion.to(args.device)

        args.best_val_loss = [float("inf")] * args.max_depth
        args.best_val_score = [0.0] * args.max_depth
        args.best_model = [None] * args.max_depth

        args.early_stopping_patience = args.patience
        args.early_stopping_patience_score = args.patience_score
        args.patience_counters = [0] * args.hmc_dataset.max_depth
        args.patience_counters_score = [0] * args.hmc_dataset.max_depth

        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]

        for epoch in range(1, args.epochs + 1):
            args.model.train()
            for inputs, targets, _ in args.train_loader:
                inputs = inputs.to(args.device)
                outputs = args.model(inputs.float())

                # Zerar os gradientes antes de cada batch
                args.optimizer.zero_grad()

                total_loss = 0.0

                output = outputs[level].to(args.device)
                target = targets[level].to(args.device)

                loss = args.criterions[level](output.double(), target)
                local_train_losses[level] += loss.item()  # Acumula média por batch
                total_loss += loss  # Soma da loss para backward

                # Após terminar loop dos níveis, execute backward
                total_loss.backward()
                args.optimizer.step()

            local_train_losses[level] = local_train_losses[level] / len(
                args.train_loader
            )
            logging.info("Trial %d - Epoch %d/%d", trial.number, epoch, args.epochs)
            show_local_losses(local_train_losses, dataset=f"Train n {trial.number}")

            if epoch % args.epochs_to_evaluate == 0:
                args.epoch = epoch
                val_loss, val_score = val_optimizer(args, level)

                if not any(args.level_active):
                    logging.info("All levels have triggered early stopping.")
                    break

                # Reporta o valor de validação para Optuna
                trial.report(val_score, step=epoch)

                logging.info(
                    "Trial %d Local validation loss: %f AVG Score: %f",
                    trial.number,
                    val_loss,
                    val_score,
                )

                logging.info(
                    "Local best validation loss: %f AVG precision: %f",
                    args.best_val_loss[level],
                    args.best_val_score[level],
                )

                # Early stopping (pruning)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return val_score

    args.job_id = create_job_id_name(prefix="hpo")

    args.results_path = (
        f"{args.output_path}/hpo/{args.method}/{args.dataset_name}/{args.job_id}"
    )
    
    args.best_params_per_level = {}

    args.input_size = args.input_dims[args.data]

    create_dir(args.results_path)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    

    for level in args.active_levels:
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, level),
            n_trials=args.n_trials,
        )

        logging.info("Best hyperparameters for level %d: %s", level, study.best_params)

        num_layers = study.best_params[f"num_layers_level_{level}"]

        hidden_dims = [
            study.best_params[f"hidden_dim_level_{level}_layer_{i}"]
            for i in range(num_layers)
        ]

        level_parameters = {
            "hidden_dims": hidden_dims,
            "dropout": study.best_params[f"dropout_level_{level}"],
            "num_layers": num_layers,
            "weight_decay": study.best_params[f"weight_decay_level_{level}"],
            "lr": study.best_params[f"lr_level_{level}"],
        }

        args.best_params_per_level[level] = level_parameters

        save_dict_to_json(
            level_parameters,
            f"{args.results_path}/best_params_{args.dataset_name}-{level}.json",
        )

        logging.info(
            "✅ Best hyperparameters for level %s: %s", level, study.best_params
        )

    save_dict_to_json(
        args.best_params_per_level,
        f"{args.results_path}/best_params_{args.dataset_name}.json",
    )

    return args.best_params_per_level


def val_optimizer(args, level):
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

    args.model.eval()

    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    args.local_val_scores = {
        level: None for _, level in enumerate(args.active_levels)
    }

    args.local_val_losses = [0.0] * args.max_depth

    # Get local scores
    threshold = 0.2

    with torch.no_grad():
        total_loss = 0.0  # <-- inicializa aqui, fora do loop
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            if torch.cuda.is_available():
                inputs = inputs.to(args.device)
            outputs = args.model(inputs.float())

            if not args.level_active[level]:
                continue  # Pula se não estiver ativo
            output = outputs[level].to(args.device)
            target = targets[level].to(args.device)
            loss = args.criterions[level](output.double(), target)
            args.local_val_losses[level] += loss.item()
            total_loss += loss

            binary_outputs = (output > threshold).float()

            if i == 0:  
                local_outputs[level] = binary_outputs
                local_inputs[level] = target
            else:
                local_outputs[level] = torch.cat(
                    (local_outputs[level], binary_outputs), dim=0
                )
                local_inputs[level] = torch.cat((local_inputs[level], target), dim=0)

    if args.level_active[level]:
        y_pred = local_outputs[level].to("cpu").int().numpy()
        y_true = local_inputs[level].to("cpu").int().numpy()

        score = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        avg_score = average_precision_score(
            y_true,
            y_pred,
            average="micro",
        )

        logging.info(
            "Level %d: precision=%.4f, recall=%.4f, f1-score=%.4f avg score=%.4f",
            level,
            score[0],
            score[1],
            score[2],
            avg_score,
        )

        args.local_val_scores[level] = avg_score

    args.local_val_losses = [
        loss / len(args.val_loader) for loss in args.local_val_losses
    ]
    logging.info("Levels to evaluate: %s", args.active_levels)

    check_early_stopping_normalized(args, active_levels=[level], save_model=True)

    return args.local_val_losses[level], args.local_val_scores[level]

import logging
import sys

import optuna
import torch
from sklearn.metrics import precision_recall_fscore_support

from hmc.model.local_classifier.baseline.model import HMCLocalModel
from hmc.train.utils import (
    create_job_id_name,
    save_dict_to_json,
    show_global_loss,
    show_local_losses,
)
from hmc.utils.dir import create_dir


def check_metrics(metric, best_metric, metric_type="loss"):
    if metric_type == "loss":
        if metric < best_metric:
            return True
        else:
            return False
    elif metric_type == "f1":
        if metric > best_metric:
            return True
        else:
            return False
    else:
        return False


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
        hidden_dim = trial.suggest_int("hidden_dim_level_%s" % level, 64, 512, log=True)
        dropout = trial.suggest_float("dropout_level_%s" % level, 0.3, 0.8, log=True)
        num_layers = trial.suggest_int("num_layers_level_%s" % level, 1, 3, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay_level_%s" % level, 1e-6, 1e-2, log=True
        )
        lr = trial.suggest_float("lr_level_%s" % level, 1e-6, 1e-3, log=True)
        args.active_levels = [level]

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "active_levels": args.active_levels,
        }

        args.model = HMCLocalModel(**params).to(args.device)

        optimizer = torch.optim.Adam(
            args.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        args.optimizer = optimizer

        args.model = args.model.to(args.device)
        args.criterions = [criterion.to(args.device) for criterion in args.criterions]
        args.level_active = [
            level in args.active_levels for level in range(args.max_depth)
        ]
        patience = args.patience if args.patience is not None else 3
        patience_counter = 0
        args.early_stopping_patience = patience
        args.patience_counters = [0] * args.hmc_dataset.max_depth

        args.best_total_val_loss = float("inf")
        args.best_val_loss = [float("inf") for _ in range(args.max_depth)]
        args.best_val_score = [0.0] * args.max_depth

        logging.info("Levels to evaluate: %s", args.active_levels)

        for epoch in range(1, args.epochs + 1):
            args.model.train()
            local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
            for inputs, targets, _ in args.train_loader:

                inputs, targets = inputs.to(args.device), [
                    target.to(args.device) for target in targets
                ]
                outputs = args.model(inputs.float())

                # Zerar os gradientes antes de cada batch
                args.optimizer.zero_grad()

                total_loss = 0.0

                for index in args.active_levels:
                    if args.level_active[index]:
                        output = outputs[index]  # Preferencialmente float32
                        target = targets[index]
                        loss = args.criterions[index](output.double(), target)
                        local_train_losses[
                            index
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

            logging.info("Trial %d - Epoch %d/%d", trial.number, epoch, args.epochs)
            show_local_losses(local_train_losses, dataset=f"Train-{trial.number}")
            show_global_loss(global_train_loss, dataset=f"Train-{trial.number}")

            if epoch % args.epochs_to_evaluate == 0:
                total_loss = val_optimizer(args)
                if round(total_loss.item(), 4) < args.best_total_val_loss:
                    args.best_total_val_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info(
                        "Early stopping triggered for trial %d at epoch %d.",
                        trial.number,
                        epoch,
                    )
                    break

                if not any(args.level_active):
                    logging.info("All levels have triggered early stopping.")
                    # Reporta o valor de validação para Optuna
                    break

                # Reporta o valor de validação para Optuna
                trial.report(total_loss.item(), step=epoch)

                logging.info("Local loss %d: %f", trial.number, total_loss.item())

                # Early stopping (pruning)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return total_loss.item()

    best_params_per_level = {}

    create_dir("results/hpo")
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if args.active_levels is None:
        args.active_levels = list(args.max_depth)
        logging.info("Active levels: %s", args.active_levels)
    else:
        args.active_levels = [int(x) for x in args.active_levels]
        logging.info("Active levels: %s", args.active_levels)

    for level in args.active_levels:
        study = optuna.create_study()
        study.optimize(
            lambda trial: objective(trial, level),
            n_trials=args.n_trials,
        )

        logging.info("Best hyperparameters for level %d: %s", level, study.best_params)
        level_parameters = {
            "hidden_dim": study.best_params[f"hidden_dim_level_{level}"],
            "dropout": study.best_params[f"dropout_level_{level}"],
            "num_layers": study.best_params[f"num_layers_level_{level}"],
            "weight_decay": study.best_params[f"weight_decay_level_{level}"],
            "lr": study.best_params[f"lr_level_{level}"],
        }

        best_params_per_level[level] = level_parameters

        logging.info(
            "✅ Best hyperparameters for level %s: %s", level, study.best_params
        )

    best_params_per_level["global"] = {}

    job_id = create_job_id_name(prefix="hpo")

    save_dict_to_json(
        best_params_per_level,
        f"results/hpo/best_params_{args.dataset_name}-{job_id}.json",
    )

    return best_params_per_level


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

    args.model.eval()
    local_val_losses = [0.0] * args.max_depth

    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    # Get local scores
    local_val_score = {level: None for _, level in enumerate(args.active_levels)}
    threshold = 0.2

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.to(args.device), [
                    target.to(args.device) for target in targets
                ]
            outputs = args.model(inputs.float())

            total_loss = 0.0

            for index in args.active_levels:
                if not args.level_active[index]:
                    continue  # Pula se não estiver ativo
                output = outputs[index]
                target = targets[index]
                loss = args.criterions[index](output.double(), target)
                local_val_losses[index] += loss.item()
                total_loss += loss

                # *** Para métricas e concatenação ***
                # Se quiser outputs binários para avaliação:
                binary_outputs = (output > threshold).float()

                # Acumulação de outputs e targets para métricas
                if i == 0:  # Primeira iteração: inicia tensor
                    local_outputs[index] = binary_outputs
                    local_inputs[index] = target
                else:  # Nas seguintes, empilha ao longo do batch
                    local_outputs[index] = torch.cat(
                        (local_outputs[index], binary_outputs), dim=0
                    )
                    local_inputs[index] = torch.cat(
                        (local_inputs[index], target), dim=0
                    )

    for idx in args.active_levels:
        if args.level_active[idx]:
            y_pred = local_outputs[idx].to("cpu").int().numpy()
            y_true = local_inputs[idx].to("cpu").int().numpy()

            score = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )
            # local_val_score[idx] = score
            logging.info(
                "Level %d: precision=%.4f, recall=%.4f, f1-score=%.4f",
                idx,
                score[0],
                score[1],
                score[2],
            )

            local_val_score[idx] = score[2]

    results_path = f"results/train/{args.method}-{args.dataset_name}"
    create_dir(results_path)

    local_val_losses = [loss / len(args.val_loader) for loss in local_val_losses]
    logging.info("Levels to evaluate: %s", args.active_levels)
    for i in args.active_levels:
        metric, best_metric = 0, 0
        if args.level_active[i]:
            if args.early_metric == "loss":
                metric = round(local_val_losses[i], 4)
                best_metric = args.best_val_loss[i]
            elif args.early_metric == "f1":
                metric = round(local_val_score[i], 4)
                best_metric = args.best_val_score[i]
            if check_metrics(metric, best_metric, metric_type=args.early_metric):
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[i] = round(local_val_losses[i], 4)
                args.best_val_score[i] = round(local_val_score[i], 4)
                args.patience_counters[i] = 0
                logging.info(
                    "Level %d: improved (F1 score=%.4f)", i, local_val_score[i]
                )

            else:
                # Incrementar o contador de paciência
                args.patience_counters[i] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    i,
                    args.patience_counters[i],
                    args.early_stopping_patience,
                )
                if args.patience_counters[i] >= args.early_stopping_patience:
                    args.level_active[i] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d —\
                            freezing its parameters",
                        i,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(i)].parameters():
                        param.requires_grad = False

    total_loss = total_loss / len(args.val_loader)
    # logging.info(f"Levels to evaluate: {args.active_levels}")

    return total_loss

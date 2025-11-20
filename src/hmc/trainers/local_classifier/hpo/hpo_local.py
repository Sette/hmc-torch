import logging
import sys
import optuna
import torch

from hmc.models.local_classifier.baseline.model import HMCLocalModel
from hmc.models.local_classifier.constrained.model import ConstrainedHMCLocalModel
from hmc.trainers.local_classifier.core.valid_local import valid_step
from hmc.utils.dir import create_dir
from hmc.utils.job import create_job_id_name
from hmc.utils.labels import (
    get_probs_ancestral_descendent,
    show_local_losses,
)
from hmc.utils.path.output import save_dict_to_json
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

    def objective(trial, args):
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

        weight_decays = {level: 0 for level in range(args.hmc_dataset.max_depth)}
        dropouts = {level: 0 for level in range(args.hmc_dataset.max_depth)}
        lrs = {level: 0 for level in range(args.hmc_dataset.max_depth)}
        num_layers = {level: 0 for level in range(args.hmc_dataset.max_depth)}
        hidden_dims = {level: [] for level in range(args.hmc_dataset.max_depth)}

        logging.info("Tentativa número: %d", trial.number)
        for level in range(args.hmc_dataset.max_depth):
            logging.info("Otimizando nível %d", level)
            dropouts[level] = trial.suggest_float(
                "dropout_level_%s" % level, 0.3, 0.8, log=True
            )
            weight_decays[level] = trial.suggest_float(
                "weight_decay_level_%s" % level, 1e-6, 1e-2, log=True
            )
            lrs[level] = trial.suggest_float(
                "lr_level_%s" % level, 1e-6, 1e-2, log=True
            )

            num_layers[level] = trial.suggest_int(
                "num_layers_level_%s" % level, 1, 5, log=True
            )

            local_hidden_dims = []

            # 3. Use um laço para sugerir a dimensão de CADA camada
            for i in range(num_layers[level]):
                # O nome do parâmetro agora inclui o índice da camada (ex: 'hidden_dim_level_0_layer_0')
                if i == 0:
                    dim = trial.suggest_int(
                        "hidden_dim_level_%s_layer_%s" % (level, i),
                        args.input_size,
                        args.input_size * 3,
                        log=True,
                    )
                else:
                    dim = trial.suggest_int(
                        "hidden_dim_level_%s_layer_%s" % (level, i),
                        args.levels_size[level],
                        args.levels_size[level] * 3,
                        log=True,
                    )
                local_hidden_dims.append(dim)

            hidden_dims[level] = local_hidden_dims

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_size,
            "hidden_dims": hidden_dims,
            "num_layers": num_layers,
            "dropouts": dropouts,
            "results_path": args.results_path,
        }

        if args.method == "local_constraint":
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
            params["device"] = args.device
            params["nodes_idx"] = args.hmc_dataset.nodes_idx
            params["local_nodes_reverse_idx"] = args.hmc_dataset.local_nodes_reverse_idx
            params["r"] = args.hmc_dataset.r.to(args.device)
            params["edges_matrix_dict"] = (
                args.hmc_dataset.edges_matrix_dict
            )  # Precomputed mapping matrices
            args.model = ConstrainedHMCLocalModel(**params).to(args.device)
        else:
            args.model = HMCLocalModel(**params).to(args.device)

        args.criterions = [criterion.to(args.device) for criterion in args.criterions]

        args.early_stopping_patience = args.patience
        args.early_stopping_patience_score = args.patience_score
        # if args.early_metric == "f1-score":
        #     args.early_stopping_patience = 20
        args.patience_counters = [0] * args.hmc_dataset.max_depth
        args.patience_counters_score = [0] * args.hmc_dataset.max_depth
        args.level_active = [
            level in args.active_levels for level in range(args.max_depth)
        ]
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
                lr=lrs[level],
                weight_decay=weight_decays[level],
            )
            for level in range(args.hmc_dataset.max_depth)
        ]

        args.model.train()
        next_level = 1
        if args.model_regularization == "soft":
            args.r = args.hmc_dataset.r.to(args.device)
            args.level_active = [False] * len(args.level_active)
            args.level_active[0] = True

            logging.info(
                "Using soft regularization with %d warm-up epochs", args.n_warmup_epochs
            )
            #

            print(args.r.shape)
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

        else:
            args.level_active = [True] * len(args.max_depth)

            print("not using soft regularization")

        for epoch in range(1, args.epochs + 1):
            args.epoch = epoch
            args.model.train()
            local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
            logging.info(
                "Level active: %s",
                [
                    level
                    for level, level_bool in enumerate(args.level_active)
                    if level_bool
                ],
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
                    local_train_losses[level] = local_train_loss / len(
                        args.train_loader
                    )

            logging.info("Epoch %d/%d", epoch, args.epochs)
            show_local_losses(local_train_losses, dataset="Train")

            if epoch % args.epochs_to_evaluate == 0:
                valid_step(args)
                if not any(args.level_active):
                    logging.info("All levels have triggered early stopping.")
                    break

                val_loss = 0.0
                val_score = 0.0
                best_val_loss = 0.0
                best_val_score = 0.0
                for level in args.active_levels:
                    if args.level_active[level]:
                        logging.info(
                            "Level %d - Best val loss: %f - Best val precision: %f",
                            level,
                            args.best_val_loss[level],
                            args.best_val_score[level],
                        )

                        val_loss += args.local_val_losses[level]
                        val_score += args.local_val_scores[level]
                        best_val_loss += args.best_val_loss[level]
                        best_val_score += args.best_val_score[level]
                    else:
                        val_loss += args.best_val_loss[level]
                        val_score += args.best_val_score[level]
                        best_val_loss += args.best_val_loss[level]
                        best_val_score += args.best_val_score[level]
                if next_level > args.max_depth:
                    val_score = val_score / args.max_depth
                else:
                    val_score = val_score / (next_level)
                val_loss = val_loss / args.max_depth
                best_val_loss = best_val_loss / args.max_depth
                best_val_score = best_val_score / args.max_depth

                logging.info(
                    "Trial %d Local validation loss: %f AVG Score: %f",
                    trial.number,
                    val_loss,
                    val_score,
                )

                logging.info(
                    "Local best validation loss: %f AVG precision: %f",
                    best_val_loss,
                    best_val_score,
                )

                if next_level > args.max_depth:
                    # Reporta o valor de validação para Optuna
                    trial.report(val_score, step=epoch)

                    # Early stopping (pruning)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if epoch % args.n_warmup_epochs == 0:
                if next_level < args.max_depth:
                    args.level_active[next_level] = True
                    logging.info("Activating level %d", next_level)
                    next_level += 1
                    args.n_warmup_epochs += args.n_warmup_epochs_increment

        return val_score

    best_params_per_level = {}

    args.job_id = create_job_id_name(prefix="hpo")

    args.results_path = (
        f"{args.output_path}/hpo/{args.method}/{args.dataset_name}/{args.job_id}"
    )

    args.input_size = args.input_dims[args.data]

    create_dir(args.results_path)
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # Cria um sampler com seed fixa
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
    )

    for level in args.hmc_dataset.levels.keys():
        logging.info("Best hyperparameters for level %d: %s", level, study.best_params)
        num_layers = study.best_params[f"num_layers_level_{level}"]

        # Reconstrói a lista de dimensões
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

        best_params_per_level[level] = level_parameters

        save_dict_to_json(
            level_parameters,
            f"{args.results_path}/best_params_{args.dataset_name}-{level}.json",
        )

        logging.info(
            "✅ Best hyperparameters for level %s: %s", level, study.best_params
        )

    save_dict_to_json(
        best_params_per_level,
        f"{args.results_path}/best_params_{args.dataset_name}.json",
    )

    return best_params_per_level

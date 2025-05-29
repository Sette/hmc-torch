import json
import logging

import optuna
import torch
from sklearn.metrics import average_precision_score
from hmc.model.local_classifier.baseline.model import HMCLocalModel
from hmc.train.utils import (
    local_to_global_predictions,
    show_global_loss,
    show_local_losses,
    create_job_id_name)
from hmc.utils.dir import create_dir


def save_dict_to_json(dictionary, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)


def optimize_hyperparameters_per_level(args):
    def objective(trial, level):
        # hidden_dim = {
        #     i: trial.suggest_int(f"hidden_dim_level_{i}", 64, 512, log=True)
        #     for i in range(args.max_depth)
        # }

        # # lr = [trial.suggest_float("lr", 1e-4, 1e-2, log=True) for _ in range(args.max_depth)]
        # lr_by_level = {
        #     i: trial.suggest_float(f"lr_level_{i}", 1e-6, 1e-3, log=True)
        #     for i in range(args.max_depth)
        # }
        # dropout = {
        #     i: trial.suggest_float(f"dropout_level_{i}", 0.3, 0.8, log=True)
        #     for i in range(args.max_depth)
        # }
        # num_layers = {
        #     i: trial.suggest_int(f"num_layers_level_{i}", 1, 3, log=True)
        #     for i in range(args.max_depth)
        # }
        # weight_decay = {
        #     i: trial.suggest_float(f"weight_decay_level_{i}", 1e-6, 1e-2, log=True)
        #     for i in range(args.max_depth)
        # }

        hidden_dim = trial.suggest_int(f"hidden_dim_level_{level}", 64, 512, log=True)
        lr_by_level = trial.suggest_float(f"lr_level_{level}", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float(f"dropout_level_{level}", 0.3, 0.8, log=True)
        num_layers = trial.suggest_int(f"num_layers_level_{level}", 1, 3, log=True)
        weight_decay = trial.suggest_float(f"weight_decay_level_{level}", 1e-6, 1e-2, log=True)

        active_levels_train = [level]

        params = {
            "levels_size": args.levels_size[level],
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "active_levels": active_levels_train,
        }

        args.model = HMCLocalModel(**params).to(args.device)

        optimizer = torch.optim.Adam(
            args.model.parameters(),
            lr=lr_by_level,
            weight_decay=weight_decay,
        )
        args.optimizer = optimizer

        if torch.cuda.is_available():
            args.model = args.model.to(args.device)
            args.criterions = [
                criterion.to(args.device) for criterion in args.criterions
            ]

        args.early_stopping_patience = 3
        args.patience_counters = [0] * args.hmc_dataset.max_depth
        args.level_active = [False] * args.hmc_dataset.max_depth
        args.level_active[level] = True

        args.best_val_loss = [float("inf")] * args.max_depth
        logging.info(f"Best val loss created {args.best_val_loss}")

        args.epochs_to_eval = 10
        args.count_epochs_eval = 0

        for epoch in range(1, args.epochs + 1):
            args.model.train()
            local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
            for inputs, targets, _ in args.train_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to("cuda"), [
                        target.to("cuda") for target in targets
                    ]
                output = args.model(inputs.float())

                # Zerar os gradientes antes de cada batch
                args.optimizer.zero_grad()
                target = targets[level].float()

                loss = args.criterions[level](output, target)
                local_train_losses[level] += loss

            # Backward pass (cálculo dos gradientes)
            for total_loss in local_train_losses:
                if total_loss > 0:
                    total_loss.backward()
            args.optimizer.step()

            local_train_losses = [
                loss / len(args.train_loader) for loss in local_train_losses
            ]
            non_zero_losses = [loss for loss in local_train_losses if loss > 0]
            global_train_loss = (
                sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0
            )

            logging.info(f"Epoch {epoch}/{args.epochs}")
            show_local_losses(local_train_losses, set="Train")
            show_global_loss(global_train_loss, set="Train")

            local_val_losses, local_val_precision = val_optimizer(args)
            logging.info(f"Local loss: {local_val_losses}")
            logging.info(f"Local precision: {local_val_precision}")

        return local_val_precision[level]

    best_params_per_level = {}

    create_dir("results/hpo")

    for level in args.active_levels:
        args.level = level
        logging.info(f"\n🔍 Optimizing hyperparameters for level {level}...\n")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, level), n_trials=args.n_trials)
        level_parameters = {
            "hidden_dim": study.best_params[f"hidden_dim_level_{level}"],
            "lr": study.best_params[f"lr_level_{level}"],
            "dropout": study.best_params[f"dropout_level_{level}"],
            "num_layers": study.best_params[f"num_layers_level_{level}"],
            "weight_decay": study.best_params[f"weight_decay_level_{level}"],
        }
        best_params_per_level[level] = level_parameters

        logging.info(f"✅ Best hyperparameters for level {level}: {study.best_params}")

    job_id = create_job_id_name(prefix="hpo")

    save_dict_to_json(
        best_params_per_level,
        f"results/hpo/best_params_{args.dataset_name}-{job_id}.json",
    )

    return best_params_per_level


def val_optimizer(args):
    args.model.eval()
    local_val_loss = 0.0
    output_val = 0.0
    y_val = 0.0
    local_val_precision = 0.0

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.to("cuda"), [
                    target.to("cuda") for target in targets
                ]
            output = args.model(inputs.float())

            target = targets[args.level].float()
            local_val_loss += args.criterions[args.level](output, target)

            if i == 0:
                output_val = output.to("cpu")
                y_val = target.to("cpu")
            else:
                output_val = torch.cat(
                    (output_val, output.to("cpu")), dim=0
                )
                y_val = torch.cat((y_val, target.to("cpu")), dim=0)

    local_val_precision = average_precision_score(
        y_val, output_val, average="micro"
    )

    local_val_loss = local_val_loss / len(args.val_loader)
    logging.info(f"Levels to evaluate: {args.active_levels}")
    for i in args.active_levels:
        if round(local_val_loss.item(), 3) < round(args.best_val_loss[i], 3):
            args.best_val_loss[i] = round(local_val_loss.item(), 3)
            args.patience_counters[i] = 0
            logging.info(f"Level {i}: improved (loss={local_val_loss:.4f})")
        else:
            args.patience_counters[i] += 1
            logging.info(
                f"Level {i}: no improvement \
                (patience {args.patience_counters[i]}/{args.early_stopping_patience})"
            )
            if args.patience_counters[i] >= args.early_stopping_patience:
                args.level_active[i] = False
                logging.info(
                    f"🚫 Early stopping triggered for level {i} — freezing its parameters"
                )
                # ❄️ Congelar os parâmetros desse nível
                for param in args.model.levels[i].parameters():
                    param.requires_grad = False
    return local_val_loss, local_val_precision

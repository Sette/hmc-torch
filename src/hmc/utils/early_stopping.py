import json
import logging
import os
import torch
from datetime import datetime

import numpy as np


def check_metrics(metric, best_metric, metric_type="loss"):
    if metric_type == "loss":
        if metric < best_metric:
            return True
        else:
            return False
    elif metric_type == "f1-score":
        if metric > best_metric:
            return True
        else:
            return False
    else:
        return False


def check_early_stopping_normalized(args, active_levels=[], save_model=True):
    """
    Checks if early stopping criteria are met for each active level.
    Args:
        args: An object containing all necessary arguments and attributes.
    """

    for level in active_levels:
        if args.level_active[level]:
            if args.best_model[level] is None:
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                logging.info("Level %d: initialized best model", level)
            loss = round(args.local_val_losses[level], 4)
            best_loss = args.best_val_loss[level]
            metric = round(args.local_val_score[level], 4)
            best_metric = args.best_val_score[level]

            if args.epoch % args.epochs_to_evaluate_f1 == 0:

                is_better_metric = check_metrics(
                    metric,
                    best_metric,
                    metric_type="f1-score",
                )

                logging.info(
                    "Is better level %d f1 %s",
                    level,
                    is_better_metric,
                )

            is_better_loss = check_metrics(
                loss,
                best_loss,
                metric_type="loss",
            )

            logging.info(
                "Is better level %d loss %s",
                level,
                is_better_loss,
            )

            if is_better_loss:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[level] = round(args.local_val_losses[level], 4)
                args.patience_counters[level] = 0
                logging.info(
                    "Level %d: improved (Loss=%.4f)",
                    level,
                    round(args.local_val_losses[level], 4),
                )

            else:
                # Incrementar o contador de paciência
                args.patience_counters[level] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level,
                    args.patience_counters[level],
                    args.early_stopping_patience,
                )

                if args.patience_counters[level] >= args.early_stopping_patience:
                    args.level_active[level] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d by loss\
                            — freezing its parameters",
                        level,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(level)].parameters():
                        param.requires_grad = False

        if args.epoch % args.epochs_to_evaluate_f1 == 0:
            if is_better_metric:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_score[level] = round(args.local_val_score[level], 4)
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                args.patience_counters_f1[level] = 0
                logging.info(
                    "Level %d: improved (F1 score=%.4f)",
                    level,
                    round(args.local_val_score[level], 4),
                )
                if save_model:
                    # Salvar em disco
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.levels[str(level)].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            else:
                # Incrementar o contador de paciência
                args.patience_counters_f1[level] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level,
                    args.patience_counters_f1[level],
                    args.early_stopping_patience_f1,
                )
                if args.patience_counters_f1[level] >= args.early_stopping_patience_f1:
                    args.level_active[level] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d by loss\
                            — freezing its parameters",
                        level,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(level)].parameters():
                        param.requires_grad = False


def check_early_stopping(args, active_levels, save_model=True):
    """
    Checks if early stopping criteria are met for each active level.
    Args:
        args: An object containing all necessary arguments and attributes.
    """

    for level in active_levels:
        if args.level_active[level]:
            if args.best_model[level] is None:
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                logging.info("Level %d: initialized best model", level)
            best_value, local_value = 0, 0

            if args.early_metric == "loss":
                local_value = round(args.local_val_losses[level], 4)
                best_value = args.best_val_loss[level]
            elif args.early_metric == "f1-score":
                local_value = round(args.local_val_score[level], 4)
                best_value = args.best_val_score[level]

            is_better = check_metrics(
                local_value, best_value, metric_type=args.early_metric
            )
            logging.info(
                "Is better %s level %d f1 %s", args.early_metric, level, is_better
            )

            if is_better:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[level] = round(args.local_val_losses[level], 4)
                args.best_val_score[level] = round(args.local_val_score[level], 4)
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                args.patience_counters[level] = 0
                logging.info(
                    "Level %d: improved (F1 score=%.4f)",
                    level,
                    round(args.local_val_score[level], 4),
                )
                if save_model:
                    # Salvar em disco
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.levels[str(level)].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            else:
                # Incrementar o contador de paciência
                args.patience_counters[level] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level,
                    args.patience_counters[level],
                    args.early_stopping_patience,
                )
                if args.patience_counters[level] >= args.early_stopping_patience:
                    args.level_active[level] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d\
                            — freezing its parameters",
                        level,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(level)].parameters():
                        param.requires_grad = False

import logging
import os
import torch

import numpy as np


def check_metrics(
        metric: float, 
        best_metric: float, 
        metric_type: str = "loss",
        ) -> bool:
    """Checks if the current metric is better than the best metric based on the specified metric type.  
    Args:
        metric (float): The current metric value to compare.
        best_metric (float): The best metric value observed so far.
        metric_type (str, optional): The type of metric being evaluated. 
            Can be either "loss" (where lower is better) or "score" (where higher is better). 
            Defaults to "loss".
    Returns:
        bool: True if the current metric is better than the best metric based on the specified metric type, False otherwise.
    """
    if np.isnan(best_metric):
        return True

    if metric_type == "loss":
        if metric < best_metric:
            return True
        else:
            return False
    elif metric_type == "score":
        if metric > best_metric:
            return True
        else:
            return False
    else:
        return False


def check_early_stopping_normalized(
        args: object,
        active_levels: list[int],
        save_model: bool = True,
        ) -> None:
    """
    Checks if early stopping criteria are met for each active level.
    Args:
        args: An object containing all necessary arguments and attributes.
    """
    args.level_active = list(args.level_active)  
    for level in active_levels:
        if args.level_active[level]:
            if args.best_model[level] is None:
                args.best_model[level] = args.model.levels[level]['level_classifier'].state_dict()
                logging.info("Level %d: initialized best model", level)
                if save_model:
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.levels[level]['level_classifier'].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            loss = round(args.local_val_losses[level], 4)
            best_loss = args.best_val_loss[level]
            metric = round(args.local_val_scores[level], 4)
            best_metric = args.best_val_score[level]

            is_better_metric = check_metrics(
                metric,
                best_metric,
                metric_type="score",
            )

            logging.info(
                "Is better level %d %s? %s",
                level,
                args.early_metric,
                is_better_metric,
            )

            is_better_loss = check_metrics(
                loss,
                best_loss,
                metric_type="loss",
            )

            logging.info(
                "Is better level %d loss? %s",
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
                    args.model.level_active[level] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d by loss\
                            — freezing its parameters",
                        level,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[level]['level_classifier'].parameters():
                        param.requires_grad = False

            if is_better_metric:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_score[level] = round(args.local_val_scores[level], 4)
                args.best_model[level] = args.model.levels[level]['level_classifier'].state_dict()
                args.patience_counters_score[level] = 0
                logging.info(
                    "Level %d: improved (%s=%.4f)",
                    level,
                    args.early_metric,
                    round(args.local_val_scores[level], 4),
                )
                if save_model:
                    # Salvar em disco
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.levels[level]['level_classifier'].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            else:
                # Incrementar o contador de paciência
                args.patience_counters_score[level] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level,
                    args.patience_counters_score[level],
                    args.early_stopping_patience_score,
                )
                if (
                    args.patience_counters_score[level]
                    >= args.early_stopping_patience_score
                ):
                    args.level_active[level] = False
                    args.model.level_active[level] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d by %s\
                            — freezing its parameters",
                        level,
                        args.early_metric,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(level)].parameters():
                        param.requires_grad = False



def check_early_stopping_tabat(
        args: object,
        active_levels: list[int],
        save_model: bool = True,
        ) -> None:
    """
    Checks if early stopping criteria are met for each active level.
    Args:
        args: An object containing all necessary arguments and attributes.
    """
    args.level_active = list(args.level_active)  
    for level in active_levels:
        if args.level_active[level]:
            if args.best_model[level] is None:
                args.best_model[level] = args.model.heads[level].state_dict()
                logging.info("Level %d: initialized best model", level)
                if save_model:
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.heads[level].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            loss = round(args.local_val_losses[level], 4)
            best_loss = args.best_val_loss[level]
            metric = round(args.local_val_scores[level], 4)
            best_metric = args.best_val_score[level]

            is_better_metric = check_metrics(
                metric,
                best_metric,
                metric_type="score",
            )

            logging.info(
                "Is better level %d %s? %s",
                level,
                args.early_metric,
                is_better_metric,
            )

            is_better_loss = check_metrics(
                loss,
                best_loss,
                metric_type="loss",
            )

            logging.info(
                "Is better level %d loss? %s",
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
                    args.model.level_active[level] = False
                    logging.info(
                        "🚫 Early stopping triggered for level %d by loss\
                            — freezing its parameters",
                        level,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.heads[level].parameters():
                        param.requires_grad = False

            if is_better_metric:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_score[level] = round(args.local_val_scores[level], 4)
                args.best_model[level] = args.model.heads[level].state_dict()
                args.patience_counters_score[level] = 0
                logging.info(
                    "Level %d: improved (%s=%.4f)",
                    level,
                    args.early_metric,
                    round(args.local_val_scores[level], 4),
                )
                if save_model:
                    # Salvar em disco
                    logging.info("Saving best model for Level %d", level)
                    torch.save(
                        args.model.heads[level].state_dict(),
                        os.path.join(
                            args.results_path, f"best_model_level_{level}.pth"
                        ),
                    )
                    logging.info("best model updated and saved for Level %d", level)

            else:
                # Incrementar o contador de paciência
                args.patience_counters_score[level] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level,
                    args.patience_counters_score[level],
                    args.early_stopping_patience_score,
                )
                if (
                    args.patience_counters_score[level]
                    >= args.early_stopping_patience_score
                ):
                    args.level_active[level] = False
                    args.model.level_active[level] = False
                    logging.info(
                        "🚫 Early stopping triggered for level %d by %s\
                            — freezing its parameters",
                        level,
                        args.early_metric,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.heads[level].parameters():
                        param.requires_grad = False

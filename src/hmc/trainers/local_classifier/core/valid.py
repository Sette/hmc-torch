import logging
import os
import torch
from sklearn.metrics import precision_recall_fscore_support

from hmc.utils.dir import create_dir
from hmc.trainers.losses import calculate_local_loss


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


def check_early_stopping(args, active_levels):
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
                # Salvar em disco
                logging.info("Saving best model for Level %d", level)
                torch.save(
                    args.model.levels[str(level)].state_dict(),
                    os.path.join(args.results_path, f"best_model_level_{level}.pth"),
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


def check_early_stopping_regularized(args, active_levels):
    """
    Checks if early stopping criteria are met for each active level.
    Args:
        args: An object containing all necessary arguments and attributes.
    """

    for level in active_levels:
        metric, best_metric, loss, best_loss = 0, 0, 0, 0
        if args.level_active[level]:
            if args.best_model[level] is None:
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                logging.info("Level %d: initialized best model", level)
            loss = round(args.local_val_losses[level], 4)
            best_loss = args.best_val_loss[level]
            metric = round(args.local_val_score[level], 4)
            best_metric = args.best_val_score[level]

            is_better_metric = check_metrics(
                metric, best_metric, metric_type="f1-score"
            )
            is_better_loss = check_metrics(loss, best_loss, metric_type="loss")
            logging.info("Is better level %d f1 %s", level, is_better_metric)
            logging.info("Is better level %d loss %s", level, is_better_loss)
            is_better, is_better_oposite = False, False
            if args.early_metric == "loss":
                is_better = is_better_loss
                is_better_oposite = is_better_metric
            elif args.early_metric == "f1-score":
                is_better = is_better_metric
                is_better_oposite = is_better_loss
            elif args.early_metric == "both":
                is_better = is_better_loss and is_better_metric

            if is_better:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[level] = loss
                args.best_val_score[level] = metric
                args.best_model[level] = args.model.levels[str(level)].state_dict()
                args.patience_counters[level] = 0
                logging.info("Level %d: improved (F1 score=%.4f)", level, metric)
                # Salvar em disco
                logging.info("Saving best model for Level %d", level)
                torch.save(
                    args.model.levels[str(level)].state_dict(),
                    os.path.join(args.results_path, f"best_model_level_{level}.pth"),
                )
                logging.info("best model updated and saved for Level %d", level)

            else:
                if not is_better_oposite:
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


def valid_step(args):
    """
    Performs a validation step for a hierarchical multi-level classifier model.
    Args:
        args: An object containing all necessary arguments and attributes, including:
            - model: The model to evaluate, with attribute `levels` for each depth.
            - val_loader: DataLoader for the validation dataset.
            - criterions: List of loss functions, one per level.
            - device: Device to run computations on.
            - max_depth: Maximum number of levels in the hierarchy.
            - active_levels: List of indices for currently active levels.
            - best_val_loss: List of best validation losses per level.
            - best_model: List to store the best model state_dict per level.
            - patience_counters: List of patience counters for early \
                stopping per level.
            - early_stopping_patience: Number of epochs to wait before \
                early stopping.
            - level_active: List indicating if a level is still active.
    Returns:
        tuple:
            - local_val_losses (list of float): Average validation loss per level.
            - local_val_precision (list of float): Average precision score per level.
    Side Effects:
        - Updates `args.best_val_loss`, `args.best_model`, and \
            `args.patience_counters` for improved levels.
        - Freezes parameters of levels that triggered early stopping by \
            setting `requires_grad` to False.
        - Logs progress and early stopping events.
    """

    args.model.eval()

    args.results_path = f"results/train/{args.method}-{args.dataset_name}-{args.early_metric}/{args.job_id}"

    args.result_path = "results/train/%s-%s-%s-%s/%s" % (
        args.method,
        args.dataset_name,
        args.early_metric,
        args.early_stopping_patience,
        args.job_id,
    )

    local_inputs = {
        level: torch.tensor([]) for _, level in enumerate(args.active_levels)
    }
    local_outputs = {
        level: torch.tensor([]) for _, level in enumerate(args.active_levels)
    }

    threshold = 0.2

    # Get local scores
    args.local_val_score = [0.0] * args.max_depth
    args.local_val_losses = [0.0] * args.max_depth

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            # Se ainda estamos no warm-up, só treine o nível 0
            if args.epoch <= args.n_warmup_epochs:
                index = 0
                output = outputs[index].double()
                target = targets[index].double()
                loss = args.criterions[index](output, target)
                args.local_val_losses[index] += loss.item()
            else:

                for level in args.active_levels:
                    if args.level_active[level]:
                        loss = calculate_local_loss(
                            outputs,
                            targets,
                            args,
                            level,
                        )
                        args.local_val_losses[level] += loss.item()

                        # *** Para métricas e concatenação ***
                        # Se quiser outputs binários para avaliação:
                        binary_outputs = (outputs[level] > threshold).float()

                        # Acumulação de outputs e targets para métricas
                        if i == 0:  # Primeira iteração: inicia tensor
                            local_outputs[level] = binary_outputs
                            local_inputs[level] = targets[level]
                        else:  # Nas seguintes, empilha ao longo do batch
                            local_outputs[level] = torch.cat(
                                (local_outputs[level], binary_outputs), dim=0
                            )
                            local_inputs[level] = torch.cat(
                                (local_inputs[level], targets[level]), dim=0
                            )

    if args.epoch <= args.n_warmup_epochs:
        active_levels = [0]
    else:
        active_levels = args.active_levels
    for idx in active_levels:
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

            args.local_val_score[idx] = score[2]

    create_dir(args.results_path)

    args.local_val_losses = [
        loss / len(args.val_loader) for loss in args.local_val_losses
    ]
    logging.info("Levels to evaluate: %s", args.active_levels)
    check_early_stopping_regularized(args, active_levels)

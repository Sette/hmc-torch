import logging
import os
import torch
from sklearn.metrics import precision_recall_fscore_support

from hmc.utils.dir import create_dir


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
            - patience_counters: List of patience counters for early stopping per level.
            - early_stopping_patience: Number of epochs to wait before early stopping.
            - level_active: List indicating if a level is still active.
    Returns:
        tuple:
            - local_val_losses (list of float): Average validation loss per level.
            - local_val_precision (list of float): Average precision score per level.
    Side Effects:
        - Updates `args.best_val_loss`, `args.best_model`, and `args.patience_counters` for improved levels.
        - Freezes parameters of levels that triggered early stopping by setting `requires_grad` to False.
        - Logs progress and early stopping events.
    """

    args.model.eval()
    local_val_losses = [0.0] * args.max_depth

    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    # Get local scores
    local_val_score = {level: None for _, level in enumerate(args.active_levels)}
    threshold = 0.2

    R_global = args.hmc_dataset.R.to(args.device)
    R_global = R_global.squeeze(0)

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            total_loss = 0.0

            for index in args.active_levels:
                if args.level_active[index]:
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
            y_pred = local_outputs[idx].to('cpu').int().numpy()
            y_true = local_inputs[idx].to('cpu').int().numpy()

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
        if args.level_active[i]:
            if args.best_model[i] is None:
                args.best_model[i] = args.model.levels[str(i)].state_dict()
                logging.info("Level %d: initialized best model", i)
            if round(local_val_losses[i], 4) < args.best_val_loss[i]:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[i] = round(local_val_losses[i], 4)
                args.best_val_score[i] = round(local_val_score[i], 4)
                args.best_model[i] = args.model.levels[str(i)].state_dict()
                args.patience_counters[i] = 0
                logging.info(
                    "Level %d: improved (F1 score=%.4f)", i, local_val_score[i]
                )
                # Salvar em disco
                logging.info("Saving best model for Level %d", i)
                torch.save(
                    args.model.levels[str(i)].state_dict(),
                    os.path.join(results_path, f"best_model_baseline_level_{i}.pth"),
                )
                logging.info("best model updated and saved for Level %d", i)

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
                        "🚫 Early stopping triggered for level %d — freezing its parameters",
                        i,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(i)].parameters():
                        param.requires_grad = False

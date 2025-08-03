import logging

import torch
from sklearn.metrics import precision_recall_fscore_support

from hmc.model.local_classifier.constrained.utils import get_constr_out

import networkx as nx
import numpy as np

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


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
    y_val = [0.0] * args.max_depth

    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    # Get local scores
    local_val_score = {level: None for _, level in enumerate(args.active_levels)}
    threshold = 0.3

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    R = np.zeros(args.hmc_dataset.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        args.hmc_dataset.A
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(args.hmc_dataset.A)):
        # here we need to use the function nx.descendants() \
        # because in the directed graph the edges have source \
        # from the descendant and point towards the ancestor
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(args.device)

    class_indices_per_level = {
        lvl: torch.tensor(
            [
                args.hmc_dataset.nodes_idx[n.replace("/", ".")]
                for n in args.hmc_dataset.levels[lvl]
            ],
            device=args.device,
        )
        for lvl in args.hmc_dataset.levels.keys()
    }

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            for index in args.active_levels:
                if args.level_active[index]:
                    output = outputs[str(index)]
                    target = targets[index]
                    child_indices = class_indices_per_level[
                        index
                    ]  # [n_classes_nivel_atual]
                    # MCLoss
                    if index == 0:
                        loss = args.criterions[index](output.double(), target)
                    else:
                        # Índices globais dos pais para cada amostra
                        parent_target = targets[index - 1]
                        parent_indices = class_indices_per_level[index - 1]
                        parent_index_each_sample = parent_target.argmax(dim=1)
                        parent_global_idxs = parent_indices[parent_index_each_sample]
                        # Constrói a máscara usando R_global (shape: [1, n, n])
                        mask = torch.stack(
                            [
                                R[0, child_indices, parent_global_idxs[b]]
                                for b in range(output.shape[0])
                            ],
                            dim=0,
                        )  # [batch, n_classes_nivel_atual]
                        masked_output = output + (1 - mask) * (-1e9)
                        loss = args.criterions[index](
                            torch.sigmoid(masked_output), target
                        )

                    if i == 0:
                        local_outputs[index] = output.to("cpu")
                        y_val[index] = target.to("cpu")
                    else:
                        local_outputs[index] = torch.cat(
                            (local_outputs[index], output.to("cpu")), dim=0
                        )
                        y_val[index] = torch.cat(
                            (y_val[index], target.to("cpu")), dim=0
                        )
    for level_idx in args.active_levels:
        if args.level_active[level_idx]:
            y_pred_binary = local_outputs[level_idx].data > threshold

            score = precision_recall_fscore_support(
                y_val[level_idx], y_pred_binary, average="micro"
            )
            # local_val_score[idx] = score
            logging.info(
                "Level %d: Validation set precision=%.4f, recall=%.4f, f1-score=%.4f",
                level_idx,
                score[0],
                score[1],
                score[2],
            )

            local_val_score[level_idx] = round(score[2], 4)

    local_val_losses = [loss / len(args.val_loader) for loss in local_val_losses]
    logging.info("Levels to evaluate: %s", args.active_levels)
    for level_idx in args.active_levels:
        if args.level_active[level_idx]:
            if args.best_model[level_idx] is None:
                args.best_model[level_idx] = args.model.levels[
                    str(level_idx)
                ].state_dict()
                logging.info("Level %d: initialized best model", level_idx)
            if local_val_score[level_idx] > args.best_val_score[level_idx]:
                # Atualizar o melhor modelo e as melhores métricas
                args.best_val_loss[level_idx] = round(local_val_losses[level_idx], 4)
                args.best_val_score[level_idx] = local_val_score[level_idx]
                args.best_model[level_idx] = args.model.levels[
                    str(level_idx)
                ].state_dict()
                args.patience_counters[level_idx] = 0
                logging.info(
                    "Level %d: improved (F1 score=%.4f)",
                    level_idx,
                    local_val_score[level_idx],
                )
                # Salvar em disco
                logging.info("Saving best model for Level %d", i)
                torch.save(
                    args.model.levels[str(i)].state_dict(),
                    f"best_model_mask_level_{i}.pth",
                )
                logging.info("best model updated and saved for Level %d", i)

            else:
                # Incrementar o contador de paciência
                args.patience_counters[level_idx] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    level_idx,
                    args.patience_counters[level_idx],
                    args.early_stopping_patience,
                )
                if args.patience_counters[level_idx] >= args.early_stopping_patience:
                    args.level_active[level_idx] = False
                    # args.active_levels.remove(level_idx)
                    logging.info(
                        "🚫 Early stopping triggered for level %d — freezing its parameters",
                        level_idx,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[str(level_idx)].parameters():
                        param.requires_grad = False
    return None

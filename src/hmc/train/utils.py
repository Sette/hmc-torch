import json
import logging
from datetime import datetime

import numpy as np


def create_job_id_name(prefix="job"):
    """
    Create a unique job ID using the current date and time.

    Args:
        prefix (str): Optional prefix for the job ID (default is "job").

    Returns:
        str: A unique job ID string.
    """
    now = datetime.now()
    job_id = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    return job_id


def local_to_global_predictions(local_labels, local_nodes_idx, nodes_idx):
    n_samples = local_labels[0].shape[0]
    n_global_labels = len(nodes_idx)
    global_preds = np.zeros((n_samples, n_global_labels))
    sorted_levels = sorted(local_nodes_idx.keys())
    local_nodes_reverse = {
        level: {v: k for k, v in local_nodes_idx[level].items()}
        for level in sorted_levels
    }
    # logging.info(f"Local nodes idx: {local_nodes_idx}")
    # logging.info(f"Local nodes reverse: {local_nodes_reverse}")

    logging.info(f"Exemplos: {n_samples}")
    # logging.info(f"Shape local_preds: {len(local_labels)}")
    # logging.info(f"Local nodes idx: {local_nodes_reverse}")

    # Etapa 1: montar node_names ativados por exemplo
    activated_nodes_by_example = [[] for _ in range(n_samples)]

    for level_index, level in enumerate(sorted_levels):
        level_preds = local_labels[
            level_index
        ]  # shape: [n_samples, n_classes_at_level]
        for idx_example, label in enumerate(level_preds):
            local_indices = np.where(label == 1)[0]  # aceita floats ou binários
            for local_idx in local_indices:
                node_name = local_nodes_reverse[level].get(local_idx)
                if node_name:
                    activated_nodes_by_example[idx_example].append(node_name)
                else:
                    logging.info(
                        f"[WARN] Índice local {local_idx} não encontrado no nível {level}"
                    )

    # logging.info(f"Node names ativados por exemplo: {activated_nodes_by_example[0]}")
    global_indices = []
    for node in activated_nodes_by_example[0]:
        # logging.info(f"Node names ativados: {node}")
        if "/" in node:
            node = node.replace("/", ".")
        global_indices.append(nodes_idx.get(node))
    logging.info(global_indices)
    # Etapa 2: converter node_names para índices globais
    for idx_example, node_names in enumerate(activated_nodes_by_example):
        for node_name in node_names:
            key = node_name.replace("/", ".")
            if key in nodes_idx:
                global_idx = nodes_idx[key]
                global_preds[idx_example][global_idx] = 1
            else:
                logging.info(f"[WARN] Node '{key}' não encontrado em nodes_idx")

    return global_preds


def show_metrics(losses, scores, dataset="Train"):
    """
    Show metrics for losses and scores.

    Args:
        losses (list): List of local losses.
        scores (dict): Dictionary of local scores.
        dataset (str): Dataset name (default is "Train").
    """
    show_local_losses(losses, dataset)
    show_local_score(scores, dataset)


def show_local_losses(local_losses, dataset="Train"):
    formatted_string = ""
    for level, local_loss in enumerate(local_losses):
        if local_loss is not None and local_loss != 0.0:
            formatted_string += "level %d: %.4f // " % (
                level,
                local_loss,
            )
    logging.info(formatted_string)


def show_global_loss(global_loss, dataset="Train"):
    logging.info("Global average loss %s Loss: %s", dataset, global_loss)


def show_local_score(local_scores, dataset="Train"):
    formatted_string = ""
    for level, local_score in local_scores.items():
        if local_score is not None and local_score != 0.0:
            formatted_string += "level %s score %s // " % (
                level,
                local_score,
            )

    logging.info(formatted_string)


def save_dict_to_json(dictionary, file_path):
    """
    Saves a dictionary to a JSON file.

    Args:
        dictionary (dict): The dictionary to be saved.
        file_path (str): The path to the JSON file where the dictionary will be saved.

    Raises:
        TypeError: If the dictionary contains non-serializable objects.
        OSError: If the file cannot be written.

    Example:
        save_dict_to_json({'a': 1, 'b': 2}, 'output.json')
    """
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)

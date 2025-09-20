import pickle
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


logging.basicConfig(level=logging.INFO)

def get_structure(genres_id, df_genres):
    """
    Retrieve the hierarchical structure (ancestry) of each genre_id from the DataFrame.

    Args:
        genres_id (list or iterable): List of genre IDs to retrieve the structure for.
        df_genres (pd.DataFrame): DataFrame containing the columns "genre_id" and "parent".
            It is expected that "parent" is the parent genre_id for each genre.

    Returns:
        list: A list where each element is a list of genre IDs representing the hierarchy (ancestry)
              for the corresponding genre_id in `genres_id`. Each list starts from the genre_id and
              recursively adds its parents up to the root (where parent is 0 or missing).
    """

    def get_from_df(genre_id, df_genres, output):
        """
        Recursively retrieves the ancestry of a given genre_id.

        Args:
            genre_id (int): The current genre ID.
            df_genres (pd.DataFrame): DataFrame containing genre hierarchy.
            output (list): The list being built up with the ancestry.

        Returns:
            list: List including this genre_id and its ancestors up to the root.
        """
        if genre_id != 0:
            parent_genre = df_genres[df_genres["genre_id"] == genre_id].parent.values[0]
            output.append(genre_id)
            get_from_df(parent_genre, df_genres, output=output)
            return output

    output_list = []
    for genre_id in genres_id:
        output_list.append(get_from_df(genre_id, df_genres, output=[]))
    return output_list


def group_labels_by_level(df, max_depth):
    """
    Groups hierarchical labels in the DataFrame by each level, up to the specified max_depth.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'y_true', where each item is a list of hierarchical label lists per track.
        max_depth (int): Maximum number of hierarchy levels to consider.

    Returns:
        list: A list of length max_depth, where each element is a list of unique labels per example at that level.
    """
    # Initialize empty lists for each level based on max_depth
    levels = [[] for _ in range(max_depth)]

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Iterate over each level and append the labels to the corresponding list
        for level in range(max_depth):
            level_labels = []
            for label in row["y_true"]:
                if level < len(label):
                    level_labels.append(label[level])
            levels[level].append(list(set(level_labels)))

    # Return the grouped labels by level
    return levels


def binarize_labels(dataset_df, args):
    """
    Binarizes multi-level hierarchical labels in the dataset, using MultiLabelBinarizer per level.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing 'y_true' column with hierarchical label lists.
        args (Namespace): Arguments namespace with:
            - max_depth (int): Maximum hierarchy depth.
            - mlb_path (str): Path to save the list of fitted MultiLabelBinarizer objects.

    Returns:
        pd.DataFrame: DataFrame with columns ['track_id', 'y_true', 'all_binarized'], where 'all_binarized' contains binarized label lists per level.
    """
    # Labels
    mlbs = []

    grouped_labels = group_labels_by_level(dataset_df, args.max_depth)

    labels_name = []
    for level, level_labels in enumerate(grouped_labels):
        labels_name.append(f"level{level + 1}")
        # Cria e aplica o MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binary_labels = mlb.fit_transform(level_labels).tolist()

        mlbs.append(mlb)
        for i in range(len(dataset_df)):
            # Verifica se o índice está dentro do limite
            if i < len(binary_labels):
                # Adiciona os rótulos binarizados à lista
                binary_labels[i] = binary_labels[i]
            else:
                # Se o índice estiver fora do limite, adiciona uma lista vazia
                binary_labels[i] = [0] * len(mlb.classes_)

        dataset_df.loc[:, labels_name[level]] = binary_labels

    # Serializar a lista de mlb
    with open(args.mlb_path, "wb") as file:
        pickle.dump(mlbs, file)

    dataset_df["all_binarized"] = dataset_df.apply(lambda row: row[labels_name], axis=1)
    tracks_df = dataset_df[["track_id", "y_true", "all_binarized"]]
    return tracks_df



def local_to_global_predictions(local_labels, local_nodes_idx, nodes_idx, is_go=True):
    """
    Converts local-level predictions to global predictions.

    Args:
        local_labels (list of np.array): List where each element is an array of shape [n_samples, n_classes_at_level],
                                         containing the local (per-level) binary predictions.
        local_nodes_idx (dict): Dictionary mapping level name to {node_name_local: idx_local} dicts.
        nodes_idx (dict): Dictionary mapping global node names to their global indices.

    Returns:
        np.ndarray: Array of shape [n_samples, n_global_labels] with the corresponding global binary predictions.
    """
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

    logging.info("Exemplos: %d", n_samples)
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
                        "[WARN] Índice local %d não encontrado no nível %d ",
                        local_idx,
                        level,
                    )

    # logging.info(f"Node names ativados por exemplo: {activated_nodes_by_example[0]}")
    global_indices = []
    for node in activated_nodes_by_example[0]:
        # logging.info(f"Node names ativados: {node}")
        if "/" in node:
            node = node.replace("/", ".")
        global_indices.append(nodes_idx.get(node))
    # Etapa 2: converter node_names para índices globais
    for idx_example, node_names in enumerate(activated_nodes_by_example):
        for node_name in node_names:
            node_name = node_name.replace("/", ".")
            node_name_parsed = []
            if len(node_name.split(".")) > 1:
                node_name_parsed.append(node_name.split("."))
            else:
                node_name_parsed.append(node_name)

            for key in node_name.split("."):
                if key in nodes_idx:
                    global_idx = nodes_idx[key]
                    global_preds[idx_example][global_idx] = 1
                else:
                    logging.info(f"[WARN] Node '{key}' não encontrado em nodes_idx")

    return global_preds


def global_to_local_predictions(global_preds, local_nodes_idx, nodes_idx):
    """
    Parâmetros:
        global_preds: np.array [n_samples, n_global_labels] \
            - rótulos/predições globais binárias
        local_nodes_idx: dict[level_name] = dict[node_name_local \
            -> idx_local]
        nodes_idx: dict[node_name_global -> idx_global]
    Retorna:
        local_labels: lista np.array [n_samples, n_local_labels_por_nivel]
    """
    # Inverter nodes_idx: global_idx -> node_name
    idx_to_node = {v: k for k, v in nodes_idx.items()}
    sorted_levels = sorted(local_nodes_idx.keys())

    n_samples = global_preds.shape[0]

    local_labels = []
    for level in sorted_levels:
        n_classes_local = len(local_nodes_idx[level])
        lvl_preds = np.zeros((n_samples, n_classes_local))
        # Inverter local_nodes_idx[level]: node_name_local->idx_local
        node_to_local_idx = local_nodes_idx[level]  # node_name_local → idx_local

        for sample_idx in range(n_samples):
            # Quais globais estão ativados neste sample
            active_globals = np.where(global_preds[sample_idx] == 1)[0]
            for global_idx in active_globals:
                node_name = idx_to_node[global_idx]
                # Ajustar para nomes locais, se necessário
                node_name_local = node_name.replace(".", "/")
                # Verifica se é nó deste nível
                if node_name_local in node_to_local_idx:
                    local_idx = node_to_local_idx[node_name_local]
                    lvl_preds[sample_idx, local_idx] = 1
        local_labels.append(lvl_preds)
    return local_labels


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
    """
    Logs the local (per-level) losses for a given dataset.

    Args:
        local_losses (list): A list containing the loss value for each hierarchy level.
        dataset (str): The name of the dataset, e.g., "Train", "Validation", or "Test". Defaults to "Train".

    Returns:
        None
    """
    formatted_string = ""
    for level, local_loss in enumerate(local_losses):
        if local_loss is not None and local_loss != 0.0:
            formatted_string += "level %d: %.4f // " % (
                level,
                local_loss,
            )
    logging.info(formatted_string)


def show_global_loss(global_loss, dataset="Train"):
    """
    Logs the global (average) loss for a given dataset.

    Args:
        global_loss (float): The global loss value.
        dataset (str): The name of the dataset, e.g., "Train", "Validation", or "Test". Defaults to "Train".

    Returns:
        None
    """
    logging.info("Global average loss %s Loss: %s", dataset, global_loss)


def show_local_score(local_scores, dataset="Train"):
    """
    Logs the local (per-level) scores for a given dataset.

    Args:
        local_scores (dict): A dictionary mapping each level to its corresponding score.
        dataset (str): The name of the dataset, e.g., "Train", "Validation", or "Test". Defaults to "Train".

    Returns:
        None
    """
    formatted_string = ""
    for level, local_score in local_scores.items():
        if local_score is not None and local_score != 0.0:
            formatted_string += "level %s score %s // " % (
                level,
                local_score,
            )

    logging.info(formatted_string)

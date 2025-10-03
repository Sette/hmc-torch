import pickle
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

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


def local_to_global_predictions(local_labels, local_nodes_idx, nodes_idx, threshold=0.6):
    """
    Converte previsões de nível local para vetores globais de scores e predições binárias.

    Esta função processa as previsões locais uma única vez e retorna tanto os scores
    contínuos quanto as predições binarizadas baseadas em um limiar.

    Args:
        local_labels (list of np.array): Lista onde cada elemento é um array de shape
                                         [n_samples, n_classes_at_level], contendo
                                         os scores locais.
        local_nodes_idx (dict): Dicionário mapeando nomes de níveis para seus índices de nós locais.
        nodes_idx (dict): Dicionário mapeando nomes de nós globais para seus índices globais.
        g_t: Parâmetro não utilizado nesta implementação, mantido para compatibilidade.
        threshold (float): O limiar a ser usado para converter scores em predições
                           binárias (0 ou 1). Padrão: 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Uma tupla contendo dois arrays:
            - global_scores (np.ndarray): Matriz de shape [n_samples, n_global_labels]
                                          com os scores contínuos.
            - global_binary_preds (np.ndarray): Matriz de shape [n_samples, n_global_labels]
                                                com as predições binárias (0 ou 1).
    """
    if not local_labels:
        return np.array([]), np.array([])

    n_samples = local_labels[0].shape[0]
    n_global_labels = len(nodes_idx)

    # Inicializa ambas as matrizes de saída
    global_scores = np.zeros((n_samples, n_global_labels))
    global_binary_preds = np.zeros((n_samples, n_global_labels), dtype=int)

    # Cria um mapeamento reverso de índice local para nome do nó para facilitar a busca
    sorted_levels = sorted(local_nodes_idx.keys())
    local_nodes_reverse = {
        level: {v: k for k, v in local_nodes_idx[level].items()}
        for level in sorted_levels
    }

    logging.info(f"Processando {n_samples} exemplos com threshold de {threshold}...")

    # Itera através de cada nível hierárquico
    for level_index, level in enumerate(sorted_levels):
        level_preds = local_labels[level_index]

        for idx_example, sample_scores in enumerate(level_preds):
            non_zero_indices = np.where(sample_scores > 0)[0]

            for local_idx in non_zero_indices:
                node_name = local_nodes_reverse[level].get(local_idx)
                if not node_name:
                    continue

                key = node_name.replace("/", ".")
                global_idx = nodes_idx.get(key)

                if global_idx is None:
                    continue  # O warning para isso pode ser muito verboso

                # 1. Atribui o score original à matriz de scores
                score = sample_scores[local_idx]
                global_scores[idx_example, global_idx] = score

                # 2. Binariza o score e atribui à matriz de predições binárias
                if score >= threshold:
                    global_binary_preds[idx_example, global_idx] = 1

    return global_scores, global_binary_preds

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


def apply_hierarchy_consistency(outputs, g, device, local_nodes_idx):
    """
    Apply hard hierarchy consistency to model outputs using a hierarchy graph g.

    Args:
    outputs (dict): Dictionary of tensors, one tensor per level (already with sigmoid applied
                    if the model includes it). Each tensor should have shape [n_samples, n_classes_at_level].
    g (networkx.DiGraph): Directed graph where nodes are tuples (level, class_idx) and edges represent
                          hierarchical relationships.
    device (torch.device): The device tensors are or should be moved to for computation.

    Returns:
    list of torch.Tensor: Adjusted outputs such that child classes do not contradict parent classes.
    """
    new_outputs = []

    # Cria um mapeamento reverso de índice local para nome do nó para facilitar a busca
    sorted_levels = sorted(local_nodes_idx.keys())
    local_nodes_reverse = {
        level: {v: k for k, v in local_nodes_idx[level].items()}
        for level in sorted_levels
    }

    for level_index, level in enumerate(sorted_levels):
        level_preds = outputs[level_index].to("cpu")
        if level == 0:
            new_outputs.append(level_preds)  # root has no parent, pass through directly
            continue

        print(level_preds)  # Debug output to trace values, can be removed in production
        mask = torch.ones_like(level_preds, device=device)

        for idx_example, sample_scores in enumerate(level_preds):
            non_zero_indices = np.where(sample_scores > 0)[0]

            for local_idx in non_zero_indices:
                node_name = local_nodes_reverse[level].get(local_idx)
                if not node_name:
                    continue

                key = node_name.replace("/", ".")
                print("nó analisado")
                print(key)

                parents = [p for p in g.predecessors(key)]

                print(parents)
                if parents:
                    # Gather the corresponding output values from the parent nodes
                    parent_vals = [outputs[(p_level)][:, p_idx] for p_level, p_idx in parents]

                    # Compute the maximum of the parent node values.
                    # This enforces that a child can be active only if at least one parent is active.
                    valid_parent = torch.stack(parent_vals).max(0).values

                    # Update the mask based on parent values
                    mask[:, key] = valid_parent

        # Apply the mask: Child class activations are limited by the max of the parent activations
        adjusted_output = outputs[level_index] * mask
        new_outputs.append(adjusted_output)

    return new_outputs

def hierarchy_regularization(outputs, g):
    """
    Penalize child probability > parent probability
    """
    penalty = 0.0
    for parent, child in g.edges():
        p_level, p_idx = parent
        c_level, c_idx = child

        parent_probs = torch.sigmoid(outputs[p_level][:, p_idx])
        child_probs = torch.sigmoid(outputs[c_level][:, c_idx])

        penalty += torch.mean(torch.relu(child_probs - parent_probs))
    return penalty


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

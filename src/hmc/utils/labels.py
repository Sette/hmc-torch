import pickle

from sklearn.preprocessing import MultiLabelBinarizer


def get_structure(genres_id, df_genres):
    def get_from_df(genre_id, df_genres, output):
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

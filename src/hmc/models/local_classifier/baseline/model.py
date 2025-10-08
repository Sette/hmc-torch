import logging

import torch.nn as nn


def transform_predictions(predictions):
    transformed = []
    # Loop through each index to form examples with the first element from each level
    for i in range(len(predictions[0])):  # Iterate over the number of examples
        example = []
        for level in predictions:  # Iterate over the levels
            example.append(level[i])  # Get the first element from each level at index i
        transformed.append(example)

    return transformed


class ExpandOutputClassification(nn.Module):
    def __init__(self, input_shape=512, output_shape=512):
        super().__init__()
        self.dense = nn.Linear(input_shape, output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x


class BuildClassification(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dims,
        output_size,
        dropout=0.5,
    ):
        super(BuildClassification, self).__init__()
        layers = []
        current_dim = int(input_shape)
        # Itera sobre a lista de dimensões ocultas
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(h_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = int(h_dim)  # A saída desta camada é a entrada da próxima

        layers.append(nn.Linear(current_dim, int(output_size)))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class HMCLocalModel(nn.Module):
    def __init__(
        self,
        levels_size,
        input_size=None,
        hidden_dims=None,
        num_layers=None,
        dropout=None,
        active_levels=None,
    ):
        super(HMCLocalModel, self).__init__()
        if not input_size:
            logging.info("input_size is None, error in HMCLocalClassificationModel")
            raise ValueError("input_size is None")
        if not levels_size:
            logging.info("levels_size is None, error in HMCLocalClassificationModel")
            raise ValueError("levels_size is None")
        if active_levels is None:
            active_levels = list(range(len(levels_size)))
            logging.info("active_levels is None, using all levels: %s", active_levels)

        self.input_size = input_size
        self.levels_size = levels_size
        self.mum_layers = num_layers
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.levels = nn.ModuleDict()
        self.active_levels = active_levels
        # if hpo:
        #     # levels_size = {level: levels_size for level in active_levels}
        #     dropout = {level: dropout for level in active_levels}
        #     num_layers = {level: num_layers for level in active_levels}
        #     hidden_dims = {level: hidden_dims for level in active_levels}

        self.max_depth = len(levels_size)

        logging.info(
            "HMCLocalModel: input_size=%s, levels_size=%s, "
            "hidden_dims=%s, num_layers=%s, dropout=%s, "
            "active_levels=%s",
            input_size,
            levels_size,
            hidden_dims,
            num_layers,
            dropout,
            active_levels,
        )
        for index in active_levels:
            self.levels[str(index)] = BuildClassification(
                input_shape=input_size,
                hidden_dims=hidden_dims[index],
                output_size=levels_size[index],
                dropout=dropout[index],
            )

    def forward(self, x):
        outputs = {}
        for index, level in self.levels.items():
            index = int(index)
            local_output = level(x)
            outputs[index] = local_output
        return outputs

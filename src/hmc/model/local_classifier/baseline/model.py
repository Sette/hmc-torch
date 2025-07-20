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
        hidden_size,
        output_size,
        nb_layers,
        dropout_rate=0.5,
    ):
        super(BuildClassification, self).__init__()
        layers = []
        layers.append(nn.Linear(input_shape, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(nb_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class HMCLocalModel(nn.Module):
    def __init__(
        self,
        levels_size,
        input_size=None,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        active_levels=None,
    ):
        super(HMCLocalModel, self).__init__()
        if not input_size:
            print("input_size is None, error in HMCLocalClassificationModel")
            raise ValueError("input_size is None")
        if not levels_size:
            print("levels_size is None, error in HMCLocalClassificationModel")
            raise ValueError("levels_size is None")
        if active_levels is None:
            print("active_levels is not valid, error in HMCLocalClassificationModel")
            raise ValueError("active_levels is not valid")

        self.input_size = input_size
        self.levels_size = levels_size
        self.mum_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.levels = nn.ModuleDict()
        self.active_levels = active_levels
        if isinstance(levels_size, int):
            levels_size = {level: levels_size for level in active_levels}
        else:
            self.max_depth = len(levels_size)
        if isinstance(hidden_size, int):
            hidden_size = {level: hidden_size for level in active_levels}
        if isinstance(num_layers, int):
            num_layers = {level: num_layers for level in active_levels}
        if isinstance(dropout, float):
            dropout = {level: dropout for level in active_levels}

        logging.info(
            "HMCLocalModel: input_size=%s, levels_size=%s, "
            "hidden_size=%s, num_layers=%s, dropout=%s, "
            "active_levels=%s",
            input_size,
            levels_size,
            hidden_size,
            num_layers,
            dropout,
            active_levels,
        )
        for index in active_levels:
            self.levels[str(index)] = BuildClassification(
                input_shape=input_size,
                hidden_size=hidden_size[index],
                output_size=levels_size[index],
                nb_layers=num_layers[index],
                dropout_rate=dropout[index],
            )

    def forward(self, x):
        outputs = {}
        for index, level in self.levels.items():
            local_output = level(x)
            outputs[index] = local_output
        return outputs

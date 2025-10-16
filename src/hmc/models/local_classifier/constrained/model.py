import logging
import os
import torch.nn as nn
import torch

from hmc.models.local_classifier.constrained.utils import (
    apply_hierarchical_constraint_vectorized_corrected,
)


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


class ConstrainedHMCLocalModel(nn.Module):
    def __init__(
        self,
        levels_size,
        input_size=None,
        hidden_dims=None,
        num_layers=None,
        dropouts=None,
        active_levels=None,
        results_path=None,
        device="cpu",
        nodes_idx=None,
        local_nodes_reverse_idx=None,
        edges_matrix_dict=None,
        r=None,
    ):
        super(ConstrainedHMCLocalModel, self).__init__()
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
        self.dropouts = dropouts
        self.results_path = results_path
        self.levels = nn.ModuleDict()
        self.active_levels = active_levels
        self.level_active = [True] * len(levels_size)
        self.device = device
        self.r = r
        self.nodes_idx = nodes_idx
        self.local_nodes_reverse_idx = local_nodes_reverse_idx
        self.edges_matrix_dict = edges_matrix_dict

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
            dropouts,
            active_levels,
        )
        for index in active_levels:

            self.levels[str(index)] = BuildClassification(
                input_shape=input_size,
                hidden_dims=hidden_dims[index],
                output_size=levels_size[index],
                dropout=dropouts[index],
            )

    def forward(
        self,
        x: torch.Tensor,
    ):
        outputs = {}
        for level_idx, level in self.levels.items():
            level_idx = int(level_idx)
            if not self.level_active[level_idx]:
                # logging.info(
                #     "Level %d is not active, skipping model creation", level_idx
                # )
                model_path = os.path.join(
                    self.results_path, "best_model_level_" + str(level_idx) + ".pth"
                )

                self.levels[str(level_idx)].load_state_dict(torch.load(model_path))
                # logging.info(
                #     "Loaded trained model from %s for level %d", model_path, level_idx
                # )
            local_output = level(x)
            # 1. Obter a Matriz de Mapeamento (pré-calculada)

            if level_idx != 0 and self.training:
                r_sub_level = self.edges_matrix_dict[level_idx].to(self.device)

                # Pega o output dos ancestrais no batch, já calculado
                probs_ancestors = outputs[level_idx - 1].double()
                local_output = apply_hierarchical_constraint_vectorized_corrected(
                    outputs=local_output,
                    prev_level_outputs=probs_ancestors,
                    R_sub=r_sub_level,
                )
            outputs[level_idx] = local_output
        return outputs

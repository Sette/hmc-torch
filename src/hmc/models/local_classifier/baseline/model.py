import logging
import os
import torch.nn as nn
import torch


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
        dropouts=None,
        active_levels=None,
        results_path=None,
        resitual=False,
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
        if not results_path:
            logging.info("results_path is None, error in HMCLocalClassificationModel")
            raise ValueError("results_path is None")
        self.resitual = resitual
        self.input_size = input_size
        self.levels_size = levels_size
        self.mum_layers = num_layers
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.results_path = results_path
        self.levels = nn.ModuleDict()
        self.output_norms = nn.ModuleDict() 
        self.active_levels = active_levels
        self.level_active = [True] * len(levels_size)
        
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
            if not self.resitual or index == 0:
                self.levels[str(index)] = BuildClassification(
                    input_shape=input_size,
                    hidden_dims=hidden_dims[index],
                    output_size=levels_size[index],
                    dropout=dropouts[index],
                )
            else:
                self.levels[str(index)] = BuildClassification(
                    input_shape=input_size+levels_size[index - 1],
                    hidden_dims=hidden_dims[index],
                    output_size=levels_size[index],
                    dropout=dropouts[index],
                )

    def forward(self, x):
        outputs = {}
        current_input = x
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
            if self.resitual and level_idx > 0:
                previous_output = outputs[level_idx - 1]
                previous_output_norm = previous_output > 0.2

                # print(f"\nLevel {level_idx}:")
                # print(f"x original shape: {x.shape}")
                # print(f"previous_output shape: {previous_output.shape}")
                # print(f"previous_output_norm shape: {previous_output_norm.shape}")
                    
                current_input = torch.cat((x, previous_output_norm), dim=1)
                # print(f"  concatenated shape: {current_input.shape}")
            local_output = level(current_input)
            outputs[level_idx] = local_output
            # print(f"  output shape: {local_output.shape}")
        return outputs

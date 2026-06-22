"""Local classifier: one MLP per hierarchy level.

Each level receives the same input features and predicts only the classes
at that level.  All levels are trained jointly with per-level BCE loss.
"""

import torch
import torch.nn as nn


class LevelMLP(nn.Module):
    """Single-level MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[
            non_lin
        ]

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class LocalModel(nn.Module):
    """One MLP per hierarchy level, trained jointly.

    Args:
        input_dim: Feature dimension (e.g. 768 for SPECTER2).
        levels_size: dict {level_idx: n_classes}.
        hidden_dim: Hidden size for each level's MLP.
        num_layers: Number of hidden layers per MLP.
        dropout: Dropout probability.
        non_lin: Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        levels_size: dict,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        self.levels_size = levels_size
        self.max_depth = len(levels_size)

        self.classifiers = nn.ModuleDict()
        for level, n_classes in sorted(levels_size.items()):
            self.classifiers[str(level)] = LevelMLP(
                input_dim=input_dim,
                output_dim=n_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                non_lin=non_lin,
            )

    def forward(self, x: torch.Tensor) -> dict:
        """Return {level_str: tensor(N, n_classes_level)}."""
        return {lvl: clf(x) for lvl, clf in self.classifiers.items()}

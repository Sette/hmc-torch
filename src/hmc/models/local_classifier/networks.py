from typing import List

import torch
import torch.nn as nn

# ============================================================================
# UTILITY CLASS - Classification Network
# ============================================================================

class ClassificationNetwork(nn.Module):
    """
    Multi-layer neural network for classification.

    Used as building block within hierarchical models.
    """

    def __init__(
            self,
            input_size: int,
            hidden_dims: List[int],
            output_size: int,
            num_layers: int = 2,
            dropout: float = 0.0,
    ):
        """
        Initialize classification network.

        Args:
            input_size: Dimension of input
            hidden_dims: Hidden layer dimensions
            output_size: Number of output classes
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super(ClassificationNetwork, self).__init__()

        layers = []
        current_size = input_size

        # Build hidden layers
        for i in range(num_layers):
            hidden_size = hidden_dims[i] if isinstance(hidden_dims, list) else hidden_dims

            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)



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
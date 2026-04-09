"""
This module contains the classification networks.
"""

import torch
from torch import nn

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
        kwargs: dict,
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
        super().__init__()
        self.level = kwargs["level"]
        input_size = kwargs["input_size"]
        hidden_dims = kwargs["hidden_dims"]
        output_size = kwargs["output_size"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs["dropout"]

        layers = []
        current_size = input_size

        # Build hidden layers
        for i in range(num_layers):
            hidden_size = (
                hidden_dims[i] if isinstance(hidden_dims, list) else hidden_dims
            )

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
    """
    Unified classifier supporting:
    - Pure MLP
    - GCNConv (PyTorch Geometric)
    - GATConv (PyTorch Geometric, multi-head)
    """

    def __init__(
        self,
        kwargs: dict,
    ):
        super().__init__()
        self.level = kwargs["level"]
        input_size = kwargs["input_size"]
        hidden_dims = kwargs["hidden_dims"]
        output_size = kwargs["output_size"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs["dropout"]
        device = kwargs["device"]

        layers = []
        current_size = input_size
        # Build hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_dims[i]))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_size = hidden_dims[i]

        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.head = nn.Sequential(*layers).to(device)

    # ======================================================================
    # Forward
    # ======================================================================

    def forward(self, x):
        """
        x : Feature vector of shape (batch_size, input_size)

        """

        # =====================================================
        # Head Classifier
        # =====================================================
        return self.head(x)

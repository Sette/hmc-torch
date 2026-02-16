from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

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
        level: int = 0,
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
        self.level = level

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
    - Attention MLP
    - GCNConv (PyTorch Geometric)
    - GATConv (PyTorch Geometric, multi-head)
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        level: int = 0,
        device: str = 'cuda',
    ):
        super().__init__()

        self.level = level
        layers = []
        current_size = input_size

        print(input_size)
        print(hidden_dim)

        # Build hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_dim[i]))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_size = hidden_dim[i]

        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.mpl = nn.Sequential(*layers).to(device)


    # ======================================================================
    # Forward
    # ======================================================================
    def forward(self, x):
        """
        x :

        edge_index :
            Only required for GCN or GAT
        """

        # =====================================================
        # Pure MLP
        # =====================================================
        return self.mpl(x)


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
        hidden_dims: List[int],
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        attention: bool = False,
        gcnconv: bool = False,
        gatconv: bool = False,
        num_heads: int = 1,
    ):
        super().__init__()

        self.attention = attention
        self.gcn = gcnconv
        self.gat = gatconv
        self.num_heads = num_heads

        # ============================================
        # 1. BACKBONE MLP (sempre presente)
        # ============================================
        mlp_layers = []
        current_size = input_size

        # Build hidden layers
        for i in range(num_layers):
            hidden_size = (
                hidden_dims[i] if isinstance(hidden_dims, list) else hidden_dims
            )

            mlp_layers.append(nn.Linear(current_size, hidden_size))
            mlp_layers.append(nn.ReLU())

            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))

            current_size = hidden_size

        mlp_layers.append(nn.Linear(current_size, output_size))
        mlp_layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*mlp_layers)

        # ============================================
        # 2. Attention MLP (opcional)
        # ============================================
        if self.attention:
            self.att_q = nn.Linear(input_size, input_size)
            self.att_k = nn.Linear(input_size, input_size)
            self.att_v = nn.Linear(input_size, input_size)
            self.softmax = nn.Softmax(dim=-1)

        # ============================================
        # 3. GCNConv (real, PyG)
        # ============================================
        if self.gcn:
            # 1 camada GCN antes da MLP
            self.gcn_layer = GCNConv(input_size, input_size)

        # ============================================
        # 4. GATConv (real, PyG)
        # ============================================
        if self.gat:
            self.gat_layer = GATConv(
                in_channels=input_size,
                out_channels=input_size // num_heads,
                heads=num_heads,
                concat=True,
            )

    # ======================================================================
    # Forward
    # ======================================================================
    def forward(self, x, edge_index=None):
        """
        x :
            MLP mode   -> [B, F]
            GNN mode   -> [N_nodes, F]

        edge_index :
            Only required for GCN or GAT
        """

        # =====================================================
        # GCNConv
        # =====================================================
        if self.gcn:
            if edge_index is None:
                raise ValueError("edge_index must be provided for GCNConv")
            x = self.gcn_layer(x, edge_index)
            x = F.relu(x)
            return self.mlp(x)

        # =====================================================
        # GATConv
        # =====================================================
        if self.gat:
            if edge_index is None:
                raise ValueError("edge_index must be provided for GATConv")
            x = self.gat_layer(x, edge_index)
            x = F.relu(x)
            return self.mlp(x)

        # =====================================================
        # Attention MLP
        # =====================================================
        if self.attention:
            Q = self.att_q(x)
            K = self.att_k(x)
            V = self.att_v(x)

            att = self.softmax(Q @ K.T)
            x = att @ V
            return self.mlp(x)

        # =====================================================
        # Pure MLP
        # =====================================================
        return self.mlp(x)

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
        attention: bool = False,
        gcn: bool = False,
        gat: bool = False,
        num_heads: int = 1,
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
        self.attention = attention
        self.gcn = gcn
        self.gat = gat

        layers = []
        current_size = input_size
        
        # ============================================
        # 3. GCNConv (real, PyG)
        # ============================================
        if self.gcn:
            # 1 camada GCN antes da MLP
            layers.append(GCNConv(current_size, current_size))
            
        if self.gat:
            layers.append(GATConv(
                in_channels=current_size,
                out_channels=current_size // num_heads,
                heads=num_heads,
                concat=True,
            ))

        if self.attention:
            layers.append(LocalLevelAttentionBlock(input_dim=current_size, attn_dim=512, num_heads=4))

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



class LocalLevelAttentionBlock(nn.Module):
    def __init__(self, input_dim, attn_dim=256, num_heads=4):
        """
        Local Attention Block for a single hierarchy level.
        - input_dim: dimension of incoming embedding (ex: 529)
        - attn_dim: internal projected dimension (must be divisible by num_heads)
        - num_heads: number of attention heads
        """
        super().__init__()

        # Ensure attn_dim is valid
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})")

        # Project input_dim → attn_dim
        self.proj_in = nn.Linear(input_dim, attn_dim)

        # Local multi-head attention (per-level)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Normalize after attention
        self.norm1 = nn.LayerNorm(attn_dim)

        # Feed-forward block (transformer-like)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.ReLU(),
            nn.Linear(attn_dim * 2, attn_dim)
        )

        self.norm2 = nn.LayerNorm(attn_dim)

        # Project back to original dim → keeps compatibilidade
        self.proj_out = nn.Linear(attn_dim, input_dim)

    def forward(self, x):
        """
        x shape expected: (batch, features)  → convert to sequence length = 1
        """
        # Reshape: (B, F) → (B, 1, F)
        x = x.unsqueeze(1)

        # Project to attention dimension
        h = self.proj_in(x)

        # Apply local self-attention
        attn_output, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_output)

        # FFN
        ffn_output = self.ffn(h)
        h = self.norm2(h + ffn_output)

        # Project back to original dim
        h = self.proj_out(h)

        # Remove sequence dimension: (B, 1, F) → (B, F)
        return h.squeeze(1)
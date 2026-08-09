"""Label-graph encoders: GCN and GAT over class-node embeddings.

Propagates information along the hierarchy graph to produce
context-aware label representations for downstream scoring.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LabelGCN(nn.Module):
    """Graph Convolutional Network over class-label hierarchy.

    Args:
        n_nodes: Number of class nodes.
        embed_dim: Dimensionality of initial label embeddings.
        hidden_dim: Hidden dimension for GCN layers.
        num_layers: Number of GCN layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.num_layers = num_layers

        # Learnable label embeddings
        self.label_embed = nn.Parameter(torch.empty(n_nodes, embed_dim))
        nn.init.xavier_uniform_(self.label_embed)

        # GCN layers
        self.convs = nn.ModuleList()
        in_dim = embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            self.convs.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute context-aware label embeddings.

        Args:
            edge_index: ``(2, n_edges)`` undirected edge index tensor.
                Both ``(i→j)`` and ``(j→i)`` must be present.

        Returns:
            ``(n_nodes, embed_dim)`` label representations.
        """
        x = self.label_embed
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class LabelGAT(nn.Module):
    """Graph Attention Network over class-label hierarchy.

    Args:
        n_nodes: Number of class nodes.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GAT layers.
        heads: Number of attention heads (only for intermediate layers).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.num_layers = num_layers

        self.label_embed = nn.Parameter(torch.empty(n_nodes, embed_dim))
        nn.init.xavier_uniform_(self.label_embed)

        self.convs = nn.ModuleList()
        in_dim = embed_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = embed_dim if is_last else hidden_dim
            layer_heads = 1 if is_last else heads
            self.convs.append(
                GATConv(
                    in_dim, out_dim // layer_heads, heads=layer_heads, dropout=dropout
                )
            )
            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted label embeddings.

        Args:
            edge_index: ``(2, n_edges)`` undirected edge index tensor.

        Returns:
            ``(n_nodes, embed_dim)`` label representations.
        """
        x = self.label_embed
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Simplified GCN / GAT layers (no external dependency)
# ---------------------------------------------------------------------------


class GCNConv(nn.Module):
    """Simple GCN convolution layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN forward pass.

        Args:
            x: ``(n_nodes, in_dim)`` node features.
            edge_index: ``(2, n_edges)``.

        Returns:
            ``(n_nodes, out_dim)``.
        """
        row, col = edge_index[0], edge_index[1]
        n_nodes = x.shape[0]

        # Normalised adjacency aggregation
        deg = (
            torch.zeros(n_nodes, device=x.device)
            .scatter_add(0, row, torch.ones_like(row, dtype=torch.float))
            .clamp(min=1)
        )
        deg_inv_sqrt = deg.pow(-0.5)

        # Message passing
        out = x.new_zeros(n_nodes, self.weight.shape[1])
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        messages = x[col] @ self.weight
        out = out.scatter_add(
            0, row.unsqueeze(-1).expand_as(messages), messages * norm.unsqueeze(-1)
        )
        return out + self.bias


class GATConv(nn.Module):
    """Simple GAT convolution layer (single-head or multi-head)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout

        self.weight = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src.view(heads, 1, out_dim))
        nn.init.xavier_uniform_(self.att_dst.view(heads, 1, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GAT forward pass.

        Args:
            x: ``(n_nodes, in_dim)``.
            edge_index: ``(2, n_edges)``.

        Returns:
            ``(n_nodes, heads * out_dim)``.
        """
        row, col = edge_index[0], edge_index[1]
        n_nodes = x.shape[0]

        outputs = []
        for h in range(self.heads):
            # Linear transform
            x_src = x[row] @ self.weight[h]  # (E, out_dim)
            x_dst = x[col] @ self.weight[h]  # (E, out_dim)

            # Attention coefficients
            alpha_src = (x_src * self.att_src[h]).sum(-1)  # (E,)
            alpha_dst = (x_dst * self.att_dst[h]).sum(-1)  # (E,)
            alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)

            # Softmax over neighbours
            alpha_max = torch.zeros(n_nodes, device=x.device).scatter_reduce(
                0, row, alpha, reduce="amax", include_self=False
            )
            alpha = alpha - alpha_max[row]
            alpha = torch.exp(alpha)

            # Normalise
            alpha_sum = (
                torch.zeros(n_nodes, device=x.device)
                .scatter_add(0, row, alpha)
                .clamp(min=1)
            )
            alpha = alpha / alpha_sum[row]

            # Aggregate
            out = x.new_zeros(n_nodes, self.out_dim)
            out = out.scatter_add(
                0, row.unsqueeze(-1).expand_as(x_src), x_src * alpha.unsqueeze(-1)
            )
            outputs.append(out)

        return torch.cat(outputs, dim=-1)

"""Residual MLP for tabular HMC.

A shared fully-connected encoder with residual blocks followed by a
:class:`GlobalSigmoidHead`.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """MLP block with residual connection: ``out = x + Dropout(Act(Linear(x)))``."""

    def __init__(self, dim: int, dropout: float = 0.3, activation: str = "relu"):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.norm(x + self.net(x))


class ResidualMLPEncoder(nn.Module):
    """Shared residual encoder for tabular features.

    Architecture::

        Linear(in, hidden) → [ResidualBlock(hidden) × n_blocks] → output

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Hidden dimension for all residual blocks.
    n_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout probability.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout, activation) for _ in range(n_blocks)]
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features through residual blocks."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class TabularMLPModel(nn.Module):
    """Residual MLP + GlobalSigmoidHead for tabular HMC.

    Parameters
    ----------
    input_dim : int
        Number of input features (after preprocessing).
    n_nodes : int
        Total number of class nodes.
    hidden_dim : int
        Hidden dimension.
    n_blocks : int
        Number of residual blocks.
    head_layers : int
        Number of layers in the classification head (0 = linear probe).
    dropout : float
        Dropout probability.
    activation : str
        Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        n_nodes: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        head_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        self.encoder = ResidualMLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            activation=activation,
        )
        from hmc.models.hierarchical.heads import (  # pylint: disable=import-outside-toplevel
            GlobalSigmoidHead,
        )

        self.head = GlobalSigmoidHead(
            input_dim=hidden_dim,
            n_nodes=n_nodes,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            non_lin=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_nodes)`` probability tensor."""
        emb = self.encoder(x)
        return self.head(emb)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder embeddings without the classification head."""
        return self.encoder(x)

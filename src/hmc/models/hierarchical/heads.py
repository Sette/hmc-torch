"""Reusable HMC classification heads.

Each head accepts encoded features and outputs ``node_scores`` of shape
``(batch, n_nodes)``.
"""

from __future__ import annotations


import torch
from torch import nn

# ---------------------------------------------------------------------------
# Global sigmoid head
# ---------------------------------------------------------------------------


class GlobalSigmoidHead(nn.Module):
    """Single MLP + sigmoid over all class nodes (global multi-label).

    Architecture::

        encoder → hidden layers → Linear(n_nodes) → Sigmoid

    Args:
        input_dim: Encoded feature dimension.
        n_nodes: Total number of class nodes.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers (0 = linear probe).
        dropout: Dropout probability.
        non_lin: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        n_nodes: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        activation = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }[non_lin]

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_nodes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_nodes)`` probabilities."""
        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# Local-level head
# ---------------------------------------------------------------------------


class LocalLevelHead(nn.Module):
    """One MLP per hierarchy level, sharing the same encoder output.

    Architecture::

        encoder → ┬─ MLP_level0 → sigmoid → (batch, n_classes_0)
                  ├─ MLP_level1 → sigmoid → (batch, n_classes_1)
                  └─ ...

    Args:
        input_dim: Encoded feature dimension.
        level_sizes: ``{level_idx: n_classes}``.
        hidden_dim: Hidden layer size per MLP.
        num_layers: Number of hidden layers per level MLP.
        dropout: Dropout probability.
        non_lin: Activation.
    """

    def __init__(
        self,
        input_dim: int,
        level_sizes: dict[int, int],
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        self.level_sizes = level_sizes

        activation = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }[non_lin]

        self.heads = nn.ModuleDict()
        for lvl, n_classes in sorted(level_sizes.items()):
            layers = []
            in_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_classes))
            self.heads[str(lvl)] = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{level_str: (batch, n_classes_level)}``."""
        return {lvl: torch.sigmoid(head(x)) for lvl, head in self.heads.items()}

    def to_global(
        self,
        level_preds: dict[str, torch.Tensor],
        local_nodes_idx: dict[int, dict[str, int]],
        nodes_idx: dict[str, int],
        n_nodes: int,
    ) -> torch.Tensor:
        """Convert per-level predictions to a global ``(batch, n_nodes)`` tensor.

        Args:
            level_preds: ``{lvl_str: (batch, n_classes)}`` predictions.
            local_nodes_idx: ``{lvl: {node_name: local_index}}``.
            nodes_idx: ``{node_name: global_index}``.
            n_nodes: Total number of global nodes.

        Returns:
            ``(batch, n_nodes)`` tensor.
        """
        device = next(iter(level_preds.values())).device
        batch_size = next(iter(level_preds.values())).shape[0]
        global_pred = torch.zeros(batch_size, n_nodes, device=device)

        for lvl_str, preds in level_preds.items():
            lvl = int(lvl_str)
            if lvl not in local_nodes_idx:
                continue
            local_idx = local_nodes_idx[lvl]
            for name, lidx in local_idx.items():
                if name in nodes_idx:
                    global_pred[:, nodes_idx[name]] = preds[:, lidx]

        return global_pred


# ---------------------------------------------------------------------------
# Tree-path head (single-path classification)
# ---------------------------------------------------------------------------


class TreePathHead(nn.Module):
    """Single-path classification head for strictly tree-structured tasks.

    Each sample follows exactly one path from root to leaf.  The head
    applies a masked softmax at each level, where the mask restricts
    choices to children of the chosen parent node.

    Args:
        input_dim: Encoded feature dimension.
        hierarchy: :class:`TreeHierarchy` instance.
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        input_dim: int,
        level_sizes: dict[int, int],
        hierarchy=None,  # Optional TreeHierarchy for mask construction
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.level_sizes = level_sizes
        self.hierarchy = hierarchy

        # Per-level classifiers
        self.heads = nn.ModuleDict()
        for lvl, n_classes in sorted(level_sizes.items()):
            layers = []
            in_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_classes))
            self.heads[str(lvl)] = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        prev_choices: dict[int, torch.Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> dict[str, torch.Tensor]:
        """Return per-level logits.

        Args:
            x: ``(batch, input_dim)`` features.
            prev_choices: Optional ``{lvl: (batch, 1)}`` one-hot indices
                of parent choices for masked softmax.

        Returns:
            ``{lvl_str: (batch, n_classes)}`` logits (before softmax).
        """
        outputs = {}
        for lvl_str, head in self.heads.items():
            logits = head(x)
            outputs[lvl_str] = logits
        return outputs

    def decode_path(self, x: torch.Tensor) -> list[list[str]]:
        """Decode the most probable path for each sample.

        Returns a list of per-sample path lists (node names from root to leaf).
        """
        with torch.no_grad():
            level_preds = self.forward(x)
            paths = [["root"] for _ in range(x.shape[0])]
            for lvl_str, logits in sorted(
                level_preds.items(), key=lambda kv: int(kv[0])
            ):
                probs = torch.softmax(logits, dim=-1)
                best_idx = probs.argmax(dim=-1)  # (batch,)
                lvl = int(lvl_str)
                if self.hierarchy is not None and lvl in self.hierarchy.levels:
                    level_nodes = self.hierarchy.levels[lvl]
                    for b in range(x.shape[0]):
                        idx = int(best_idx[b].item())
                        if idx < len(level_nodes):
                            paths[b].append(level_nodes[idx])
            return paths

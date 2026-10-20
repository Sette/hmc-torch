"""The networks (C-HMCNN's and Surj's) and C-HMCNN's hierarchy constraint.

``ChmcnnNet``: C-HMCNN's ``ConstrainedFFNNModel`` without the constraint inside (it is applied
by ``Constraint`` instead, so every net can run with or without it).  Layers are created in the
same order, so the same seed gives the same initial weights.

``SurjNet`` (Sec. 3.3): the same three fully connected layers, the last one mapping to the
embedding dimension D; cosine similarity between that output and each of the V frozen node
embeddings (Eq. 4); another fully connected layer ``V -> V``; sigmoid.  With ``z=None`` the
labels are one-hot (``Z = I``, ``D = V``): the "w/o ontology learning" baseline of Table 5.

``Constraint`` is C-HMCNN's max-constraint module (MCM): every node gets the maximum score of
its descendants.  On FunCat and ``_others`` it is the dense ``get_constr_out`` verbatim; on GO
(4k labels) it is ``BlockDiagonalR``, which computes the same values in O(B·E) memory.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch import nn

from compare_chmcnn_loader import get_constr_out, their_r  # pylint: disable=wrong-import-order
from hmc.models.hierarchical.sparse_r import BlockDiagonalR


class ChmcnnNet(nn.Module):
    """input -> h -> h -> V, ReLU and dropout between layers, sigmoid."""

    def __init__(self, input_dim: int, hidden_dim: int, n_labels: int, dropout: float = 0.7):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Linear(hidden_dim, n_labels)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc[:-1]:
            x = self.drop(torch.relu(layer(x)))
        return torch.sigmoid(self.fc[-1](x))


class SurjNet(nn.Module):
    """FC layers -> cosine with frozen label embeddings -> FC(V, V) -> sigmoid."""

    def __init__(self, input_dim: int, hidden_dim: int, n_labels: int,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 z: torch.Tensor | None = None, dropout: float = 0.7):
        super().__init__()
        z = torch.eye(n_labels) if z is None else torch.as_tensor(z, dtype=torch.float32)
        if z.shape[0] != n_labels:
            raise ValueError(f"{z.shape[0]} embeddings for {n_labels} labels")
        self.register_buffer("z", nn.functional.normalize(z, dim=1))
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Linear(hidden_dim, z.shape[1])])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_labels, n_labels)

    def similarities(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc[:-1]:
            x = self.drop(torch.relu(layer(x)))
        return nn.functional.normalize(self.fc[-1](x), dim=1) @ self.z.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.similarities(x)))


class Constraint:
    """C-HMCNN's MCM (``__call__``) and MCLoss input (``mcloss_output``) for adjacency ``a``."""

    def __init__(self, a: np.ndarray, device, sparse: bool):
        self.sparse = sparse
        if sparse:
            child, parent = np.nonzero(a)
            g = nx.DiGraph()
            g.add_nodes_from(range(len(a)))
            g.add_edges_from(zip(child.tolist(), parent.tolist()))
            self.r = BlockDiagonalR(g, column_index={i: i for i in range(len(a))})
        else:
            self.r = their_r(a).unsqueeze(0).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            return self.r.reconcile(x.double())
        return get_constr_out(x, self.r)

    def mcloss_output(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """C-HMCNN/main.py's training output: MCM of the positives' own scores, MCM elsewhere."""
        constr = self(output)
        pos = self(labels * output.double())
        return (1 - labels) * constr + labels * pos

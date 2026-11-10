"""Surj (Yang & Howe, WWW'22): classify into a learned embedding of the ontology.

Surj separates hierarchical multi-label classification into two stages:

1. **Ontology learning** (Section 3.2, Eq. 1--2): a graph autoencoder learns
   one vector per label from the hierarchy alone
   (:func:`~hmc.models.hierarchical.graph_autoencoder.pretrain_label_gae`).
2. **Classification** (Section 3.3, Eq. 4--5): fully connected layers map a
   sample into the same ``D``-dimensional space, a cosine-similarity layer
   compares it with every (frozen) label vector, and a further fully connected
   layer followed by a sigmoid turns the ``V`` similarities into ``V`` label
   probabilities, trained with binary cross-entropy.

:class:`SurjClassifier` is stage 2.  With ``label_embeddings=None`` the label
vectors are the identity (one-hot, ``D = V``): the same network without
ontology learning, which is the ablation of the paper's Table 5.
"""

from __future__ import annotations

import torch
from torch import nn


class SurjClassifier(nn.Module):
    """MLP into the label-embedding space + cosine layer + FC + sigmoid.

    Args:
        input_dim: feature dimension.
        output_dim: number of labels ``V``.
        label_embeddings: ``(V, D)`` frozen label vectors; ``None`` for one-hot.
        hidden_dim: width of the two hidden layers.
        dropout: dropout after each hidden layer.
        cosine_fc: apply the ``V -> V`` fully connected layer after the cosine
            similarities (the paper's architecture); ``False`` replaces it by a
            learned scale and a per-label bias.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dim: int,
        output_dim: int,
        label_embeddings: torch.Tensor | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        cosine_fc: bool = True,
    ):
        super().__init__()
        if label_embeddings is None:
            label_embeddings = torch.eye(output_dim)
        if label_embeddings.shape[0] != output_dim:
            raise ValueError(
                f"label_embeddings has {label_embeddings.shape[0]} rows, expected {output_dim}"
            )
        self.register_buffer(
            "label_embeddings", nn.functional.normalize(label_embeddings.float(), dim=1)
        )
        dim = label_embeddings.shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.cosine_fc = cosine_fc
        if cosine_fc:
            self.fc = nn.Linear(output_dim, output_dim)
        else:
            # Cosines live in [-1, 1]; a learned scale (init 10) lets the
            # logits reach the range BCE needs.
            self.log_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of each sample with each label, ``(batch, V)``."""
        h = nn.functional.normalize(self.encoder(x), dim=1)
        return h @ self.label_embeddings.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Label probabilities, ``(batch, V)``."""
        cos = self.similarities(x)
        if self.cosine_fc:
            return torch.sigmoid(self.fc(cos))
        return torch.sigmoid(self.log_scale.exp() * cos + self.bias)

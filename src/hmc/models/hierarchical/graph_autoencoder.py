"""Graph autoencoder (Kipf & Welling, 2016) for label-hierarchy embeddings.

Surj (Yang & Howe, WWW'22) learns a representation of every label offline
with a graph autoencoder over the ontology and then scores samples against it
by cosine similarity.  :func:`pretrain_label_gae` reproduces that first step:
a two-layer GCN encoder over the symmetric normalized hierarchy and an
inner-product decoder, trained either against sampled non-edges, to
reconstruct the whole adjacency matrix (``loss="full"``), or with the
Sibling-Repulsive Graph Autoencoder objective (``loss="sibling_repulsive"``)
that prevents sibling nodes sharing a parent from collapsing onto a single
ray on the unit hypersphere.  The embeddings feed
:class:`~hmc.models.hierarchical.surj.SurjClassifier` and the ``"gae"`` modes
of :class:`~hmc.models.hierarchical.label_embedding.LabelEmbeddingHead`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn

from hmc.models.hierarchical.label_embedding import normalized_adjacency

LOSSES = ("sampled", "full", "sibling_repulsive")


def sibling_pairs(edges: np.ndarray) -> np.ndarray:
    """Return all unique ``(u, v)`` pairs (``u < v``) of nodes sharing a parent.

    Args:
        edges: ``(E, 2)`` array of ``(child, parent)`` column indices.

    Returns:
        np.ndarray: ``(S, 2)`` int64 array of sibling index pairs.
    """
    children_of: dict[int, list[int]] = defaultdict(list)
    for child, parent in np.asarray(edges, dtype=np.int64):
        children_of[int(parent)].append(int(child))
    pairs: set[tuple[int, int]] = set()
    for children in children_of.values():
        unique = sorted(set(children))
        for i, u in enumerate(unique):
            for v in unique[i + 1 :]:
                pairs.add((u, v))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


class _GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Eq. 2 of Surj: ``A ReLU(A X W0) W1``."""
        h = torch.relu(adjacency @ self.lin1(x))
        return adjacency @ self.lin2(h)


class _ResGCNEncoder(nn.Module):
    """Residual GCN encoder that preserves ego-node identity alongside neighbourhood smoothing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, skip_scale: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.skip1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.skip2 = nn.Linear(hidden_dim, out_dim)
        self.skip_scale = skip_scale

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(adjacency @ self.lin1(x) + self.skip_scale * self.skip1(x))
        return adjacency @ self.lin2(h) + self.skip_scale * self.skip2(h)


def pretrain_label_gae(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    edges: np.ndarray,
    n: int,
    dim: int = 128,
    hidden_dim: int = 256,
    epochs: int = 200,
    lr: float = 1e-2,
    depth: np.ndarray | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    loss: str = "sampled",
    features: np.ndarray | None = None,
    sib_weight: float = 5.0,
    sib_margin: float = 0.0,
) -> np.ndarray:
    """Learn ``(n, dim)`` label embeddings by reconstructing the hierarchy edges.

    Args:
        edges: ``(E, 2)`` array of (child, parent) column indices.
        n: number of labels.
        dim: embedding size.
        hidden_dim: width of the first GCN layer.
        epochs: full-batch training steps.
        lr: Adam learning rate.
        depth: optional per-label depth, appended to the node features.
        seed: seed for initialization and negative sampling.
        device: where to train.
        loss: ``"sampled"`` -- BCE on the edges against as many uniformly
            sampled pairs; ``"full"`` -- Kipf & Welling's GAE loss, BCE over
            every entry of ``A + I`` with positives weighted by
            ``(n^2 - P) / P``; ``"sibling_repulsive"`` -- SR-GAE, combining
            residual GCN encoding and full adjacency reconstruction with a
            squared hinge repulsion penalty on sibling cosine similarities.
        features: optional ``(n, K)`` node features (e.g. text embeddings or
            training co-occurrence profiles), appended to the identity.
        sib_weight: weight of the sibling repulsion penalty when
            ``loss="sibling_repulsive"``.
        sib_margin: target maximum cosine similarity between siblings when
            ``loss="sibling_repulsive"``.

    Returns:
        np.ndarray: float32 embeddings, one row per label column.
    """
    if loss not in LOSSES:
        raise ValueError(f"Unknown GAE loss {loss!r}; choose from {LOSSES}")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    adjacency = normalized_adjacency(edges, n).to(device)
    feats = [torch.eye(n)]
    if features is not None:
        f = torch.as_tensor(features, dtype=torch.float32)
        feats.append(f / f.norm(dim=1, keepdim=True).clamp_min(1e-8))
    if depth is not None:
        d = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(1)
        feats.append(d / max(float(d.max()), 1.0))
    x = torch.cat(feats, dim=1).to(device)

    pos = torch.as_tensor(
        np.concatenate([edges, edges[:, ::-1]]), dtype=torch.long, device=device
    )
    encoder_cls = _ResGCNEncoder if loss == "sibling_repulsive" else _GCNEncoder
    model = encoder_cls(x.shape[1], hidden_dim, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.functional.binary_cross_entropy_with_logits
    if loss in ("full", "sibling_repulsive"):
        target = torch.eye(n, device=device)
        target[pos[:, 0], pos[:, 1]] = 1.0
        n_pos = float(target.sum())
        pos_weight = torch.tensor((n * n - n_pos) / n_pos, device=device)
        norm = n * n / (2 * (n * n - n_pos))
    sib = (
        torch.as_tensor(sibling_pairs(edges), dtype=torch.long, device=device)
        if loss == "sibling_repulsive"
        else None
    )
    for _ in range(epochs):
        opt.zero_grad()
        z = model(x, adjacency)
        if loss in ("full", "sibling_repulsive"):
            step_loss = norm * bce(z @ z.T, target, pos_weight=pos_weight)
            if sib is not None and len(sib) > 0:
                zn = nn.functional.normalize(z, dim=1)
                sib_cos = (zn[sib[:, 0]] * zn[sib[:, 1]]).sum(dim=1)
                step_loss = (
                    step_loss
                    + sib_weight * torch.relu(sib_cos - sib_margin).pow(2).mean()
                )
        else:
            neg = torch.randint(0, n, pos.shape, generator=gen).to(device)
            pos_logit = (z[pos[:, 0]] * z[pos[:, 1]]).sum(1)
            neg_logit = (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)
            step_loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(
                neg_logit, torch.zeros_like(neg_logit)
            )
        step_loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(x, adjacency).cpu().numpy().astype(np.float32)

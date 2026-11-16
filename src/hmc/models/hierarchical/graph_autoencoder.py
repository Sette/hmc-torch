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

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, skip_scale: float = 0.5
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.skip1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.skip2 = nn.Linear(hidden_dim, out_dim)
        self.skip_scale = skip_scale

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Smooth neighbourhoods while preserving the ego-node signal via skips."""
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


class HierarchicalMultiLabelContrastiveLoss(nn.Module):
    """Hierarchical Multi-Label Contrastive Loss (H-MLCL).

    Combines two complementary contrastive objectives in the unit hypersphere
    induced by SR-GAE ontology embeddings:
      1. Sibling-Repulsive Prototype Contrastive Loss: aligns normalized sample
         projections with active class prototypes while up-weighting inactive
         sibling prototypes by ``sib_gamma > 1``.
      2. R-Jaccard Instance Multi-Label Contrastive Loss with a FIFO memory
         queue: pulls sample representations together proportionally to the
         Jaccard overlap of their ancestor-closed label sets.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        z: np.ndarray | torch.Tensor,
        edges: np.ndarray,
        proto_temp: float = 0.2,
        inst_temp: float = 0.2,
        sib_gamma: float = 2.0,
        proto_weight: float = 1.0,
        inst_weight: float = 1.0,
        queue_size: int = 128,
    ):
        super().__init__()
        z_t = torch.as_tensor(z, dtype=torch.float32)
        n_classes, dim = z_t.shape
        self.register_buffer("z_norm", nn.functional.normalize(z_t, dim=1))
        sib = sibling_pairs(edges)
        sib_t = torch.as_tensor(sib, dtype=torch.long)
        self.register_buffer("sib_pairs", sib_t)
        sib_adj = torch.zeros(n_classes, n_classes, dtype=torch.float32)
        if len(sib) > 0:
            sib_adj[sib_t[:, 0], sib_t[:, 1]] = 1.0
            sib_adj[sib_t[:, 1], sib_t[:, 0]] = 1.0
        self.register_buffer("sib_adj", sib_adj)
        self.proto_temp = float(proto_temp)
        self.inst_temp = float(inst_temp)
        self.sib_gamma = float(sib_gamma)
        self.proto_weight = float(proto_weight)
        self.inst_weight = float(inst_weight)
        self.queue_size = int(queue_size)

        self.register_buffer("queue_u", torch.zeros(self.queue_size, dim))
        self.register_buffer("queue_y", torch.zeros(self.queue_size, n_classes))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_len", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, un: torch.Tensor, y: torch.Tensor) -> None:
        if self.queue_size <= 0:
            return
        bsz = un.shape[0]
        if bsz >= self.queue_size:
            self.queue_u.copy_(un[-self.queue_size :])
            self.queue_y.copy_(y[-self.queue_size :])
            self.queue_ptr[0] = 0
            self.queue_len[0] = self.queue_size
            return
        ptr = int(self.queue_ptr[0].item())
        end = ptr + bsz
        if end <= self.queue_size:
            self.queue_u[ptr:end].copy_(un)
            self.queue_y[ptr:end].copy_(y)
        else:
            first = self.queue_size - ptr
            self.queue_u[ptr:].copy_(un[:first])
            self.queue_y[ptr:].copy_(y[:first])
            rem = bsz - first
            self.queue_u[:rem].copy_(un[first:])
            self.queue_y[:rem].copy_(y[first:])
        self.queue_ptr[0] = (ptr + bsz) % self.queue_size
        self.queue_len[0] = min(self.queue_size, int(self.queue_len[0].item()) + bsz)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute H-MLCL loss given sample projections ``u`` and binary labels ``y``."""
        un = nn.functional.normalize(u, dim=1)
        yb = y.float()
        total = u.new_zeros(())

        if self.proto_weight > 0.0:
            sim = (un @ self.z_norm.T) / self.proto_temp
            sim = sim - sim.max(dim=1, keepdim=True).values.detach()
            exp_sim = torch.exp(sim)
            neg_exp = exp_sim * (1.0 - yb)
            neg_sum = neg_exp.sum(dim=1, keepdim=True)
            if self.sib_pairs.numel() > 0 and self.sib_gamma > 1.0:
                sib_boost = (self.sib_gamma - 1.0) * (neg_exp @ self.sib_adj)
            else:
                sib_boost = 0.0
            denom = exp_sim + neg_sum + sib_boost
            log_prob = sim - torch.log(denom.clamp_min(1e-12))
            pos_cnt = yb.sum(dim=1)
            valid_p = pos_cnt > 0
            if valid_p.any():
                proto_loss = -(
                    (yb[valid_p] * log_prob[valid_p]).sum(dim=1) / pos_cnt[valid_p]
                ).mean()
                total = total + self.proto_weight * proto_loss

        if self.inst_weight > 0.0:
            q_len = int(self.queue_len[0].item())
            if q_len > 0:
                all_u = torch.cat([un, self.queue_u[:q_len].detach()], dim=0)
                all_y = torch.cat([yb, self.queue_y[:q_len].detach()], dim=0)
            else:
                all_u = un
                all_y = yb
            bsz = un.shape[0]
            if all_u.shape[0] > 1:
                inter = yb @ all_y.T
                union = (
                    yb.sum(dim=1, keepdim=True) + all_y.sum(dim=1).unsqueeze(0) - inter
                )
                jaccard = inter / union.clamp_min(1.0)
                self_mask = torch.zeros_like(jaccard, dtype=torch.bool)
                idx = torch.arange(bsz, device=u.device)
                self_mask[idx, idx] = True
                jaccard = jaccard.masked_fill(self_mask, 0.0)

                logits = (un @ all_u.T) / self.inst_temp
                logits = (
                    logits
                    - logits.masked_fill(self_mask, -1e9)
                    .max(dim=1, keepdim=True)
                    .values.detach()
                )
                exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
                log_prob_i = logits - torch.log(
                    exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
                )
                w_sum = jaccard.sum(dim=1)
                valid_i = w_sum > 1e-6
                if valid_i.any():
                    inst_loss = -(
                        (jaccard[valid_i] * log_prob_i[valid_i]).sum(dim=1)
                        / w_sum[valid_i]
                    ).mean()
                    total = total + self.inst_weight * inst_loss

        if self.training:
            self._enqueue(un.detach(), yb.detach())
        return total


@torch.no_grad()
def contrastive_knn_predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    u_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    u_eval: torch.Tensor | np.ndarray,
    k: int = 25,
    temp: float = 0.1,
    chunk_size: int = 512,
) -> np.ndarray:
    """Non-parametric multi-label k-NN score in the contrastive embedding space.

    Because every training label vector ``y_train[m]`` satisfies the R-matrix
    ancestor-closure constraint, any convex softmax combination over the top-k
    neighbors automatically satisfies zero hierarchy violations.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ut = nn.functional.normalize(
        torch.as_tensor(u_train, dtype=torch.float32, device=dev), dim=1
    )
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    ue = nn.functional.normalize(
        torch.as_tensor(u_eval, dtype=torch.float32, device=dev), dim=1
    )
    k_eff = min(int(k), ut.shape[0])
    out = []
    for i in range(0, ue.shape[0], chunk_size):
        sim = ue[i : i + chunk_size] @ ut.T
        vals, idx = torch.topk(sim, k=k_eff, dim=1)
        weights = torch.softmax(vals / float(temp), dim=1)
        preds = torch.einsum("nk,nkc->nc", weights, yt[idx])
        out.append(preds.cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32)

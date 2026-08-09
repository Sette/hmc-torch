"""Loss functions for hierarchical multi-label classification.

Each loss accepts ``(logits_or_probs, targets)`` plus optional keyword
arguments for weights, masks, and hyperparameters.
"""

from __future__ import annotations


import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Weighted BCE
# ---------------------------------------------------------------------------


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with per-class weights.

    Weights can be set from label prevalence to up-weight rare terms.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE.

        Args:
            inputs: ``(batch, n_nodes)`` predicted probabilities.
            targets: ``(batch, n_nodes)`` binary ground-truth.

        Returns:
            Scalar loss.
        """
        return F.binary_cross_entropy(
            inputs, targets, weight=self.pos_weight, reduction="mean"
        )

    @staticmethod
    def compute_pos_weight(
        labels: torch.Tensor, clip_min: float = 0.01
    ) -> torch.Tensor:
        """Compute positive-class weight from label statistics.

        ``pos_weight[j] = #negatives_j / #positives_j``

        Args:
            labels: ``(n_samples, n_nodes)`` binary label matrix.
            clip_min: Minimum weight to avoid division by zero.

        Returns:
            Tensor of shape ``(n_nodes,)``.
        """
        n_pos = labels.sum(dim=0).clamp(min=1)
        n_neg = labels.shape[0] - n_pos
        return (n_neg / n_pos).clamp(min=clip_min)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for multi-label classification.

    Down-weights easy examples so the model focuses on hard ones.
    ``FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)``
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: ``(batch, n_nodes)`` predicted probabilities.
            targets: ``(batch, n_nodes)`` binary ground-truth.

        Returns:
            Scalar loss.
        """
        eps = 1e-7
        inputs = inputs.clamp(eps, 1.0 - eps)

        # p_t: probability of the target class
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = -alpha_t * ((1 - p_t) ** self.gamma) * torch.log(p_t)

        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets) * 1.0
            loss = loss * weight

        return loss.mean()


# ---------------------------------------------------------------------------
# Hierarchical consistency loss
# ---------------------------------------------------------------------------


class HierarchicalConsistencyLoss(nn.Module):
    """Penalise hierarchy violations: parent score < child score.

    For a tree hierarchy, this adds a penalty whenever a parent's predicted
    probability is lower than any of its children's probabilities.
    """

    def __init__(self, r_matrix: torch.Tensor, margin: float = 0.0):
        """
        Args:
            r_matrix: Ancestor matrix ``(1, n_nodes, n_nodes)`` where
                ``r_matrix[0, i, j] = 1`` if *i* is ancestor of *j*.
            margin: Minimum gap enforced between parent and child score.
        """
        super().__init__()
        self.r_matrix = r_matrix
        self.margin = margin

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss.

        Args:
            probs: ``(batch, n_nodes)`` predicted probabilities.

        Returns:
            Scalar penalty.
        """
        device = probs.device
        r = self.r_matrix.to(device)  # (1, N, N): r[0, child_idx, anc_idx] = 1 if
        # anc is ancestor of child

        # diff[b, child, anc] = probs[child] - probs[ancestor]
        p_child = probs.unsqueeze(2)  # (B, N, 1)
        p_anc = probs.unsqueeze(1)  # (B, 1, N)
        diff = p_child - p_anc  # (B, N, N): diff[b, child, anc]

        # r has dims (1, child, ancestor), matching diff's (B, child, ancestor).
        # Penalty when prob[child] > prob[ancestor]:
        violations = r * torch.clamp(diff + self.margin, min=0.0)

        n_pairs = r.sum().clamp(min=1)
        return violations.sum() / (probs.shape[0] * n_pairs)


# ---------------------------------------------------------------------------
# Contrastive loss (label embedding)
# ---------------------------------------------------------------------------


class ContrastiveLabelLoss(nn.Module):
    """Contrastive loss between document embeddings and label embeddings.

    Encourages document representations to be close to their positive
    label embeddings and far from negative ones.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, doc_emb: torch.Tensor, label_emb: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE-style contrastive loss.

        Args:
            doc_emb: ``(batch, embed_dim)`` document embeddings.
            label_emb: ``(n_nodes, embed_dim)`` label embeddings.
            labels: ``(batch, n_nodes)`` binary label matrix.

        Returns:
            Scalar loss.
        """
        # Normalize embeddings
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        label_emb = F.normalize(label_emb, p=2, dim=-1)

        # Similarity matrix: (batch, n_nodes)
        sim = torch.matmul(doc_emb, label_emb.T) / self.temperature

        # For each document, positive labels should have high similarity
        # Use a multi-label variant of InfoNCE
        pos_mask = labels > 0.5
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=doc_emb.device)

        # Log-sum-exp over all labels for each document
        lse = torch.logsumexp(sim, dim=-1)  # (batch,)

        # Mean log-prob of positive labels
        pos_sim = (sim * pos_mask.float()).sum(dim=-1) / pos_mask.float().sum(
            dim=-1
        ).clamp(min=1)

        loss = (lse - pos_sim).mean()
        return loss

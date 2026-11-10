"""Training objectives of the HPT + R-matrix model.

All objectives act on logits.  The max-constraint commutes with the sigmoid
(both are monotone), so reconciling logits is the same as reconciling
probabilities, and C-HMCNN's MCLoss carries over by replacing the ``y * h``
mask (probability 0) with a large negative logit.
"""

from __future__ import annotations

import torch

OBJECTIVES = ("none", "mc", "hinge")

# Logit standing in for "probability 0" in the MCLoss positive branch.  Finite
# so the backward pass of the scatter-amax never sees -inf - -inf.
_NEG = -1e4


def zmlce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Zero-bounded multi-label cross-entropy (Su, 2020; HPT's loss).

    ``log(1 + sum_neg e^s) + log(1 + sum_pos e^-s)`` per sample, averaged over
    the batch.  Its decision threshold is logit 0 by construction.
    """
    logits = logits.float()
    pos = y.bool()
    s = torch.where(pos, -logits, logits)
    zeros = s.new_zeros(s.shape[0], 1)
    neg_term = torch.logsumexp(
        torch.cat([s.masked_fill(pos, float("-inf")), zeros], 1), 1
    )
    pos_term = torch.logsumexp(
        torch.cat([s.masked_fill(~pos, float("-inf")), zeros], 1), 1
    )
    return (neg_term + pos_term).mean()


def mc_logits(logits: torch.Tensor, y: torch.Tensor, sparse_r) -> torch.Tensor:
    """C-HMCNN's max-constraint training outputs, in logit space.

    Negatives take the reconciled logits ``MCM(z)``; positives take
    ``MCM(z masked to the positive labels)``, so a positive node is only lifted
    by positive descendants.  ``sparse_r`` is a ``BlockDiagonalR`` over the
    full column space.
    """
    logits = logits.float()
    pos = y.bool()
    neg_out = sparse_r.reconcile(logits)
    pos_out = sparse_r.reconcile(logits.masked_fill(~pos, _NEG))
    return torch.where(pos, pos_out, neg_out)


def hpt_objective(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    kind: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    eval_cols: torch.Tensor,
    sparse_r,
    anc: torch.Tensor,
    desc: torch.Tensor,
    lambda_hier: float = 1.0,
) -> torch.Tensor:
    """ZMLCE plus the hierarchy term selected by ``kind``.

    * ``none``  -- ZMLCE alone (HPT);
    * ``mc``    -- ZMLCE on the MCLoss outputs (:func:`mc_logits`);
    * ``hinge`` -- ZMLCE + ``lambda_hier`` x the Eq. 2 penalty over all
      ancestor pairs (``anc``/``desc`` columns), on probabilities.
    """
    if kind not in OBJECTIVES:
        raise ValueError(f"unknown objective {kind!r}; expected one of {OBJECTIVES}")
    logits = logits.float()
    out = mc_logits(logits, y, sparse_r) if kind == "mc" else logits
    loss = zmlce(out[:, eval_cols], y[:, eval_cols])
    if kind == "hinge" and anc.numel():
        probs = torch.sigmoid(logits)
        loss = loss + lambda_hier * torch.relu(probs[:, desc] - probs[:, anc]).mean()
    return loss

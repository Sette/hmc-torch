"""Hierarchy constraint for the global classifier.

Implements the max-constraint module (MCM) and the training outputs of MCLoss
described by Giunchiglia & Lukasiewicz, "Coherent Hierarchical Multi-Label
Classification Networks" (NeurIPS 2020), written from the paper's equations.
"""

import torch


def _as_square(r_matrix: torch.Tensor) -> torch.Tensor:
    """Drop the optional leading batch axis of a ``(1, C, C)`` R-matrix."""
    return r_matrix.reshape(r_matrix.shape[-2], r_matrix.shape[-1])


def get_constr_out(x: torch.Tensor, r_matrix: torch.Tensor) -> torch.Tensor:
    """Apply the max-constraint module to a batch of scores.

    ``out[b, i] = max_j R[i, j] * x[b, j]``: with the pipeline convention
    (``R[i, j] = 1`` iff ``j`` is ``i`` or one of its descendants), every class
    scores at least as high as each of its descendants, so the output is
    consistent with the hierarchy.  Leaves are left unchanged.

    Args:
        x: Scores of shape ``(B, C)``.
        r_matrix: Constraint matrix of shape ``(C, C)`` or ``(1, C, C)``.

    Returns:
        torch.Tensor: The constrained scores, ``(B, C)`` in float64.
    """
    r = _as_square(r_matrix).to(torch.float64)
    scores = x.to(torch.float64)
    # (B, 1, C) * (1, C, C) -> (B, C, C); reduce over the candidate column j.
    weighted = scores.unsqueeze(1) * r.unsqueeze(0)
    return weighted.max(dim=-1).values


def mc_training_outputs(
    output: torch.Tensor, labels: torch.Tensor, r_matrix: torch.Tensor
) -> torch.Tensor:
    """Outputs that MCLoss feeds to the binary cross-entropy.

    A negative class sees the constrained score ``MCM(h)``.  A positive class
    sees ``MCM(y * h)``, the constraint restricted to positive classes, so its
    score is only raised by descendants that are positive too.

    Args:
        output: Unconstrained sigmoid outputs ``h``, ``(B, C)``.
        labels: Binary targets ``y``, ``(B, C)``.
        r_matrix: Constraint matrix, ``(C, C)`` or ``(1, C, C)``.

    Returns:
        torch.Tensor: The training outputs, ``(B, C)`` in float64.
    """
    labels = labels.to(torch.float64)
    on_negatives = get_constr_out(output, r_matrix)
    on_positives = get_constr_out(labels * output.to(torch.float64), r_matrix)
    return torch.where(labels.bool(), on_positives, on_negatives)

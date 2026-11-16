"""Global Hierarchy Violation (Yang & Howe, WWW'22, Section 4.3).

A hierarchy violation is an ancestor--descendant pair whose ancestor receives
a lower score than the descendant.  The metric compares every valid pair: one
whose shortest path does not pass through the root, i.e. every ancestor--
descendant pair in which neither node is a root.  Passing the
evaluation mask (``to_eval``) excludes the root, and on Gene Ontology also the
three sub-ontology roots.

The count is threshold-independent.  With ``threshold``, only the violations
that change a prediction are counted: the descendant is predicted and the
ancestor is not.  Those are the ones the paper's Figure 3(b) calls relevant.
"""

from __future__ import annotations

import numpy as np
import torch


def global_hierarchy_violation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    scores: np.ndarray | torch.Tensor,
    anc: np.ndarray,
    desc: np.ndarray,
    eval_mask: np.ndarray | None = None,
    threshold: float | None = None,
    chunk: int = 256,
) -> dict:
    """Count ancestor < descendant score pairs over all samples.

    Args:
        scores: ``(n_samples, n_labels)`` predicted probabilities.
        anc: ancestor column of every ancestor--descendant pair.
        desc: descendant column of every pair (same length as ``anc``).
        eval_mask: ``(n_labels,)`` boolean; pairs touching a column outside it
            are not valid.  ``None`` keeps every pair.
        threshold: count only pairs with ``desc >= threshold > anc``.
        chunk: samples per vectorized block.

    Returns:
        dict: ``"violations"`` (total count), ``"samples_with_violation"``,
        ``"pairs"`` (valid pairs per sample) and ``"rate"`` (violations per
        sample--pair).
    """
    anc = np.asarray(anc, dtype=np.int64)
    desc = np.asarray(desc, dtype=np.int64)
    if eval_mask is not None:
        mask = np.asarray(eval_mask, dtype=bool)
        keep = mask[anc] & mask[desc]
        anc, desc = anc[keep], desc[keep]
    s = torch.as_tensor(np.asarray(scores) if not torch.is_tensor(scores) else scores)
    n = int(s.shape[0])
    if len(anc) == 0 or n == 0:
        return {
            "violations": 0,
            "samples_with_violation": 0,
            "pairs": len(anc),
            "rate": 0.0,
        }
    a_idx = torch.from_numpy(anc).to(s.device)
    d_idx = torch.from_numpy(desc).to(s.device)
    total, samples = 0, 0
    for i in range(0, n, chunk):
        block = s[i : i + chunk]
        pa, pd = block[:, a_idx], block[:, d_idx]
        bad = pa < pd if threshold is None else (pd >= threshold) & (pa < threshold)
        total += int(bad.sum())
        samples += int(bad.any(dim=1).sum())
    return {
        "violations": total,
        "samples_with_violation": samples,
        "pairs": len(anc),
        "rate": total / (n * len(anc)),
    }

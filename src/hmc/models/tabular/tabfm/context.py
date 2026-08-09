"""Conditional node dataset and stratified context sampler for TabFM.

Each node in the hierarchy is treated as a binary classification task.
The context for a node is composed of rows where its **parent** is
positive (or the full dataset for root-level nodes).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConditionalNodeDataset:
    """Binary dataset for one hierarchy node.

    For a given node, uses only training rows where the node's
    **parent** label is positive.  For root-level nodes, all rows
    are used.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix ``(n_samples, n_features)``.
    y : np.ndarray
        Label matrix ``(n_samples, n_nodes)``.
    node_idx : int
        Global index of the target node.
    parent_indices : list[int]
        Global indices of the node's parents.
        For a tree this is a single index; for a DAG it may be
        multiple.  If empty, the node is treated as root (all rows).
    verbose : bool
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node_idx: int,
        parent_indices: list[int],
        verbose: bool = False,
    ):
        self.X = X
        self.y_full = y
        self.node_idx = node_idx
        self.parent_indices = parent_indices
        self.verbose = verbose

        # Build mask: rows where ALL parents are positive
        if not parent_indices:
            # Root node: use all rows
            self.mask = np.ones(len(X), dtype=bool)
        else:
            self.mask = y[:, parent_indices].all(axis=1).astype(bool)

        self.X_filtered = X[self.mask]  # pylint: disable=invalid-name
        self.y_filtered = y[self.mask, node_idx].astype(np.float32)

        n_pos = int(self.y_filtered.sum())
        n_total = int(self.mask.sum())
        logger.debug(
            "Node %d: %d/%d rows selected (%.1f%% positive)",
            node_idx,
            n_pos,
            n_total,
            100 * n_pos / max(1, n_total),
        )

    @property
    def n_samples(self) -> int:
        """Number of positive samples in the context."""
        return int(self.mask.sum())

    @property
    def prevalence(self) -> float:
        """Prevalence of positive labels in the context."""
        return float(self.y_filtered.mean()) if self.n_samples > 0 else 0.0


class StratifiedContextSampler:
    """Draw balanced contexts for TabFM inference.

    For each node, samples *k_pos* positive and *k_neg* negative
    examples from the training set as context for TabFM's in-context
    learning mechanism.

    Parameters
    ----------
    k_pos : int
        Number of positive examples per context (default 50).
    k_neg : int
        Number of negative examples per context (default 50).
    seed : int
        Random seed for reproducibility.
    fallback : str
        Strategy when insufficient positives/negatives:
        - ``"reduce"``: use whatever is available.
        - ``"resample"``: sample with replacement.
    """

    def __init__(
        self,
        k_pos: int = 50,
        k_neg: int = 50,
        seed: int = 42,
        fallback: str = "reduce",
    ):
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.seed = seed
        self.fallback = fallback
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node_idx: int,
        parent_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a context (X_ctx, y_ctx) for a single node.

        Args:
            X: Full training features.
            y: Full training labels.
            node_idx: Target node index.
            parent_indices: Parent indices; if None or empty,
                uses the full dataset.

        Returns:
            ``(X_ctx, y_ctx)`` — contextual features and binary labels.
            Shapes are ``(ctx_size, n_features)`` and ``(ctx_size,)``.
        """
        # Filter to rows where parents are positive
        if parent_indices:
            mask = y[:, parent_indices].all(axis=1).astype(bool)
        else:
            mask = np.ones(len(X), dtype=bool)

        if mask.sum() == 0:
            logger.warning("Node %d: no rows with positive parents", node_idx)
            return (
                np.zeros((0, X.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        X_filt = X[mask]
        y_filt = y[mask, node_idx].astype(np.float32)

        pos_idx = np.where(y_filt >= 0.5)[0]
        neg_idx = np.where(y_filt < 0.5)[0]

        # Handle insufficient examples
        k_pos = (
            min(self.k_pos, len(pos_idx)) if self.fallback == "reduce" else self.k_pos
        )
        k_neg = (
            min(self.k_neg, len(neg_idx)) if self.fallback == "reduce" else self.k_neg
        )

        if len(pos_idx) == 0 and len(neg_idx) == 0:
            return (
                np.zeros((0, X.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        sampled_pos = (
            self._rng.choice(pos_idx, size=k_pos, replace=self.fallback == "resample")
            if len(pos_idx) > 0 and k_pos > 0
            else np.array([], dtype=int)
        )

        sampled_neg = (
            self._rng.choice(neg_idx, size=k_neg, replace=self.fallback == "resample")
            if len(neg_idx) > 0 and k_neg > 0
            else np.array([], dtype=int)
        )

        all_idx = np.concatenate([sampled_pos, sampled_neg])
        self._rng.shuffle(all_idx)

        return X_filt[all_idx].copy(), y_filt[all_idx].copy()

    def sample_multiple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node_indices: list[int],
        parent_map: dict[int, list[int]],
        n_contexts: int = 1,
    ) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
        """Sample *n_contexts* contexts per node.

        Returns ``{node_idx: [(X_ctx, y_ctx), ...]}``.
        """
        results: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for nid in node_indices:
            contexts = []
            parents = parent_map.get(nid, [])
            for _ in range(n_contexts):
                ctx = self.sample(X, y, nid, parents)
                if ctx[0].shape[0] > 0:
                    contexts.append(ctx)
            results[nid] = contexts
        return results

"""Per-node conditional classifier using TabFM.

Treats each hierarchy node as an independent binary classification task,
conditioned on the parent node being positive.
"""

from __future__ import annotations

import logging

import numpy as np

from hmc.models.tabular.tabfm.adapter import TabFMAdapter
from hmc.models.tabular.tabfm.context import StratifiedContextSampler

logger = logging.getLogger(__name__)


class ConditionalNodeClassifier:
    """One binary classifier per hierarchy node, conditioned on parent.

    For each node, samples contexts from rows where the parent is
    positive, then uses TabFM for in-context binary classification.

    Parameters
    ----------
    adapter : TabFMAdapter
        Lazy TabFM wrapper.
    context_sampler : StratifiedContextSampler
        Context sampling strategy.
    n_contexts : int
        Number of context sets per node (1, 8, or 32 typical).
    calibrate : bool
        Whether to calibrate predictions with Platt scaling.
    """

    def __init__(
        self,
        adapter: TabFMAdapter,
        context_sampler: StratifiedContextSampler,
        n_contexts: int = 8,
        calibrate: bool = True,
    ):
        self.adapter = adapter
        self.context_sampler = context_sampler
        self.n_contexts = n_contexts
        self.calibrate = calibrate

        self._contexts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._calibrators: dict[int, object] = {}
        self._parent_map: dict[int, list[int]] = {}
        self._node_order: list[int] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hierarchy,
        eval_mask: np.ndarray | None = None,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> ConditionalNodeClassifier:
        """Prepare contexts for all nodes and optionally calibrate.

        Args:
            X_train: Training features.
            y_train: Training labels.
            hierarchy: :class:`TreeHierarchy` or :class:`DagHierarchy`.
            eval_mask: Boolean mask over nodes to train.
            X_valid: Validation features (for calibration).
            y_valid: Validation labels (for calibration).

        Returns:
            Self.
        """
        n_nodes = y_train.shape[1]
        mask = eval_mask if eval_mask is not None else np.ones(n_nodes, dtype=bool)

        # Build parent map
        self._parent_map = {}
        for node in hierarchy.nodes:
            node_idx = hierarchy.node_index[node]
            if not mask[node_idx]:
                continue
            parents = hierarchy.parents(node)
            parent_indices = [
                hierarchy.node_index[p] for p in parents if p in hierarchy.node_index
            ]
            self._parent_map[node_idx] = parent_indices

        # Sample contexts for each node
        self._node_order = [i for i in range(n_nodes) if mask[i]]
        self._contexts = self.context_sampler.sample_multiple(
            X_train,
            y_train,
            node_indices=self._node_order,
            parent_map=self._parent_map,
            n_contexts=self.n_contexts,
        )

        # Calibration on validation set
        if self.calibrate and X_valid is not None and y_valid is not None:
            from hmc.models.hierarchical.postprocess import (  # pylint: disable=import-outside-toplevel
                PlattCalibrator,
            )

            # Get raw scores on validation
            val_scores = self._predict_raw(X_valid, y_valid)
            cal = PlattCalibrator()
            cal.fit(val_scores, y_valid, eval_mask=eval_mask)
            self._calibrators["platt"] = cal

        logger.info(
            "ConditionalNodeClassifier fitted: %d nodes, %d contexts/node",
            len(self._node_order),
            self.n_contexts,
        )
        return self

    def _predict_raw(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Run TabFM inference for all nodes without calibration.

        Returns ``(n_samples, n_nodes)`` score matrix.
        """
        self.adapter.load()
        n_samples = X.shape[0]
        n_nodes = y_train.shape[1]
        scores = np.zeros((n_samples, n_nodes), dtype=np.float32)

        # For each node, average predictions across contexts
        for nid in self._node_order:
            contexts = self._contexts.get(nid, [])
            if not contexts:
                continue

            node_scores = np.zeros(n_samples, dtype=np.float32)
            n_valid_contexts = 0

            for X_ctx, y_ctx in contexts:
                if X_ctx.shape[0] == 0:
                    continue
                try:
                    # TabFM encodes context + target sample together
                    # This is a simplified interface; real TabFM may differ
                    pred = self.adapter.predict(X, context=(X_ctx, y_ctx))
                    if pred is not None:
                        node_scores += np.asarray(pred, dtype=np.float32).flatten()
                        n_valid_contexts += 1
                except (ValueError, RuntimeError):
                    logger.debug("TabFM inference failed for node %d", nid)
                    continue

            if n_valid_contexts > 0:
                scores[:, nid] = node_scores / n_valid_contexts

        return scores

    def predict_proba(self, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Return calibrated probability matrix.

        Args:
            X: ``(n_samples, n_features)``.
            y_train: Training labels (needed for context retrieval).

        Returns:
            ``(n_samples, n_nodes)`` probability matrix.
        """
        scores = self._predict_raw(X, y_train)

        if "platt" in self._calibrators:
            scores = self._calibrators["platt"].calibrate(scores)

        return np.clip(scores, 0.0, 1.0)

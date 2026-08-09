"""Post-processing for hierarchical predictions: reconciliation and calibration.

Reconciliation ensures that for every node, parent scores are >= child scores
(no hierarchy violations).  Calibration maps raw scores to well-calibrated
probabilities using Platt scaling or isotonic regression.
"""

from __future__ import annotations


import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from hmc.data.hierarchy import Hierarchy

# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile_tree(scores: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
    """Bottom-up tree reconciliation: parent >= max(children).

    Args:
        scores: ``(batch, n_nodes)`` raw score matrix.
        hierarchy: A :class:`TreeHierarchy` instance.

    Returns:
        Reconciled scores of the same shape.
    """
    result = scores.copy()
    node_idx = hierarchy.node_index
    for level in sorted(hierarchy.levels.keys(), reverse=True):
        for node in hierarchy.levels[level]:
            children = hierarchy.children(node)
            if not children:
                continue
            child_indices = [node_idx[c] for c in children]
            n_idx = node_idx[node]
            max_child = result[:, child_indices].max(axis=1)
            result[:, n_idx] = np.maximum(result[:, n_idx], max_child)
    return result


def reconcile_dag(
    scores: np.ndarray, hierarchy: Hierarchy, strategy: str = "max_path"
) -> np.ndarray:
    """DAG reconciliation supporting ``"max_path"`` and ``"ancestor_max"``.

    Args:
        scores: ``(batch, n_nodes)`` raw score matrix.
        hierarchy: A :class:`DagHierarchy` instance.
        strategy: One of ``"max_path"`` or ``"ancestor_max"``.

    Returns:
        Reconciled scores.
    """
    if strategy not in ("max_path", "ancestor_max"):
        raise ValueError(
            f"Unknown DAG strategy '{strategy}'; use 'max_path' or 'ancestor_max'"
        )
    result = scores.copy()
    node_idx = hierarchy.node_index
    node_list = hierarchy.nodes

    if strategy == "ancestor_max":
        for level in sorted(hierarchy.levels.keys(), reverse=True):
            for node in hierarchy.levels[level]:
                parents = hierarchy.parents(node)
                if not parents:
                    continue
                n_idx = node_idx[node]
                parent_indices = [node_idx[p] for p in parents if p in node_idx]
                for p_idx in parent_indices:
                    result[:, p_idx] = np.maximum(result[:, p_idx], result[:, n_idx])
    else:  # max_path
        g_parent_to_child = hierarchy._graph.reverse()  # pylint: disable=protected-access
        for node in node_list:
            ancestors = set()
            try:
                ancestors = nx_ancestors(g_parent_to_child, node)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            if not ancestors:
                continue
            anc_indices = [node_idx[a] for a in ancestors if a in node_idx]
            n_idx = node_idx[node]
            max_ancestor = result[:, anc_indices].max(axis=1)
            result[:, n_idx] = np.maximum(result[:, n_idx], max_ancestor)

    return result


def reconcile(
    scores: np.ndarray, hierarchy: Hierarchy, strategy: str = "ancestor_max"
) -> np.ndarray:
    """Reconcile scores for any hierarchy type.

    Args:
        scores: ``(batch, n_nodes)`` raw score matrix.
        hierarchy: Tree or DAG hierarchy.
        strategy: For trees, only ``"ancestor_max"``.  For DAGs, also
            ``"max_path"`` is supported.

    Returns:
        Reconciled scores.
    """
    if hierarchy.is_dag:
        return reconcile_dag(scores, hierarchy, strategy)
    return reconcile_tree(scores, hierarchy)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Platt scaling: fits a logistic regression on raw scores → binary labels.

    Trained on the *validation* split only.
    """

    def __init__(self):
        self._models: dict[int, LogisticRegression] = {}

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        eval_mask: np.ndarray | None = None,
    ) -> PlattCalibrator:
        """Fit one logistic regressor per class node.

        Args:
            scores: ``(n_samples, n_nodes)`` raw scores.
            labels: ``(n_samples, n_nodes)`` binary ground-truth.
            eval_mask: ``(n_nodes,)`` boolean mask; only these nodes are
                calibrated.  If ``None``, all nodes are calibrated.

        Returns:
            Self.
        """
        n_nodes = scores.shape[1]
        mask = eval_mask if eval_mask is not None else np.ones(n_nodes, dtype=bool)
        for j in range(n_nodes):
            if not mask[j]:
                continue
            s_j = scores[:, j].reshape(-1, 1)
            y_j = labels[:, j]
            # Skip nodes with only one class present
            if len(np.unique(y_j)) < 2:
                continue
            lr = LogisticRegression(penalty=None, solver="lbfgs")
            lr.fit(s_j, y_j)
            self._models[j] = lr
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw scores.

        Args:
            scores: ``(n_samples, n_nodes)``.

        Returns:
            Calibrated probabilities, same shape.
        """
        result = scores.copy()
        for j, lr in self._models.items():
            result[:, j] = lr.predict_proba(scores[:, j].reshape(-1, 1))[:, 1]
        return np.clip(result, 0.0, 1.0)


class IsotonicCalibrator:
    """Isotonic regression calibration — more flexible than Platt.

    Trained on the *validation* split only.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.y_min = y_min
        self.y_max = y_max
        self._models: dict[int, IsotonicRegression] = {}

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        eval_mask: np.ndarray | None = None,
    ) -> IsotonicCalibrator:
        """Fit one isotonic regressor per class node.

        Args:
            scores: ``(n_samples, n_nodes)`` raw scores.
            labels: ``(n_samples, n_nodes)`` binary ground-truth.
            eval_mask: Optional boolean mask over nodes.

        Returns:
            Self.
        """
        n_nodes = scores.shape[1]
        mask = eval_mask if eval_mask is not None else np.ones(n_nodes, dtype=bool)
        for j in range(n_nodes):
            if not mask[j]:
                continue
            s_j = scores[:, j].astype(np.float64)
            y_j = labels[:, j].astype(np.float64)
            if len(np.unique(y_j)) < 2:
                continue
            iso = IsotonicRegression(
                y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip"
            )
            iso.fit(s_j, y_j)
            self._models[j] = iso
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration.

        Args:
            scores: ``(n_samples, n_nodes)``.

        Returns:
            Calibrated probabilities, same shape.
        """
        result = scores.astype(np.float64).copy()
        for j, iso in self._models.items():
            result[:, j] = iso.transform(result[:, j])
        return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nx_ancestors(g, node):
    """Polyfill for ``nx.ancestors`` that handles edge cases."""
    import networkx as nx  # pylint: disable=import-outside-toplevel

    try:
        return nx.ancestors(g, node)
    except Exception:  # pylint: disable=broad-exception-caught
        return set()

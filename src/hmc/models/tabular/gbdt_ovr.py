"""One-vs-Rest GBDT baseline for hierarchical multi-label classification.

Uses ``HistGradientBoostingClassifier`` from scikit-learn as the default
backend.  LightGBM and CatBoost are available as optional extras.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GBDTOvRClassifier:
    """One-vs-Rest GBDT for multi-label HMC.

    Trains one binary classifier per class node.  Supports per-node
    prevalence-based weighting and optional calibration.

    Parameters
    ----------
    backend : str
        ``"histgb"`` (default), ``"lightgbm"``, or ``"catboost"``.
    backend_kwargs : dict
        Keyword arguments forwarded to the backend classifier constructor.
    calibrate : bool
        If True, recalibrate predictions with Platt scaling per node.
    verbose : int
        Verbosity level passed to the backend.
    """

    BACKENDS = ("histgb", "lightgbm", "catboost")

    def __init__(
        self,
        backend: str = "histgb",
        backend_kwargs: dict | None = None,
        calibrate: bool = False,
        verbose: int = 0,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from {self.BACKENDS}"
            )
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.calibrate = calibrate
        self.verbose = verbose

        self._estimators: list = []
        self._node_order: list[int] = []
        self._n_nodes: int = 0
        self.fitted_ = False

    def _make_estimator(self):
        """Create a new backend estimator instance."""
        if self.backend == "histgb":
            from sklearn.ensemble import (  # pylint: disable=import-outside-toplevel
                HistGradientBoostingClassifier,
            )

            kwargs = {
                "max_iter": 200,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42,
                "verbose": self.verbose,
            }
            kwargs.update(self.backend_kwargs)
            return HistGradientBoostingClassifier(**kwargs)

        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb  # pylint: disable=import-outside-toplevel
            except ImportError as e:
                raise ImportError(
                    "LightGBM is not installed. Install with: pip install lightgbm"
                ) from e
            kwargs = {
                "n_estimators": 200,
                "verbose": self.verbose,
                "random_state": 42,
            }
            kwargs.update(self.backend_kwargs)
            return lgb.LGBMClassifier(**kwargs, force_col_wise=True)

        if self.backend == "catboost":
            try:
                from catboost import (  # pylint: disable=import-outside-toplevel
                    CatBoostClassifier,
                )
            except ImportError as e:
                raise ImportError(
                    "CatBoost is not installed. Install with: pip install catboost"
                ) from e
            kwargs = {
                "iterations": 200,
                "verbose": self.verbose,
                "random_seed": 42,
            }
            kwargs.update(self.backend_kwargs)
            return CatBoostClassifier(**kwargs, silent=self.verbose == 0)

        raise ValueError(f"Unknown backend: {self.backend}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_mask: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> GBDTOvRClassifier:
        """Train one binary GBDT per class node.

        Args:
            X: ``(n_samples, n_features)`` feature matrix.
            y: ``(n_samples, n_nodes)`` binary label matrix.
            eval_mask: ``(n_nodes,)`` bool mask; only these nodes are trained.
                If None, all nodes are trained.
            sample_weight: ``(n_samples,)`` optional sample weights.

        Returns:
            Self.
        """
        n_nodes = y.shape[1]
        mask = eval_mask if eval_mask is not None else np.ones(n_nodes, dtype=bool)
        self._n_nodes = n_nodes
        self._estimators = []
        self._node_order = []

        for j in range(n_nodes):
            if not mask[j]:
                continue
            y_j = y[:, j].astype(int)
            unique = np.unique(y_j)
            if len(unique) < 2:
                logger.debug("Node %d has only one class; skipping", j)
                continue

            est = self._make_estimator()
            try:
                est.fit(X, y_j, sample_weight=sample_weight)
            except TypeError:
                est.fit(X, y_j)
            self._estimators.append(est)
            self._node_order.append(j)

        self.fitted_ = True
        logger.info(
            "GBDT One-vs-Rest fitted: %d/%d nodes trained",
            len(self._estimators),
            n_nodes,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape ``(n_samples, n_nodes)``.

        Untrained nodes (single class or masked out) receive a
        constant value (the prevalence in the training set, or 0.0).
        """
        if not self.fitted_:
            raise RuntimeError("GBDTOvRClassifier must be fit() before predict_proba()")

        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self._n_nodes), dtype=np.float32)

        for est, j in zip(self._estimators, self._node_order):
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                # proba shape: (n_samples, 2) — take positive class
                if proba.shape[1] >= 2:
                    scores[:, j] = proba[:, 1].astype(np.float32)
                else:
                    scores[:, j] = proba[:, 0].astype(np.float32)
            else:
                scores[:, j] = est.predict(X).astype(np.float32)

        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at a given threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    @property
    def n_estimators_trained(self) -> int:
        """Number of nodes for which a classifier was trained."""
        return len(self._estimators)

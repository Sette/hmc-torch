"""Feature preprocessing for tabular HMC datasets.

Imputation, standardisation, and optional feature selection fitted
exclusively on the training split.
"""

from __future__ import annotations


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    """Imputation + standardisation + optional feature selection.

    All parameters are fitted on ``fit()`` (train split only) and
    applied via ``transform()``.  The preprocessor is serialisable
    via ``get_params()`` / ``set_params()`` so the transformation
    can be saved alongside model artefacts.

    Parameters
    ----------
    with_imputation : bool
        If True (default), impute NaN values with the column mean.
    with_scaling : bool
        If True (default), standardise to zero mean / unit variance.
    feature_selector : str or None
        - ``"variance"`` — remove features with variance below ``selector_kwargs["threshold"]``.
        - ``"mutual_info"`` — keep top-k features by mutual information with any label.
        - ``None`` — keep all features.
    selector_kwargs : dict
        Additional arguments for the selector (e.g. ``{"k": 500}`` for mutual_info).
    """

    def __init__(
        self,
        with_imputation: bool = True,
        with_scaling: bool = True,
        feature_selector: str | None = None,
        selector_kwargs: dict | None = None,
    ):
        self.with_imputation = with_imputation
        self.with_scaling = with_scaling
        self.feature_selector = feature_selector
        self.selector_kwargs = selector_kwargs or {}

        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._selected_indices: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TabularPreprocessor:
        """Fit imputer, scaler, and feature selector on training data.

        Args:
            X: ``(n_samples, n_features)`` feature matrix.
            y: ``(n_samples, n_nodes)`` label matrix (needed for mutual_info).

        Returns:
            Self.
        """
        X_work = X.astype(np.float32).copy()

        if self.with_imputation:
            self._imputer = SimpleImputer(strategy="mean")
            X_work = self._imputer.fit_transform(X_work)

        if self.with_scaling:
            self._scaler = StandardScaler()
            X_work = self._scaler.fit_transform(X_work)

        # Feature selection
        n_features = X_work.shape[1]
        if self.feature_selector == "variance" and n_features > 0:
            threshold = self.selector_kwargs.get("threshold", 0.0)
            variances = X_work.var(axis=0)
            self._selected_indices = np.where(variances > threshold)[0]
            if len(self._selected_indices) == 0:
                self._selected_indices = np.arange(n_features)

        elif (
            self.feature_selector == "mutual_info" and y is not None and n_features > 0
        ):
            k = self.selector_kwargs.get("k", min(500, n_features))
            try:
                from sklearn.feature_selection import (  # pylint: disable=import-outside-toplevel
                    mutual_info_classif,
                )

                # Use the union of labels as the classification target
                y_cat = np.argmax(y, axis=1) if y.ndim > 1 else y
                if y.ndim > 1 and y.shape[1] > 1:
                    # Multi-label: use a combined score (sum of mutual info across labels)
                    mi_scores = np.zeros(n_features)
                    for j in range(y.shape[1]):
                        if y[:, j].sum() > 0 and (1 - y[:, j]).sum() > 0:
                            mi_j = mutual_info_classif(
                                X_work,
                                y[:, j].astype(int),
                                random_state=42,
                            )
                            mi_scores += mi_j
                else:
                    mi_scores = mutual_info_classif(X_work, y_cat, random_state=42)
                self._selected_indices = np.argsort(mi_scores)[-k:]
            except ImportError:
                self._selected_indices = np.arange(n_features)
        else:
            self._selected_indices = np.arange(n_features)

        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transformations.

        Args:
            X: ``(n_samples, n_features)``.

        Returns:
            ``(n_samples, n_selected_features)``.
        """
        if not self.fitted_:
            raise RuntimeError("TabularPreprocessor must be fit() before transform()")

        X_work = X.astype(np.float32).copy()

        if self._imputer is not None:
            X_work = self._imputer.transform(X_work)

        if self._scaler is not None:
            X_work = self._scaler.transform(X_work)

        if self._selected_indices is not None:
            X_work = X_work[:, self._selected_indices]

        return X_work

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    @property
    def n_features_out(self) -> int:
        """Number of features after preprocessing."""
        if self._selected_indices is not None:
            return len(self._selected_indices)
        return -1  # unknown until fitted

    def get_params(self) -> dict:
        """Serialisable parameter dict for run-config.json."""
        return {
            "with_imputation": self.with_imputation,
            "with_scaling": self.with_scaling,
            "feature_selector": self.feature_selector,
            "selector_kwargs": self.selector_kwargs,
            "n_features_selected": (
                len(self._selected_indices)
                if self._selected_indices is not None
                else -1
            ),
        }

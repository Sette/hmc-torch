"""Tabular feature encoder — imputation, scaling, selection, and caching.

Wraps :class:`TabularPreprocessor` in the :class:`FeatureEncoder` protocol.
"""

from __future__ import annotations

from hmc.data.base import FeatureEncoder, Split
from hmc.models.tabular.preprocessing import TabularPreprocessor


class TabularFeatureEncoder(FeatureEncoder):
    """Imputation + standardisation + optional feature selection.

    Parameters
    ----------
    with_imputation : bool
    with_scaling : bool
    feature_selector : str or None
    selector_kwargs : dict
    """

    def __init__(
        self,
        with_imputation: bool = True,
        with_scaling: bool = True,
        feature_selector: str | None = None,
        selector_kwargs: dict | None = None,
    ):
        self._pp = TabularPreprocessor(
            with_imputation=with_imputation,
            with_scaling=with_scaling,
            feature_selector=feature_selector,
            selector_kwargs=selector_kwargs,
        )

    def fit(self, split: Split) -> TabularFeatureEncoder:
        self._pp.fit(split.features, split.labels)
        return self

    def transform(self, split: Split) -> Split:
        X_out = self._pp.transform(split.features)
        return Split(
            features=X_out,
            labels=split.labels,
            local_labels=split.local_labels,
            sample_ids=split.sample_ids,
            metadata=split.metadata,
        )

    @property
    def n_features_out(self) -> int:
        return self._pp.n_features_out

    def get_params(self) -> dict:
        return self._pp.get_params()

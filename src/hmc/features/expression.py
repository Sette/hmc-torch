"""Gene-expression feature encoder.

Includes normalisation and optional denoising/masked autoencoder.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hmc.data.base import FeatureEncoder, Modality, Split

logger = logging.getLogger(__name__)


class ExpressionFeatureEncoder(FeatureEncoder):
    """Normalise and optionally autoencode gene-expression data.

    Parameters
    ----------
    normalize : bool
        If True, apply log1p + StandardScaler.
    use_autoencoder : bool
        If True, encode via a denoising/masked autoencoder.
    autoencoder_kwargs : dict
        Passed to :class:`ExpressionAutoencoder`.
    """

    def __init__(
        self,
        normalize: bool = True,
        use_autoencoder: bool = False,
        autoencoder_kwargs: Optional[dict] = None,
    ):
        self.normalize = normalize
        self.use_autoencoder = use_autoencoder
        self.autoencoder_kwargs = autoencoder_kwargs or {}
        self._scaler = None
        self._autoencoder = None
        self._fitted = False

    def fit(self, split: Split) -> "ExpressionFeatureEncoder":
        X = split.features.astype(np.float32).copy()

        if self.normalize:
            from sklearn.preprocessing import StandardScaler

            X = np.log1p(np.maximum(X, 0))
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.use_autoencoder:
            from hmc.models.expression.autoencoder import (  # pylint: disable=import-outside-toplevel
                ExpressionAutoencoder,
            )
            self._autoencoder = ExpressionAutoencoder(
                input_dim=X.shape[1],
                **self.autoencoder_kwargs,
            )
            self._autoencoder.fit(X)

        self._fitted = True
        return self

    def transform(self, split: Split) -> Split:
        if not self._fitted:
            raise RuntimeError("Encoder must be fit() before transform()")

        X = split.features.astype(np.float32).copy()

        if self.normalize and self._scaler is not None:
            X = np.log1p(np.maximum(X, 0))
            X = self._scaler.transform(X)

        if self.use_autoencoder and self._autoencoder is not None:
            X = self._autoencoder.encode(X)

        return Split(
            features=X,
            labels=split.labels,
            local_labels=split.local_labels,
            sample_ids=split.sample_ids,
            metadata=split.metadata,
        )

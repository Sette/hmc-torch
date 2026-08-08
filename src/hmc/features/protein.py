"""Protein sequence feature encoder (placeholder).

Generates embeddings via ESM-2 or ProtT5, cached per sequence and
model version.  Activated only when raw sequences are available.
"""

from __future__ import annotations

import logging

from hmc.data.base import FeatureEncoder, Split

logger = logging.getLogger(__name__)


class ProteinFeatureEncoder(FeatureEncoder):
    """Encode protein sequences with ESM-2 / ProtT5.

    Parameters
    ----------
    model_name : str
        ``"esm2_t33_650M_UR50D"`` (default) or a ProtT5 variant.
    device : str
    batch_size : int
    cache_dir : str
        Directory for per-sequence embedding cache.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cpu",
        batch_size: int = 16,
        cache_dir: str = "./.protein_cache",
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def fit(self, split: Split) -> ProteinFeatureEncoder:
        return self  # frozen embeddings, no fine-tuning initially

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import importlib

            importlib.import_module("esm")
            logger.info("ESM loaded for protein encoding")
        except ImportError:
            logger.warning("ESM not installed — install with: pip install fair-esm")

    def transform(self, split: Split) -> Split:
        """Encode sequences if raw data available; otherwise passthrough."""
        if not split.metadata.raw_available:
            logger.info(
                "Protein encoder: no raw sequences available, "
                "passing through pre-extracted features"
            )
            return split

        self._ensure_loaded()
        if self._model is None:
            return split

        logger.warning(
            "Protein encoder: raw sequences confirmed but encoding not "
            "yet implemented — passing through features"
        )
        return split

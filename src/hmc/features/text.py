"""Text feature encoder — transformer-based (SPECTER2, SciBERT, …).

Placeholder until raw text is confirmed available.
When raw text is not available, falls back to pre-extracted features
(treating them as tabular).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hmc.data.base import FeatureEncoder, Modality, Split

logger = logging.getLogger(__name__)


class TextFeatureEncoder(FeatureEncoder):
    """Encode raw text via a HuggingFace transformer.

    If raw text is unavailable, this encoder is a no-op (passthrough)
    and logs a warning.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"allenai/specter2_base"``).
    batch_size : int
        Encoding batch size.
    device : str
        PyTorch device.
    model_cache_dir : str
        Directory for cached model weights.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        batch_size: int = 32,
        device: str = "cpu",
        model_cache_dir: str = "./models",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model_cache_dir = model_cache_dir
        self._model = None
        self._tokenizer = None
        self._fitted = False

    def fit(self, split: Split) -> "TextFeatureEncoder":
        """Load tokenizer and model (no training needed for frozen encoder)."""
        self._fitted = True
        # Lazy-load on first transform
        return self

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.warning(
                "Transformers not installed; text encoder will passthrough"
            )
            return
        try:
            from hmc.utils.model_cache import (  # pylint: disable=import-outside-toplevel
                ensure_transformer_model_cached,
            )
            local_path = ensure_transformer_model_cached(
                self.model_name, self.model_cache_dir
            )
        except Exception:
            local_path = self.model_name

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                local_path, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                local_path, local_files_only=True
            ).to(self.device)
            self._model.eval()
        except Exception as e:
            logger.warning("Could not load transformer model: %s", e)

    def transform(self, split: Split) -> Split:
        """Encode features if raw text is provided; otherwise passthrough."""
        if split.metadata.modality != Modality.TEXT or not split.metadata.raw_available:
            logger.info("Text encoder: no raw text available, passing through features")
            return split

        self._ensure_loaded()
        if self._model is None:
            return split

        import torch

        # Split.features is expected to be a list of text strings
        texts = split.features
        if isinstance(texts, np.ndarray) and texts.dtype.kind in ("U", "O"):
            texts = texts.tolist()
        if not isinstance(texts, list) or not isinstance(texts[0], str):
            return split

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._model(**encoded)
                cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)

        X_out = np.concatenate(embeddings, axis=0).astype(np.float32)
        return Split(
            features=X_out,
            labels=split.labels,
            local_labels=split.local_labels,
            sample_ids=split.sample_ids,
            metadata=split.metadata,
        )

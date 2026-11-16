"""Protein sequence feature encoder.

Embeds amino-acid sequences with a protein language model (ESM-2 by default)
through ``transformers``: per-residue states are mean-pooled without the
special tokens, and proteins longer than the model window are cut into
windows whose residue states are averaged together.  Embeddings are cached on
disk under an MD5 of (model, sequences).

ESM-2 is pre-trained on UniRef sequences only, so it carries no GO
annotation; models pre-trained with GO terms (e.g. ProteinBERT) would leak
the test labels and must not be used here.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import replace

import numpy as np

from hmc.data.base import FeatureEncoder, Split
from hmc.utils.model_cache import ensure_transformer_model_cached

logger = logging.getLogger(__name__)

DEFAULT_PROTEIN_MODEL = "facebook/esm2_t33_650M_UR50D"


def sequences_hash(model_name: str, sequences: list[str]) -> str:
    """MD5 of the model name and the ordered sequences (cache key)."""
    digest = hashlib.md5(model_name.encode())
    for seq in sequences:
        digest.update(b"\0" + seq.encode())
    return digest.hexdigest()


class ProteinFeatureEncoder(FeatureEncoder):
    """Encode protein sequences with a frozen protein language model.

    Parameters
    ----------
    model_name : str
        HuggingFace id of an ESM-2 checkpoint (default 650M).
    sequences : dict[str, str] | None
        ``sample_id -> sequence``, used by :meth:`transform`.
    device : str | None
        ``None`` picks CUDA when available.
    batch_tokens : int
        Residues per forward pass (sequences are batched by length).
    window : int
        Residues per window; ESM-2 takes 1024 tokens including BOS/EOS.
    cache_dir : str
        Directory for the embedding cache; ``None`` disables it.
    model_cache_dir : str
        Where the HuggingFace weights are kept.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model_name: str = DEFAULT_PROTEIN_MODEL,
        sequences: dict[str, str] | None = None,
        device: str | None = None,
        batch_tokens: int = 8192,
        window: int = 1022,
        cache_dir: str | None = "./models/protein_embeddings",
        model_cache_dir: str = "./models",
    ):
        self.model_name = model_name
        self.sequences = sequences or {}
        self.device = device
        self.batch_tokens = batch_tokens
        self.window = window
        self.cache_dir = cache_dir
        self.model_cache_dir = model_cache_dir
        self._model = None
        self._tokenizer = None

    def fit(self, split: Split) -> ProteinFeatureEncoder:
        return self  # frozen embeddings

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModel,
            AutoTokenizer,
        )

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = ensure_transformer_model_cached(self.model_name, self.model_cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModel.from_pretrained(
            path, local_files_only=True, dtype=dtype, add_pooling_layer=False
        )
        self._model.eval().to(self.device)
        logger.info("Loaded %s on %s", self.model_name, self.device)

    def unload(self):
        """Drop the language model (and its GPU memory) once encoding is done."""
        if self._model is None:
            return
        import torch  # pylint: disable=import-outside-toplevel

        self._model = self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embed_windows(self, windows: list[str]) -> np.ndarray:
        """Mean residue embedding of each window (no BOS/EOS/padding)."""
        import torch  # pylint: disable=import-outside-toplevel

        enc = self._tokenizer(windows, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # BOS
        mask[torch.arange(mask.shape[0]), enc["attention_mask"].sum(1) - 1] = 0  # EOS
        mask = mask.unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu().numpy()

    def encode(self, sequences: list[str]) -> np.ndarray:
        """Return an ``(n, hidden)`` float32 matrix, one row per sequence."""
        cache_path = None
        if self.cache_dir:
            key = sequences_hash(self.model_name, sequences)
            cache_path = os.path.join(self.cache_dir, f"{key}.npy")
            if os.path.exists(cache_path):
                logger.info("Protein embeddings from cache: %s", cache_path)
                return np.load(cache_path)

        self._ensure_loaded()
        # Cut every protein into windows; remember (protein, residues) per window.
        windows, owner, weight = [], [], []
        for i, seq in enumerate(sequences):
            seq = seq.upper().rstrip("*") or "X"
            for start in range(0, len(seq), self.window):
                piece = seq[start : start + self.window]
                windows.append(piece)
                owner.append(i)
                weight.append(len(piece))

        order = np.argsort([len(w) for w in windows])[::-1]
        dim = self._model.config.hidden_size
        sums = np.zeros((len(sequences), dim), dtype=np.float64)
        total = np.zeros(len(sequences), dtype=np.float64)
        start = 0
        while start < len(order):
            longest = len(windows[order[start]])
            size = max(1, self.batch_tokens // (longest + 2))
            batch = order[start : start + size]
            emb = self._embed_windows([windows[k] for k in batch])
            for row, k in zip(emb, batch):
                sums[owner[k]] += weight[k] * row
                total[owner[k]] += weight[k]
            start += size

        out = (sums / total[:, None]).astype(np.float32)
        self.unload()
        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, out)
        return out

    def transform(self, split: Split) -> Split:
        """Replace the features with embeddings of ``split.sample_ids``."""
        if not split.sample_ids or not self.sequences:
            logger.info(
                "Protein encoder: no sequences for this split, "
                "passing through pre-extracted features"
            )
            return split
        seqs = [self.sequences[s] for s in split.sample_ids]
        return replace(split, features=self.encode(seqs))

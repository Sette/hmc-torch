"""Context and probability cache for TabFM conditional classifier.

Caches sampled contexts and predictions to avoid recomputation
across runs with the same seed and hyperparameters.
"""

from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np


class TabFMCache:
    """Disk-backed cache for TabFM contexts and predictions.

    Caches are keyed by a hash of the dataset identifier, node index,
    context sampler parameters, and random seed.

    Parameters
    ----------
    cache_dir : str
        Directory for cache files.
    enabled : bool
        If False, all operations are no-ops.
    """

    def __init__(self, cache_dir: str = "./.tabfm_cache", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _key(
        self,
        dataset_name: str,
        node_idx: int,
        k_pos: int,
        k_neg: int,
        seed: int,
    ) -> str:
        payload = f"{dataset_name}|{node_idx}|{k_pos}|{k_neg}|{seed}"
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def _path(self, key: str, suffix: str = ".pkl") -> str:
        return os.path.join(self.cache_dir, f"{key}{suffix}")

    def load_context(
        self,
        dataset_name: str,
        node_idx: int,
        k_pos: int,
        k_neg: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load cached context if available."""
        if not self.enabled:
            return None
        path = self._path(self._key(dataset_name, node_idx, k_pos, k_neg, seed))
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save_context(
        self,
        dataset_name: str,
        node_idx: int,
        k_pos: int,
        k_neg: int,
        seed: int,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
    ):
        """Save a context to disk."""
        if not self.enabled:
            return
        path = self._path(self._key(dataset_name, node_idx, k_pos, k_neg, seed))
        with open(path, "wb") as f:
            pickle.dump((X_ctx, y_ctx), f)

    def load_predictions(
        self,
        dataset_name: str,
        split: str,
        n_contexts: int,
    ) -> np.ndarray | None:
        """Load cached prediction matrix."""
        if not self.enabled:
            return None
        key = hashlib.md5(
            f"{dataset_name}|{split}|ctx{n_contexts}".encode()
        ).hexdigest()[:16]
        path = self._path(key, ".npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def save_predictions(
        self,
        dataset_name: str,
        split: str,
        n_contexts: int,
        scores: np.ndarray,
    ):
        """Save prediction matrix to disk."""
        if not self.enabled:
            return
        key = hashlib.md5(
            f"{dataset_name}|{split}|ctx{n_contexts}".encode()
        ).hexdigest()[:16]
        path = self._path(key, ".npy")
        np.save(path, scores)

    def clear(self):
        """Remove all cache files."""
        if not self.enabled:
            return
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))

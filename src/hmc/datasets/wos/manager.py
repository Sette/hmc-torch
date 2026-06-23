"""
WOS dataset manager — compatible with HMCDatasetManager interface.

Loads pre-split WOS JSONL files (HPT format), builds the label hierarchy from
slot.pt + value_dict.pt, computes transformer text embeddings, and exposes
train/val/test splits that the local and global pipelines consume without
modification.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from hmc.datasets.arxiv.manager import compute_transformer_embeddings
from hmc.datasets.wos.dataset_wos import WOSHierarchyManager, WOSSplit

logger = logging.getLogger(__name__)


class WOSManager:
    """HMCDatasetManager-compatible manager for WOS hierarchical text data.

    Text features are transformer embeddings extracted from document abstracts.
    Labels follow the two-level WOS taxonomy (e.g. CS → Machine learning).

    Exposed attributes (same as HMCDatasetManager):
        levels_size (dict): {level_idx: n_classes} for active training levels.
        max_depth (int): Number of active training levels.
        a (np.ndarray): Full hierarchy adjacency matrix (includes root).
        edge_index (dict): Per-level parent→child adjacency matrices.
        to_eval (list[bool]): Mask over all terms; False for root.
        nodes_idx (dict): term → global index (includes root).
        local_nodes_idx (dict): level → {term: local_index} (includes root).
        input_dim (int): Embedding dimension.
        output_dim (int): Total number of nodes in the hierarchy (incl. root).
        hierarchy_map (dict): Empty — used only by constrained models.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "allenai/specter2_base",
        cache_dir: Optional[str] = None,
        load_features: bool = True,
        model_cache_dir: str = "./models",
    ) -> None:
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self._data_dir = Path(data_dir)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._load_features = load_features
        self._fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self) -> Tuple[WOSSplit, WOSSplit, WOSSplit]:
        """Return (train, valid, test) WOSSplit objects."""
        return self._train, self._valid, self._test

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        hierarchy = self._build_hierarchy()
        self.hierarchy_manager = hierarchy

        # Load pre-split JSONL files
        train_texts, train_label_ids = self._load_records("WebOfScience_train.json")
        valid_texts, valid_label_ids = self._load_records("WebOfScience_dev.json")
        test_texts, test_label_ids = self._load_records("WebOfScience_test.json")

        all_texts = train_texts + valid_texts + test_texts
        n_train, n_valid = len(train_texts), len(valid_texts)

        X = (
            self._compute_features(all_texts)
            if self._load_features
            else np.empty((len(all_texts), 0), dtype=np.float32)
        )

        X_train = X[:n_train]
        X_valid = X[n_train : n_train + n_valid]
        X_test = X[n_train + n_valid :]

        Yg_train, Yl_train = self._compute_labels(hierarchy, train_label_ids)
        Yg_valid, Yl_valid = self._compute_labels(hierarchy, valid_label_ids)
        Yg_test, Yl_test = self._compute_labels(hierarchy, test_label_ids)

        self._train = WOSSplit(x=X_train, y=Yg_train, y_local=Yl_train)
        self._valid = WOSSplit(x=X_valid, y=Yg_valid, y_local=Yl_valid)
        self._test = WOSSplit(x=X_test, y=Yg_test, y_local=Yl_test)

        self._expose_hierarchy_attrs(hierarchy, X.shape[1], Yg_train.shape[1])

        logger.info(
            "WOS splits — train: %d  valid: %d  test: %d",
            n_train,
            n_valid,
            len(test_texts),
        )

    def _build_hierarchy(self) -> WOSHierarchyManager:
        slot_path = self._data_dir / "slot.pt"
        value_dict_path = self._data_dir / "value_dict.pt"
        if not slot_path.exists() or not value_dict_path.exists():
            raise FileNotFoundError(
                f"WOS hierarchy files not found at {self._data_dir}. "
                "Run download_wos.py first."
            )
        hierarchy = WOSHierarchyManager()
        hierarchy.load_hierarchy(str(slot_path), str(value_dict_path))
        logger.info(
            "WOS hierarchy: %d terms, levels %s",
            len(hierarchy.terms),
            dict(hierarchy.levels_size),
        )
        return hierarchy

    def _load_records(self, filename: str) -> Tuple[list, list]:
        filepath = self._data_dir / filename
        logger.info("Loading WOS records from %s …", filepath)
        texts, label_ids = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(rec["token"])
                label_ids.append(rec["label"])
        logger.info("Loaded %d records from %s.", len(texts), filename)
        return texts, label_ids

    def _cache_path(self, n_records: int) -> Path:
        """Deterministic cache path based on inputs that affect the feature matrix."""
        train_file = self._data_dir / "WebOfScience_train.json"
        stat = os.stat(str(train_file))
        key = "|".join(
            [
                str(train_file),
                str(stat.st_mtime),
                str(stat.st_size),
                self.model_name,
                str(n_records),
            ]
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_dir = (
            self._cache_dir
            if self._cache_dir
            else self._data_dir / ".feature_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{digest}.npy"

    def _compute_features(self, texts: list) -> np.ndarray:
        cache = self._cache_path(len(texts))
        if cache.exists():
            logger.info("Loading features from cache %s …", cache)
            return np.load(str(cache))

        X = compute_transformer_embeddings(
            texts,
            self.model_name,
            model_cache_dir=self.model_cache_dir,
        )

        np.save(str(cache), X)
        logger.info("Features cached to %s", cache)
        return X

    def _compute_labels(
        self, hierarchy: WOSHierarchyManager, label_id_lists: list
    ) -> Tuple[np.ndarray, list]:
        n = len(label_id_lists)
        total_terms = len(hierarchy.terms)
        Y_global = np.zeros((n, total_terms), dtype=np.float32)
        Y_local_all: list = []

        for i, label_ids in enumerate(label_id_lists):
            # Each WOS doc has exactly one parent-child pair.
            # label_ids[0] = parent_id, label_ids[1] = child_id.
            child_name = hierarchy.value_dict[label_ids[1]]
            y_global, y_local = hierarchy.get_labels(child_name)
            Y_global[i] = y_global
            # Skip root (level 0) so the local label list is 0-indexed from
            # the first *meaningful* level — matching the ARFF convention.
            Y_local_all.append(y_local[1:])

        return Y_global, Y_local_all

    def _expose_hierarchy_attrs(
        self,
        hierarchy: WOSHierarchyManager,
        actual_input_dim: int,
        total_terms: int,
    ) -> None:
        # Re-index levels to skip root (level 0) — same convention as ARFF.
        self.levels_size: dict = {
            k - 1: v for k, v in hierarchy.levels_size.items() if k > 0
        }
        self.max_depth: int = len(self.levels_size)

        # Full hierarchy attributes (needed by global pipeline and metrics).
        self.a: np.ndarray = hierarchy.a
        self.edge_index: dict = hierarchy.edge_index
        self.nodes_idx: dict = hierarchy.nodes_idx
        self.local_nodes_idx: dict = {
            k - 1: v for k, v in hierarchy.local_nodes_idx.items() if k > 0
        }
        self.to_eval: list = [term != "root" for term in hierarchy.terms]
        self.hierarchy_map: dict = {}

        self.input_dim: int = actual_input_dim
        self.output_dim: int = total_terms

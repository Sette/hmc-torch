"""
RCV1-V2 dataset manager — compatible with HMCDatasetManager interface.

Loads RCV1-V2 JSON files (HiAGM/HiMatch format or raw text), builds the
4-level topic hierarchy, computes SPECTER2 embeddings, and exposes
train/val/test splits.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from hmc.datasets.arxiv.manager import compute_transformer_embeddings
from hmc.datasets.rcv1.dataset_rcv1 import RCV1HierarchyManager, RCV1Split

logger = logging.getLogger(__name__)


class RCV1Manager:
    """HMCDatasetManager-compatible manager for RCV1-V2 hierarchical text data.

    Text features are SPECTER2 embeddings from news article text.
    Labels follow a 4-level topic hierarchy (CCAT/ECAT/GCAT/MCAT → subtopics).

    Supports two data formats:
      1. HiAGM/HiMatch JSON: ``{"token": "text", "label": ["CCAT/E21", ...]}``
      2. Raw text JSONL: ``{"text": "...", "labels": ["CCAT/E21", ...]}``

    Exposed attributes (same interface as ArXivManager):
        levels_size, max_depth, a, edge_index, to_eval, nodes_idx,
        local_nodes_idx, input_dim, output_dim, hierarchy_map
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "allenai/specter2_base",
        max_records: int | None = None,
        cache_dir: str | None = None,
        load_features: bool = True,
        model_cache_dir: str = "./models",
    ) -> None:
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self._data_dir = Path(data_dir)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._load_features = load_features
        self._max_records = max_records
        # Initialized in _fit()
        self.levels_size: dict = {}
        self.max_depth: int = 0
        self.a: np.ndarray = np.array([])
        self.edge_index: dict = {}
        self.nodes_idx: dict = {}
        self.local_nodes_idx: dict = {}
        self.to_eval: list = []
        self.hierarchy_map: dict = {}
        self.input_dim: int = 0
        self.output_dim: int = 0
        self._fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self) -> tuple[RCV1Split, RCV1Split, RCV1Split]:
        """Return (train, valid, test) RCV1Split objects."""
        return self._train, self._valid, self._test

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _discover_json_files(self) -> dict:
        """Find train/test JSON files in the data directory.

        Returns a dict with keys 'train', 'test', and optionally 'val'.
        """
        data_dir = self._data_dir
        files: dict = {}

        # HiAGM naming convention
        candidates = {
            "train": [
                data_dir / "rcv1_train.json",
                data_dir / "train.json",
                data_dir / "lyrl2004_tokens_train.json",
            ],
            "test": [
                data_dir / "rcv1_test.json",
                data_dir / "test.json",
                data_dir / "lyrl2004_tokens_test.json",
            ],
            "val": [
                data_dir / "rcv1_val.json",
                data_dir / "val.json",
                data_dir / "dev.json",
            ],
        }

        for split, paths in candidates.items():
            for p in paths:
                if p.exists():
                    files[split] = p
                    break

        if "train" not in files or "test" not in files:
            raise FileNotFoundError(
                f"RCV1 JSON files not found in {data_dir}. "
                f"Expected train.json and test.json (HiAGM format). "
                "Run download_rcv1.py first or place files manually."
            )

        logger.info(
            "Found RCV1 files: %s",
            {k: str(v.name) for k, v in files.items()},
        )
        return files

    def _load_json_records(self, filepath: Path) -> tuple[list, list]:
        """Load records from a HiAGM-format JSON file.

        Returns (texts, labels_list) where labels_list is a list of
        lists of topic path strings.
        """
        logger.info("Loading RCV1 records from %s …", filepath)

        texts, labels_list = [], []

        with open(filepath, "r", encoding="utf-8") as f:
            # HiAGM format: one JSON object per line
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # Array format
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        texts.append(self._extract_text(rec))
                        labels_list.append(self._extract_labels(rec))
            else:
                # JSONL format
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    texts.append(self._extract_text(rec))
                    labels_list.append(self._extract_labels(rec))

        logger.info("Loaded %d records from %s.", len(texts), filepath.name)
        return texts, labels_list

    @staticmethod
    def _extract_text(rec: dict) -> str:
        """Extract text from a record, handling multiple formats."""
        # HiAGM format uses 'token'
        text = rec.get("token", rec.get("text", rec.get("body", "")))
        if isinstance(text, list):
            text = " ".join(text)
        return text.strip()

    @staticmethod
    def _extract_labels(rec: dict) -> list:
        """Extract labels from a record.

        Returns a list of topic path strings.
        """
        labels = rec.get("label", rec.get("labels", []))
        if isinstance(labels, str):
            labels = labels.split()
        return list(labels)

    def _fit(self) -> None:
        files = self._discover_json_files()

        # Load all splits
        train_texts, train_labels = self._load_json_records(files["train"])
        test_texts, test_labels = self._load_json_records(files["test"])

        if "val" in files:
            val_texts, val_labels = self._load_json_records(files["val"])
        else:
            # Create val split from train (80/20)
            n_val = max(1, int(len(train_texts) * 0.2))
            val_texts = train_texts[-n_val:]
            val_labels = train_labels[-n_val:]
            train_texts = train_texts[:-n_val]
            train_labels = train_labels[:-n_val]
            logger.info(
                "No val file found. Split train→%d train / %d val.",
                len(train_texts),
                len(val_texts),
            )

        # Apply max_records cap
        if self._max_records:
            train_texts = train_texts[: self._max_records]
            train_labels = train_labels[: self._max_records]
            logger.info("Truncated train to %d records.", len(train_texts))

        # Build hierarchy from train labels only (to avoid test leakage)
        hierarchy = RCV1HierarchyManager.from_topic_paths(train_labels)
        self.hierarchy_manager = hierarchy

        # Compute embeddings
        all_texts = train_texts + val_texts + test_texts
        n_train, n_val = len(train_texts), len(val_texts)

        X = (
            self._compute_features(all_texts)
            if self._load_features
            else np.empty((len(all_texts), 0), dtype=np.float32)
        )

        X_train = X[:n_train]
        X_val = X[n_train : n_train + n_val]
        X_test = X[n_train + n_val :]

        Yg_train, Yl_train = self._compute_labels(hierarchy, train_labels)
        Yg_val, Yl_val = self._compute_labels(hierarchy, val_labels)
        Yg_test, Yl_test = self._compute_labels(hierarchy, test_labels)

        self._train = RCV1Split(x=X_train, y=Yg_train, y_local=Yl_train)
        self._valid = RCV1Split(x=X_val, y=Yg_val, y_local=Yl_val)
        self._test = RCV1Split(x=X_test, y=Yg_test, y_local=Yl_test)

        self._expose_hierarchy_attrs(hierarchy, X.shape[1], Yg_train.shape[1])

        logger.info(
            "RCV1 splits — train: %d  val: %d  test: %d",
            n_train,
            n_val,
            len(test_texts),
        )

    def _cache_path(self, n_records: int) -> Path:
        """Deterministic cache path."""
        files = self._discover_json_files()
        train_file = files["train"]
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
            self._cache_dir if self._cache_dir else self._data_dir / ".feature_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"rcv1_{digest}.npy"

    def _compute_features(self, texts: list) -> np.ndarray:
        """Compute SPECTER2 embeddings with MD5 cache."""
        cache = self._cache_path(len(texts))
        if cache.exists():
            logger.info("Loading RCV1 features from cache %s …", cache)
            return np.load(str(cache))

        X = compute_transformer_embeddings(
            texts,
            self.model_name,
            model_cache_dir=self.model_cache_dir,
        )

        np.save(str(cache), X)
        logger.info("RCV1 features cached to %s", cache)
        return X

    @staticmethod
    def _compute_labels(
        hierarchy: RCV1HierarchyManager,
        labels_list: list,
    ) -> tuple[np.ndarray, list]:
        """Convert topic paths into global + local binary matrices."""
        n = len(labels_list)
        total_terms = len(hierarchy.terms)
        Y_global = np.zeros((n, total_terms), dtype=np.float32)
        Y_local_all: list = []

        for i, topic_paths in enumerate(labels_list):
            y_global, y_local = hierarchy.get_labels(topic_paths)
            Y_global[i] = y_global
            Y_local_all.append(y_local[1:])  # skip root

        return Y_global, Y_local_all

    def _expose_hierarchy_attrs(
        self,
        hierarchy: RCV1HierarchyManager,
        actual_input_dim: int,
        total_terms: int,
    ) -> None:
        """Expose hierarchy attributes in ARFF-compatible format."""
        self.levels_size: dict = {
            k - 1: v for k, v in hierarchy.levels_size.items() if k > 0
        }
        self.max_depth: int = len(self.levels_size)

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

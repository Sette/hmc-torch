"""
AAPD dataset manager — compatible with HMCDatasetManager interface.

Loads AAPD CSV (title, abstract, labels), builds the 2-level label hierarchy,
computes SPECTER2 transformer embeddings, and exposes train/val/test splits
that the local and global pipelines consume without modification.
"""

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np

from hmc.datasets.aapd.dataset_aapd import AAPDHierarchyManager, AAPDSplit
from hmc.datasets.arxiv.manager import compute_transformer_embeddings

logger = logging.getLogger(__name__)


class AAPDManager:
    """HMCDatasetManager-compatible manager for AAPD hierarchical text data.

    Text features are SPECTER2 embeddings extracted from title + abstract.
    Labels follow a 2-level taxonomy: 6 top-level areas → 48 sub-fields.

    Exposed attributes (same interface as ArXivManager / WOSManager):
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
        self._fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self) -> tuple[AAPDSplit, AAPDSplit, AAPDSplit]:
        """Return (train, valid, test) AAPDSplit objects."""
        return self._train, self._valid, self._test

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _find_csv(self) -> Path:
        """Find the AAPD CSV file in the data directory."""
        candidates = [
            self._data_dir / "aapd.csv",
            self._data_dir / "AAPD.csv",
            self._data_dir / "arxiv_academic_paper_dataset.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No AAPD CSV found in {self._data_dir}. "
            f"Tried: {[str(c) for c in candidates]}. "
            "Run download_aapd.py first."
        )

    def _fit(self) -> None:
        csv_path = self._find_csv()
        texts, labels_list = self._load_records(csv_path)

        if self._max_records and len(texts) > self._max_records:
            texts = texts[: self._max_records]
            labels_list = labels_list[: self._max_records]
            logger.info("Truncated to %d records.", self._max_records)

        # Collect all unique labels to build the hierarchy
        all_labels: set = set()
        for lbl_str in labels_list:
            all_labels.update(lbl_str.split())

        hierarchy = AAPDHierarchyManager.from_labels(list(all_labels))
        self.hierarchy_manager = hierarchy

        X = (
            self._compute_features(texts)
            if self._load_features
            else np.empty((len(texts), 0), dtype=np.float32)
        )

        Y_global, Y_local_all = self._compute_labels(hierarchy, labels_list)
        self._create_splits(X, Y_global, Y_local_all)
        self._expose_hierarchy_attrs(hierarchy, X.shape[1], Y_global.shape[1])

    def _clean_text(self, text: str) -> str:
        """Remove XML/HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _load_records(self, csv_path: Path) -> tuple[list, list]:
        """Load AAPD CSV, returning (texts, labels_list).

        The CSV is expected to have columns: title, abstract, labels
        (with labels as space-separated sub-field codes).
        """
        import csv as _csv

        logger.info("Loading AAPD records from %s …", csv_path)
        texts, labels_list = [], []

        with open(csv_path, "r", encoding="utf-8") as f:
            # Detect delimiter: some versions use tab, others comma
            sample = f.read(4096)
            f.seek(0)
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = _csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                # Handle different column name conventions
                title = row.get("title", row.get("Title", ""))
                abstract = row.get("abstract", row.get("Abstract", ""))
                labels = row.get(
                    "labels",
                    row.get("Labels", row.get("label", "")),
                )

                title = self._clean_text(title)
                abstract = self._clean_text(abstract)

                # Some versions have labels as JSON list string
                if labels.startswith("["):
                    try:
                        import json as _json

                        parsed = _json.loads(labels)
                        labels = " ".join(parsed)
                    except (_json.JSONDecodeError, TypeError):
                        pass

                text = f"{title} {abstract}"
                texts.append(text)
                labels_list.append(labels)

        logger.info("Loaded %d records.", len(texts))
        return texts, labels_list

    def _cache_path(self, n_records: int) -> Path:
        """Deterministic cache path based on inputs."""
        csv_path = self._find_csv()
        stat = os.stat(str(csv_path))
        key = "|".join(
            [
                str(csv_path),
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
        return cache_dir / f"aapd_{digest}.npy"

    def _compute_features(self, texts: list) -> np.ndarray:
        """Compute SPECTER2 embeddings with MD5 cache."""
        cache = self._cache_path(len(texts))
        if cache.exists():
            logger.info("Loading AAPD features from cache %s …", cache)
            return np.load(str(cache))

        X = compute_transformer_embeddings(
            texts,
            self.model_name,
            model_cache_dir=self.model_cache_dir,
        )

        np.save(str(cache), X)
        logger.info("AAPD features cached to %s", cache)
        return X

    def _compute_labels(
        self,
        hierarchy: AAPDHierarchyManager,
        labels_list: list,
    ) -> tuple[np.ndarray, list]:
        """Convert text labels into global + local binary matrices."""
        n = len(labels_list)
        total_terms = len(hierarchy.terms)
        Y_global = np.zeros((n, total_terms), dtype=np.float32)
        Y_local_all: list = []

        for i, label_str in enumerate(labels_list):
            y_global, y_local = hierarchy.get_labels(label_str)
            Y_global[i] = y_global
            Y_local_all.append(y_local[1:])  # skip root

        return Y_global, Y_local_all

    def _create_splits(
        self,
        X: np.ndarray,
        Y_global: np.ndarray,
        Y_local_all: list,
    ) -> None:
        """Create 64/16/20 splits using HPT methodology."""
        from sklearn.model_selection import (
            train_test_split,
        )

        np.random.seed(7)
        n = len(X)
        idx = list(range(n))
        np.random.shuffle(idx)

        X_shuf = X[idx]
        Yg_shuf = Y_global[idx]
        Yl_shuf = [Y_local_all[i] for i in idx]

        X_train, X_test, Yg_train, Yg_test, Yl_train, Yl_test = train_test_split(
            X_shuf, Yg_shuf, Yl_shuf, test_size=0.2, random_state=0
        )
        X_train, X_val, Yg_train, Yg_val, Yl_train, Yl_val = train_test_split(
            X_train, Yg_train, Yl_train, test_size=0.2, random_state=0
        )

        self._train = AAPDSplit(x=X_train, y=Yg_train, y_local=Yl_train)
        self._valid = AAPDSplit(x=X_val, y=Yg_val, y_local=Yl_val)
        self._test = AAPDSplit(x=X_test, y=Yg_test, y_local=Yl_test)
        logger.info(
            "Splits — train: %d  valid: %d  test: %d (HPT methodology)",
            len(self._train.x),
            len(self._valid.x),
            len(self._test.x),
        )

    def _expose_hierarchy_attrs(
        self,
        hierarchy: AAPDHierarchyManager,
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

"""
EUR-Lex 57K dataset manager — compatible with HMCDatasetManager interface.

Loads EUR-Lex JSON files (train/dev/test), builds the EUROVOC hierarchy
(~4K labels, up to 8 levels deep), computes SPECTER2 embeddings, and
exposes train/val/test splits.

Because the label space is large (~4K concepts), the dense R-matrix
would exceed GPU memory.  This manager is designed to work with
**sparse R** (BlockDiagonalR) at inference time — train with identity
R-matrix, then reconcile via graph traversal during evaluation.

Reference: Chalkidis et al. (ACL 2019), "Large-Scale Multi-Label Text
Classification on EU Legislation".
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import networkx as nx
import numpy as np

from hmc.datasets.arxiv.manager import compute_transformer_embeddings
from hmc.datasets.eurlex.dataset_eurlex import EURLexHierarchyManager, EURLexSplit

logger = logging.getLogger(__name__)

# Known paths for the EUROVOC concept file
_CONCEPT_CANDIDATES = [
    "eurovoc_concepts.jsonl",
    "concepts.jsonl",
    "eurovoc_concepts.json",
    "hierarchy.json",
]


class EURLexManager:
    """Manager for EUR-Lex 57K hierarchical text classification.

    Text features are SPECTER2 embeddings from document text (header +
    recitals + main body). Labels are EUROVOC concept IDs (~4,271 in total).

    **Sparse R-matrix:**  Because the label space is large, the dense
    R-matrix is not built during training.  Instead, train with an
    identity matrix and use ``BlockDiagonalR`` at inference time::

        from hmc.models.hierarchical.sparse_r import BlockDiagonalR
        sparse_r = BlockDiagonalR(test_split.g)
        reconciled = sparse_r.reconcile(raw_scores)

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
        self._fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self) -> tuple[EURLexSplit, EURLexSplit, EURLexSplit]:
        """Return (train, valid, test) EURLexSplit objects."""
        return self._train, self._valid, self._test

    @property
    def use_sparse_r(self) -> bool:
        """True when the label space is large enough to warrant sparse R."""
        return self.output_dim > 1000

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _discover_json_files(self) -> dict:
        """Find train/dev/test JSON files in the data directory."""
        data_dir = self._data_dir
        files: dict = {}

        candidates = {
            "train": [
                data_dir / "train.jsonl",
                data_dir / "train.json",
                data_dir / "EURLEX57K_train.json",
            ],
            "dev": [
                data_dir / "dev.jsonl",
                data_dir / "dev.json",
                data_dir / "EURLEX57K_dev.json",
                data_dir / "val.json",
            ],
            "test": [
                data_dir / "test.jsonl",
                data_dir / "test.json",
                data_dir / "EURLEX57K_test.json",
            ],
        }

        for split, paths in candidates.items():
            for p in paths:
                if p.exists():
                    files[split] = p
                    break

        if "train" not in files:
            raise FileNotFoundError(
                f"EUR-Lex JSON files not found in {data_dir}. "
                "Run download_eurlex.py first."
            )

        # If no dev file, will split from train
        logger.info(
            "Found EUR-Lex files: %s",
            {k: str(v.name) for k, v in files.items()},
        )
        return files

    def _find_concept_file(self) -> Path | None:
        """Look for EUROVOC concept hierarchy file."""
        for name in _CONCEPT_CANDIDATES:
            path = self._data_dir / name
            if path.exists():
                return path
        return None

    def _load_json_records(self, filepath: Path) -> tuple[list, list]:
        """Load records from a JSON file.

        Returns (texts, labels_list) where labels_list is a list of
        lists of EUROVOC concept ID strings.
        """
        logger.info("Loading EUR-Lex records from %s …", filepath)

        texts, labels_list = [], []

        with open(filepath, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        texts.append(self._extract_text(rec))
                        labels_list.append(self._extract_labels(rec))
            else:
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
        """Extract document text — title + header + recitals + main_body."""
        title = rec.get("title", "")

        def _join(val):
            """Handle both string and list-of-strings fields."""
            if isinstance(val, list):
                return " ".join(val)
            return str(val) if val else ""

        parts = [
            _join(rec.get("header", "")),
            _join(rec.get("recitals", "")),
            _join(rec.get("main_body", "")),
        ]
        body = " ".join(p for p in parts if p)
        text = f"{title} {body}".strip()
        if not text:
            # Fallback: single 'text' field
            text = _join(rec.get("text", rec.get("body", rec.get("content", ""))))
        if not text:
            # HiAGM-style 'token' field
            token = rec.get("token", "")
            if isinstance(token, list):
                token = " ".join(token)
            text = str(token) if token else ""
        return text

    @staticmethod
    def _extract_labels(rec: dict) -> list:
        """Extract EUROVOC concept IDs."""
        labels = rec.get(
            "eurovoc_concepts",
            rec.get("labels", rec.get("label", [])),
        )
        if isinstance(labels, str):
            labels = labels.split()
        return [str(label) for label in labels]

    def _fit(self) -> None:
        files = self._discover_json_files()

        # Load all splits
        train_texts, train_labels = self._load_json_records(files["train"])
        test_texts, test_labels = self._load_json_records(files["test"])

        if "dev" in files:
            val_texts, val_labels = self._load_json_records(files["dev"])
        else:
            n_val = max(1, int(len(train_texts) * 0.12))
            val_texts = train_texts[-n_val:]
            val_labels = train_labels[-n_val:]
            train_texts = train_texts[:-n_val]
            train_labels = train_labels[:-n_val]
            logger.info(
                "No dev file found. Split train→%d train / %d val.",
                len(train_texts),
                len(val_texts),
            )

        # Apply max_records cap
        if self._max_records:
            train_texts = train_texts[: self._max_records]
            train_labels = train_labels[: self._max_records]
            logger.info("Truncated to %d records.", len(train_texts))

        # Build hierarchy
        concept_file = self._find_concept_file()
        observed_labels: set = set()
        for lbls in train_labels:
            observed_labels.update(lbls)

        if concept_file:
            hierarchy = EURLexHierarchyManager.from_concept_file(
                str(concept_file), observed_labels
            )
        else:
            logger.warning(
                "No EUROVOC concept file found. Using flat hierarchy. "
                "Download eurovoc_concepts.jsonl for full hierarchy."
            )
            hierarchy = EURLexHierarchyManager.from_labels(train_labels)

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

        # Build a DiGraph for sparse R reconciliation
        train_g = self._build_label_digraph(hierarchy)

        labels = self._compute_all_split_labels(
            hierarchy, train_labels, val_labels, test_labels
        )
        self._train = EURLexSplit(x=X_train, y=labels[0], y_local=labels[1], g=train_g)
        self._valid = EURLexSplit(x=X_val, y=labels[2], y_local=labels[3], g=train_g)
        self._test = EURLexSplit(x=X_test, y=labels[4], y_local=labels[5], g=train_g)

        self._expose_hierarchy_attrs(hierarchy, X.shape[1], labels[0].shape[1])

        logger.info(
            "EUR-Lex splits — train: %d  val: %d  test: %d  nodes: %d",
            n_train,
            n_val,
            len(test_texts),
            self.output_dim,
        )
        if self.use_sparse_r:
            logger.info(
                "Label space is large (%d nodes). Use BlockDiagonalR "
                "for inference-time reconciliation.",
                self.output_dim,
            )

    @staticmethod
    def _compute_all_split_labels(
        hierarchy,
        train_labels: list,
        val_labels: list,
        test_labels: list,
    ):
        """Compute global and local label matrices for all three splits."""
        Yg_train, Yl_train = EURLexManager._compute_labels(hierarchy, train_labels)
        Yg_val, Yl_val = EURLexManager._compute_labels(hierarchy, val_labels)
        Yg_test, Yl_test = EURLexManager._compute_labels(hierarchy, test_labels)
        return Yg_train, Yl_train, Yg_val, Yl_val, Yg_test, Yl_test

    @staticmethod
    def _build_label_digraph(
        hierarchy: EURLexHierarchyManager,
    ) -> nx.DiGraph:
        """Build a simple DiGraph for sparse R reconciliation.

        The graph has edges child→parent (same as hierarchy.g).
        BlockDiagonalR uses this to propagate scores upward.
        """
        g = nx.DiGraph()
        for node in hierarchy.terms:
            if node == "root":
                continue
            g.add_node(node)
            try:
                for parent in hierarchy.g.successors(node):
                    g.add_edge(node, parent)
            except nx.NetworkXError:
                pass
        return g

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
        return cache_dir / f"eurlex_{digest}.npy"

    def _compute_features(self, texts: list) -> np.ndarray:
        """Compute SPECTER2 embeddings with MD5 cache."""
        cache = self._cache_path(len(texts))
        if cache.exists():
            logger.info("Loading EUR-Lex features from cache %s …", cache)
            return np.load(str(cache))

        X = compute_transformer_embeddings(
            texts,
            self.model_name,
            model_cache_dir=self.model_cache_dir,
        )

        np.save(str(cache), X)
        logger.info("EUR-Lex features cached to %s", cache)
        return X

    @staticmethod
    def _compute_labels(
        hierarchy: EURLexHierarchyManager,
        labels_list: list,
    ) -> tuple[np.ndarray, list]:
        """Convert concept ID lists into global + local binary matrices."""
        n = len(labels_list)
        total_terms = len(hierarchy.terms)
        Y_global = np.zeros((n, total_terms), dtype=np.float32)
        Y_local_all: list = []

        for i, concept_ids in enumerate(labels_list):
            y_global, y_local = hierarchy.get_labels(concept_ids)
            Y_global[i] = y_global
            Y_local_all.append(y_local[1:])  # skip root

        return Y_global, Y_local_all

    def _expose_hierarchy_attrs(
        self,
        hierarchy: EURLexHierarchyManager,
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

"""
ArXiv dataset manager — compatible with HMCDatasetManager interface.

Loads ArXiv JSONL, builds the label hierarchy from the loaded records,
computes TF-IDF + TruncatedSVD text features, and exposes train/val/test
splits as ArXivSplit objects that the local and global pipelines consume
without modification.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from hmc.datasets.arxiv.dataset_arxiv import ArXivHierarchyManager, ArXivSplit

logger = logging.getLogger(__name__)


class ArXivManager:
    """HMCDatasetManager-compatible manager for ArXiv hierarchical text data.

    Text features are computed with TF-IDF (max 50 K terms) followed by
    TruncatedSVD to a fixed ``n_components``-dimensional dense vector.
    Labels follow the two-level ArXiv taxonomy (e.g. cs → cs.AI); the root
    pseudo-node is excluded from training targets so the interface matches
    the ARFF datasets where level-0 is the first *meaningful* hierarchy level.

    Exposed attributes (same as HMCDatasetManager):
        levels_size (dict): {level_idx: n_classes} for active training levels.
        max_depth (int): Number of active training levels.
        a (np.ndarray): Full hierarchy adjacency matrix (includes root).
        edge_index (dict): Per-level parent→child adjacency matrices.
        to_eval (list[bool]): Mask over all terms; False for root.
        nodes_idx (dict): term → global index (includes root).
        local_nodes_idx (dict): level → {term: local_index} (includes root).
        input_dim (int): Actual SVD output dimension.
        output_dim (int): Total number of nodes in the hierarchy (incl. root).
        hierarchy_map (dict): Empty — used only by constrained models.
    """

    def __init__(
        self,
        jsonl_path: str,
        n_components: int = 256,
        max_records: Optional[int] = 50_000,
        category_prefix: Optional[str] = None,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        seed: int = 42,
        feature_type: str = "tfidf",
        model_name: str = "allenai/specter2_base",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.n_components = n_components
        self.feature_type = feature_type
        self.model_name = model_name
        self._jsonl_path = jsonl_path
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._fit(
            jsonl_path, max_records, category_prefix, train_ratio, valid_ratio, seed
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self) -> Tuple[ArXivSplit, ArXivSplit, ArXivSplit]:
        """Return (train, valid, test) ArXivSplit objects."""
        return self._train, self._valid, self._test

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _fit(
        self,
        jsonl_path: str,
        max_records: Optional[int],
        category_prefix: Optional[str],
        train_ratio: float,
        valid_ratio: float,
        seed: int,
    ) -> None:
        texts, cats_list = self._load_records(jsonl_path, max_records, category_prefix)

        hierarchy = self._build_hierarchy(cats_list)
        self.hierarchy_manager = hierarchy  # exposed for E2E pipeline
        X = self._compute_features(texts, seed)
        Y_global, Y_local_all = self._compute_labels(hierarchy, cats_list)

        self._create_splits(X, Y_global, Y_local_all, train_ratio, valid_ratio, seed)
        self._expose_hierarchy_attrs(hierarchy, X.shape[1], Y_global.shape[1])

    def _load_records(
        self,
        jsonl_path: str,
        max_records: Optional[int],
        category_prefix: Optional[str],
    ) -> Tuple[list, list]:
        logger.info("Loading ArXiv records from %s (max=%s) …", jsonl_path, max_records)
        texts, cats_list = [], []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_records and len(texts) >= max_records:
                    break
                rec = json.loads(line)
                cats = rec.get("categories", "")
                if category_prefix:
                    tokens = cats.split()
                    if not any(c.startswith(category_prefix) for c in tokens):
                        continue
                title = rec.get("title", "").replace("\n", " ").strip()
                abstract = rec.get("abstract", "").replace("\n", " ").strip()
                texts.append(f"{title} {abstract}")
                cats_list.append(cats)
        logger.info("Loaded %d records.", len(texts))
        return texts, cats_list

    def _build_hierarchy(self, cats_list: list) -> ArXivHierarchyManager:
        unique_cats: set = set()
        for cats in cats_list:
            unique_cats.update(cats.split())

        hierarchy = ArXivHierarchyManager()
        hierarchy._build_graphs(unique_cats)  # pylint: disable=protected-access
        hierarchy._build_edge_index()  # pylint: disable=protected-access
        hierarchy.a = nx.to_numpy_array(
            hierarchy.g, nodelist=hierarchy.terms, dtype=np.float32
        )
        hierarchy._is_fitted = True  # pylint: disable=protected-access
        logger.info(
            "Hierarchy: %d terms, levels %s",
            len(hierarchy.terms),
            dict(hierarchy.levels_size),
        )
        return hierarchy

    def _cache_path(self, n_records: int) -> Path:
        """Deterministic cache path based on all inputs that affect the feature matrix."""
        stat = os.stat(self._jsonl_path)
        key = "|".join(
            [
                self._jsonl_path,
                str(stat.st_mtime),
                str(stat.st_size),
                self.feature_type,
                self.model_name,
                str(self.n_components),
                str(n_records),
            ]
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_dir = (
            self._cache_dir
            if self._cache_dir
            else Path(self._jsonl_path).parent / ".feature_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{digest}.npy"

    def _compute_features(self, texts: list, seed: int) -> np.ndarray:
        cache = self._cache_path(len(texts))
        if cache.exists():
            logger.info("Loading features from cache %s …", cache)
            return np.load(str(cache))

        if self.feature_type == "embedding":
            X = self._compute_features_embedding(texts)
        else:
            X = self._compute_features_tfidf(texts, seed)

        np.save(str(cache), X)
        logger.info("Features cached to %s", cache)
        return X

    def _compute_features_tfidf(self, texts: list, seed: int) -> np.ndarray:
        logger.info("Computing TF-IDF features …")
        tfidf = TfidfVectorizer(max_features=50_000, sublinear_tf=True, min_df=2)
        X_tfidf = tfidf.fit_transform(texts)

        n_comp = min(self.n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        logger.info("TruncatedSVD with %d components …", n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X = svd.fit_transform(X_tfidf).astype(np.float32)
        logger.info("Feature matrix shape: %s", X.shape)
        return X

    # Models that use CLS-token pooling instead of mean-pool.
    # SPECTER2 was trained with a contrastive objective where [CLS] carries the
    # document-level representation; mean-pool degrades its retrieval quality.
    _CLS_POOL_MODELS = ("specter",)

    def _use_cls_pooling(self) -> bool:
        return any(k in self.model_name.lower() for k in self._CLS_POOL_MODELS)

    def _compute_features_embedding(self, texts: list) -> np.ndarray:
        """Extract transformer embeddings using CLS-pool (SPECTER*) or mean-pool."""
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModel,
            AutoTokenizer,
        )

        cls_pool = self._use_cls_pooling()
        pool_mode = "CLS" if cls_pool else "mean"
        logger.info(
            "Loading %s for embeddings (pooling=%s) …", self.model_name, pool_mode
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        batch_size = 64
        all_embeddings: list = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            if cls_pool:
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(
                    1
                ).clamp(min=1e-9)
            all_embeddings.append(embeddings.cpu().numpy())

        X = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        logger.info("Embedding matrix shape: %s", X.shape)
        return X

    def _compute_labels(
        self, hierarchy: ArXivHierarchyManager, cats_list: list
    ) -> Tuple[np.ndarray, list]:
        n = len(cats_list)
        total_terms = len(hierarchy.terms)
        Y_global = np.zeros((n, total_terms), dtype=np.float32)
        Y_local_all: list = []

        for i, cats in enumerate(cats_list):
            y_global, y_local = hierarchy.get_labels(cats)
            Y_global[i] = y_global
            # Skip root (level 0) so the local label list is 0-indexed from
            # the first *meaningful* level — matching the ARFF convention.
            Y_local_all.append(y_local[1:])

        return Y_global, Y_local_all

    def _create_splits(
        self,
        X: np.ndarray,
        Y_global: np.ndarray,
        Y_local_all: list,
        train_ratio: float,
        valid_ratio: float,
        seed: int,
    ) -> None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        train_end = int(train_ratio * len(X))
        valid_end = train_end + int(valid_ratio * len(X))

        def _make(indices):
            return ArXivSplit(
                x=X[indices],
                y=Y_global[indices],
                y_local=[Y_local_all[i] for i in indices],
            )

        self._train = _make(idx[:train_end])
        self._valid = _make(idx[train_end:valid_end])
        self._test = _make(idx[valid_end:])
        logger.info(
            "Splits — train: %d  valid: %d  test: %d",
            len(self._train.x),
            len(self._valid.x),
            len(self._test.x),
        )

    def _expose_hierarchy_attrs(
        self,
        hierarchy: ArXivHierarchyManager,
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
        # Re-index local_nodes_idx to skip root (level 0) — same convention as
        # levels_size so that local evaluation functions see matching indices.
        self.local_nodes_idx: dict = {
            k - 1: v for k, v in hierarchy.local_nodes_idx.items() if k > 0
        }
        self.to_eval: list = [term != "root" for term in hierarchy.terms]
        self.hierarchy_map: dict = {}

        self.input_dim: int = actual_input_dim
        self.output_dim: int = total_terms

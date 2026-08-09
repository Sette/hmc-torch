"""
Module providing the complete PyTorch Dataset and Hierarchy Manager for the ArXiv dataset.
Extracts global and local labels, builds the Directed Acyclic Graph (DAG), and prepares
tensors compatible with the hmc-torch package.
Complies with Clean Architecture and Pylint standards.
"""

import json
import logging
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class _SamplesHolder:
    """Mutable holder so create_dataloader can attach tensor views."""

    x: Any | None = None
    y: Any | None = None


class ArXivSplit:
    """ARFF-compatible data split for one ArXiv partition.

    Exposes the same ``.x``, ``.y``, ``.y_local`` and ``.samples`` interface
    as ``HMCDatasetArff`` so the local and global pipelines can treat arxiv
    splits identically to ARFF splits.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_local: list[list[np.ndarray]],
    ) -> None:
        self.x = x  # (N, feat_dim) float32
        self.y = y  # (N, total_labels) float32  — global binary labels
        self.y_local = y_local  # list[list[ndarray]] — per-sample, per-level
        self.samples = _SamplesHolder()


class ArXivHierarchyManager:  # pylint: disable=too-many-instance-attributes
    """
    Manages the parsing of ArXiv categories, builds the class hierarchy graph,
    and exposes adjacency structures required for HMC architectures.
    """

    def __init__(self) -> None:
        """Initialise empty structures for the hierarchy graph."""
        self.g = nx.DiGraph()
        self.g_t = nx.DiGraph()
        self.levels: dict[int, list[str]] = defaultdict(list)
        self.levels_size: dict[int, int] = {}
        self.nodes_idx: dict[str, int] = {}
        self.local_nodes_idx: dict[int, dict[str, int]] = {}
        self.max_depth: int = 0
        self.terms: list[str] = []
        self.edge_index: dict[int, np.ndarray] = {}
        self.a: np.ndarray = np.array([])
        self._is_fitted: bool = False

    def fit_from_jsonl(self, jsonl_path: str) -> None:
        """
        Scans the downloaded ArXiv JSONL file to build the complete taxonomy.

        Args:
            jsonl_path (str): Path to the saved '.jsonl' dataset.
        """
        logger.info("Scanning %s to build the hierarchy...", jsonl_path)
        unique_categories: set[str] = set()

        # Extract all unique categories
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                cats = record.get("categories", "").split()
                unique_categories.update(cats)

        self._build_graphs(unique_categories)
        self._build_edge_index()
        self.a = nx.to_numpy_array(self.g, nodelist=self.terms, dtype=np.float32)
        self._is_fitted = True
        logger.info("Hierarchy built. Total unique nodes: %d", len(self.terms))

    def _build_graphs(self, unique_categories: set[str]) -> None:
        """
        Constructs the DiGraph and assigns depths based on ArXiv logic (e.g., cs -> cs.AI).
        """
        self.g.add_node("root")
        self.levels[0].append("root")

        for cat in unique_categories:
            parts = cat.split(".")

            # Level 1 (e.g., 'cs')
            level_1 = parts[0]
            if not self.g.has_node(level_1):
                self.g.add_edge(level_1, "root")

            # Level 2 (e.g., 'cs.AI')
            if len(parts) > 1:
                self.g.add_edge(cat, level_1)

        # Sort and map nodes
        self.g_t = self.g.reverse()
        self.terms = sorted(
            self.g.nodes(),
            key=lambda x: (nx.shortest_path_length(self.g, x, "root"), x),
        )
        self.nodes_idx = {node: idx for idx, node in enumerate(self.terms)}

        # Group by level and build local mappings
        for label in self.terms:
            if label != "root":
                depth = nx.shortest_path_length(self.g_t, "root", label)
                if label not in self.levels[depth]:
                    self.levels[depth].append(label)

        self.levels_size = {k: len(v) for k, v in self.levels.items()}
        self.max_depth = max(self.levels_size.keys())
        self.local_nodes_idx = {
            depth: {node: i for i, node in enumerate(nodes)}
            for depth, nodes in self.levels.items()
        }

    def _build_edge_index(self) -> None:
        """Builds parent->child adjacency matrices for each transition level."""
        for depth in range(1, self.max_depth + 1):
            prev_nodes = self.levels[depth - 1]
            curr_nodes = self.levels[depth]

            matrix = np.zeros((len(prev_nodes), len(curr_nodes)), dtype=np.float32)

            parent_map = {node: i for i, node in enumerate(prev_nodes)}
            child_map = {node: i for i, node in enumerate(curr_nodes)}

            for c_node in curr_nodes:
                for p_node in self.g.successors(c_node):
                    if p_node in parent_map:
                        matrix[parent_map[p_node], child_map[c_node]] = 1.0

            self.edge_index[depth] = matrix

    def get_labels(self, categories_string: str) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Converts the raw category string into global and local tensors.

        Args:
            categories_string (str): Space-separated categories (e.g., 'cs.AI stat.ML').

        Returns:
            Tuple:
                - y_global: Binary vector of length `len(self.terms)`
                - y_local: List of binary vectors, one per depth level.
        """
        y_global = np.zeros(len(self.terms), dtype=np.float32)
        y_local = [
            np.zeros(size, dtype=np.float32)
            for _, size in sorted(self.levels_size.items())
        ]

        # Root is always active
        y_global[self.nodes_idx["root"]] = 1.0
        y_local[0][self.local_nodes_idx[0]["root"]] = 1.0

        for cat in categories_string.split():
            # Activate current node and its ancestors
            if cat in self.nodes_idx:
                for ancestor in nx.ancestors(self.g_t, cat):
                    y_global[self.nodes_idx[ancestor]] = 1.0
                    depth = nx.shortest_path_length(self.g_t, "root", ancestor)
                    y_local[depth][self.local_nodes_idx[depth][ancestor]] = 1.0

                # Activate the leaf node itself
                y_global[self.nodes_idx[cat]] = 1.0
                depth = nx.shortest_path_length(self.g_t, "root", cat)
                y_local[depth][self.local_nodes_idx[depth][cat]] = 1.0

        return y_global, y_local


class ArXivPyTorchDataset(Dataset):
    """
    PyTorch Dataset for ArXiv text data and hierarchical labels.
    """

    def __init__(
        self,
        jsonl_path: str,
        hierarchy_manager: ArXivHierarchyManager,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_records: int | None = None,
    ):
        """
        Initialise the dataset, loading the JSONL into memory (suitable for abstracts).
        """
        self.hierarchy = hierarchy_manager
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict[str, str]] = []

        logger.info("Loading records from %s into memory...", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if max_records and len(self.records) >= max_records:
                    break
                record = json.loads(line)
                self.records.append(
                    {
                        "text": f"{record.get('title', '')} {record.get('abstract', '')}".strip(),
                        "categories": record.get("categories", ""),
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        record = self.records[idx]

        # Tokenize text features
        encoded_inputs = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_inputs = {key: val.squeeze(0) for key, val in encoded_inputs.items()}

        # Generate target structures
        y_global, y_local = self.hierarchy.get_labels(record["categories"])

        targets = {
            "global": torch.from_numpy(y_global),
            "local": [torch.from_numpy(yl) for yl in y_local],
        }

        return encoded_inputs, targets

    # ==========================================
    # Propriedades de Compatibilidade para o `args`
    # ==========================================

    @property
    def levels_size(self) -> dict:
        """dict: Number of unique nodes at each depth level."""
        return self.hierarchy.levels_size

    @property
    def max_depth(self) -> int:
        """int: Total number of depth levels in the hierarchy."""
        return self.hierarchy.max_depth

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """torch.Tensor: Returns the full graph adjacency matrix."""
        return torch.from_numpy(self.hierarchy.a).float()

    def get_datasets(
        self, train_ratio: float = 0.8, valid_ratio: float = 0.1, seed: int = 42
    ) -> tuple[Subset, Subset, Subset]:
        """Split into train/val/test using the same numpy permutation as ArXivManager.

        Keeping split logic identical ensures the test set is the same whether
        features are pre-computed (ArXivManager) or tokenised on-the-fly (E2E).
        """
        rng = np.random.RandomState(seed)  # pylint: disable=no-member
        idx = rng.permutation(len(self))
        train_end = int(train_ratio * len(self))
        valid_end = train_end + int(valid_ratio * len(self))

        train_dataset = Subset(self, idx[:train_end].tolist())
        valid_dataset = Subset(self, idx[train_end:valid_end].tolist())
        test_dataset = Subset(self, idx[valid_end:].tolist())

        return train_dataset, valid_dataset, test_dataset

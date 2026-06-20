"""
WOS (Web of Science) dataset structures — compatible with hmc-torch pipelines.

The WOS dataset is a standard HTC benchmark with a 2-level hierarchy:
  - Root
    - 7 parent categories: CS, Medical, Civil, ECE, biochemistry, MAE, Psychology
    - ~134 child categories (subcategories of the 7 parents)

Hierarchy is loaded from HPT-format files (slot.pt + value_dict.pt).
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)


class _SamplesHolder:
    """Mutable holder so the pipeline can attach tensor views."""

    x: Optional[Any] = None
    y: Optional[Any] = None


class WOSSplit:
    """ARFF-compatible data split for one WOS partition.

    Exposes the same ``.x``, ``.y``, ``.y_local`` and ``.samples`` interface
    as ``ArXivSplit`` and ``HMCDatasetArff`` so the pipelines can treat WOS
    splits identically to other dataset splits.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_local: List[List[np.ndarray]],
    ) -> None:
        self.x = x  # (N, feat_dim) float32
        self.y = y  # (N, total_labels) float32 — global binary labels
        self.y_local = y_local  # list[list[ndarray]] — per-sample, per-level
        self.samples = _SamplesHolder()


class WOSHierarchyManager:
    """Manages the WOS label hierarchy from HPT-format slot.pt + value_dict.pt.

    The WOS taxonomy is a strict two-level tree:
        root → parent (e.g. CS) → child (e.g. Machine learning)

    Exposes the same interface as ``ArXivHierarchyManager`` for label encoding
    and hierarchy attribute access.
    """

    def __init__(self) -> None:
        self.g = nx.DiGraph()
        self.g_t = nx.DiGraph()
        self.levels: Dict[int, List[str]] = defaultdict(list)
        self.levels_size: Dict[int, int] = {}
        self.nodes_idx: Dict[str, int] = {}
        self.local_nodes_idx: Dict[int, Dict[str, int]] = {}
        self.max_depth: int = 0
        self.terms: List[str] = []
        self.edge_index: Dict[int, np.ndarray] = {}
        self.a: np.ndarray = np.array([])
        self.slot: dict = {}  # parent_id → set(child_ids)
        self.value_dict: dict = {}  # id → label_name
        self._is_fitted: bool = False

    def load_hierarchy(self, slot_path: str, value_dict_path: str) -> None:
        """Load HPT-format slot.pt and value_dict.pt and build the graph.

        Args:
            slot_path: Path to slot.pt (parent_id → set of child_ids).
            value_dict_path: Path to value_dict.pt (id → label_name).
        """
        self.slot = torch.load(slot_path, map_location="cpu", weights_only=False)
        self.value_dict = torch.load(
            value_dict_path, map_location="cpu", weights_only=False
        )

        # Convert slot values to sets of ints (torch.save may store tensors)
        slot: Dict[int, set] = {}
        for parent_id, child_ids in self.slot.items():
            if isinstance(parent_id, torch.Tensor):
                parent_id = parent_id.item()
            slot[parent_id] = set()
            for cid in child_ids:
                if isinstance(cid, torch.Tensor):
                    cid = cid.item()
                slot[parent_id].add(cid)

        self._build_graphs(slot)
        self._build_edge_index()
        self.a = nx.to_numpy_array(self.g, nodelist=self.terms, dtype=np.float32)
        self._is_fitted = True
        logger.info(
            "WOS hierarchy built: %d terms, levels %s",
            len(self.terms),
            dict(self.levels_size),
        )

    def _build_graphs(self, slot: Dict[int, set]) -> None:
        """Build DiGraph from HPT slot dict."""
        self.g.add_node("root")
        self.levels[0].append("root")

        for parent_id, child_ids in slot.items():
            parent_name = self.value_dict[parent_id]
            if not self.g.has_node(parent_name):
                self.g.add_edge(parent_name, "root")

            for child_id in child_ids:
                child_name = self.value_dict[child_id]
                if not self.g.has_node(child_name):
                    self.g.add_edge(child_name, parent_name)

        # Sort terms by depth then name
        self.g_t = self.g.reverse()
        self.terms = sorted(
            self.g.nodes(),
            key=lambda x: (nx.shortest_path_length(self.g, x, "root"), x),
        )
        self.nodes_idx = {node: idx for idx, node in enumerate(self.terms)}

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
        """Build parent→child adjacency matrices per level transition."""
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

    def get_labels(self, category_name: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Convert a single WOS category name into global and local label vectors.

        For WOS, each document has exactly one leaf category. The method activates
        the leaf node and all its ancestors.

        Args:
            category_name: A single category name (e.g. "Machine learning").

        Returns:
            y_global: Binary vector of length len(self.terms).
            y_local: List of binary vectors, one per depth level.
        """
        y_global = np.zeros(len(self.terms), dtype=np.float32)
        y_local = [
            np.zeros(size, dtype=np.float32)
            for _, size in sorted(self.levels_size.items())
        ]

        # Root is always active
        y_global[self.nodes_idx["root"]] = 1.0
        y_local[0][self.local_nodes_idx[0]["root"]] = 1.0

        if category_name in self.nodes_idx:
            # Activate ancestors
            for ancestor in nx.ancestors(self.g_t, category_name):
                y_global[self.nodes_idx[ancestor]] = 1.0
                depth = nx.shortest_path_length(self.g_t, "root", ancestor)
                y_local[depth][self.local_nodes_idx[depth][ancestor]] = 1.0

            # Activate the category itself
            y_global[self.nodes_idx[category_name]] = 1.0
            depth = nx.shortest_path_length(self.g_t, "root", category_name)
            y_local[depth][self.local_nodes_idx[depth][category_name]] = 1.0

        return y_global, y_local

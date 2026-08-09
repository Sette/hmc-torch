"""
AAPD (arXiv Academic Paper Dataset) structures.

The AAPD dataset is a standard HTC benchmark with a 2-level hierarchy:
  - Root
    - 6 top-level fields: cs, math, physics, stat, q-bio, q-fin
    - 48 sub-fields (e.g. cs.AI, cs.CL, math.OC, physics.optics, ...)

Each paper has one or more sub-field labels, and the hierarchy propagates
upward: a paper labelled ``cs.CL`` also belongs to ``cs``.
"""

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# AAPD top-level fields and known sub-fields
AAPD_AREAS: dict[str, list[str]] = {
    "cs": [
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.CY",
        "cs.CR",
        "cs.DC",
        "cs.DS",
        "cs.GT",
        "cs.IR",
        "cs.IT",
        "cs.LG",
        "cs.LO",
        "cs.MA",
        "cs.MM",
        "cs.NE",
        "cs.NI",
        "cs.RO",
        "cs.SE",
        "cs.SI",
    ],
    "math": [
        "math.AC",
        "math.AG",
        "math.AP",
        "math.AT",
        "math.CA",
        "math.CO",
        "math.CT",
        "math.DS",
        "math.FA",
        "math.GM",
        "math.GR",
        "math.HO",
        "math.IT",
        "math.LO",
        "math.MP",
        "math.NA",
        "math.NT",
        "math.OA",
        "math.OC",
        "math.PR",
        "math.QA",
        "math.RA",
        "math.RT",
        "math.SG",
        "math.SP",
        "math.ST",
    ],
    "physics": [
        "physics.acc-ph",
        "physics.ao-ph",
        "physics.app-ph",
        "physics.atm-clus",
        "physics.bio-ph",
        "physics.chem-ph",
        "physics.class-ph",
        "physics.comp-ph",
        "physics.data-an",
        "physics.ed-ph",
        "physics.flu-dyn",
        "physics.gen-ph",
        "physics.geo-ph",
        "physics.hist-ph",
        "physics.ins-det",
        "physics.med-ph",
        "physics.optics",
        "physics.plasm-ph",
        "physics.pop-ph",
        "physics.soc-ph",
        "physics.space-ph",
    ],
    "stat": [
        "stat.AP",
        "stat.CO",
        "stat.ME",
        "stat.ML",
        "stat.OT",
        "stat.TH",
    ],
    "q-bio": [
        "q-bio.BM",
        "q-bio.CB",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.NC",
        "q-bio.OT",
        "q-bio.PE",
        "q-bio.QM",
        "q-bio.SC",
        "q-bio.TO",
    ],
    "q-fin": [
        "q-fin.CP",
        "q-fin.EC",
        "q-fin.GN",
        "q-fin.MF",
        "q-fin.PM",
        "q-fin.PR",
        "q-fin.RM",
        "q-fin.ST",
        "q-fin.TR",
    ],
}

# Flattened list: all 54 labels (6 top-level + 48 sub-fields)
AAPD_ALL_LABELS: list[str] = []
for _area, _subs in AAPD_AREAS.items():
    AAPD_ALL_LABELS.append(_area)
    AAPD_ALL_LABELS.extend(_subs)


class _SamplesHolder:
    """Mutable holder so the pipeline can attach tensor views."""

    x: Any | None = None
    y: Any | None = None


class AAPDSplit:
    """ARFF-compatible data split for one AAPD partition.

    Exposes the same ``.x``, ``.y``, ``.y_local`` and ``.samples`` interface
    as ``ArXivSplit`` and ``HMCDatasetArff`` so the pipelines treat AAPD
    splits identically to other dataset splits.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_local: list[list[np.ndarray]],
    ) -> None:
        self.x = x  # (N, feat_dim) float32
        self.y = y  # (N, total_labels) float32 — global binary labels
        self.y_local = y_local  # list[list[ndarray]] — per-sample, per-level
        self.samples = _SamplesHolder()


class AAPDHierarchyManager:
    """Manages the AAPD label hierarchy: root → 6 areas → 48 sub-fields.

    Exposes the same interface as ``ArXivHierarchyManager`` for label
    encoding and hierarchy attribute access.
    """

    def __init__(self) -> None:
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

    @classmethod
    def from_labels(cls, labels: list[str]) -> "AAPDHierarchyManager":
        """Build the hierarchy from a list of observed label strings.

        Labels are expected to be space-separated sub-field codes
        (e.g. ``"cs.CL math.OC"``).  The hierarchy is constructed from
        the known AAPD taxonomy; any label not in the predefined set is
        added as a top-level node under root.
        """
        mgr = cls()
        mgr._build_graphs(set(labels))
        mgr._build_edge_index()
        mgr.a = nx.to_numpy_array(mgr.g, nodelist=mgr.terms, dtype=np.float32)
        mgr._is_fitted = True
        logger.info(
            "AAPD hierarchy: %d terms, levels %s",
            len(mgr.terms),
            dict(mgr.levels_size),
        )
        return mgr

    def _build_graphs(self, observed_labels: set[str]) -> None:
        """Build DiGraph from observed labels and the known AAPD taxonomy."""
        self.g.add_node("root")
        self.levels[0].append("root")

        # Collect all areas and sub-fields present in the data
        seen_areas: set[str] = set()
        seen_subs: set[str] = set()

        for label in observed_labels:
            if "." in label:
                area = label.split(".")[0]
                seen_subs.add(label)
                seen_areas.add(area)
            else:
                seen_areas.add(label)

        # Build edges: root → area → sub
        for area in seen_areas:
            self.g.add_edge(area, "root")
        for sub in seen_subs:
            area = sub.split(".")[0]
            if area in seen_areas:
                self.g.add_edge(sub, area)
            else:
                self.g.add_edge(sub, "root")

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

    def get_labels(self, category_str: str) -> tuple[np.ndarray, list[np.ndarray]]:
        """Convert a space-separated category string into global and local labels.

        Each category in *category_str* activates that node plus all its
        ancestors in the hierarchy.

        Args:
            category_str: Space-separated label codes (e.g. ``"cs.CL math.OC"``).

        Returns:
            y_global: Binary vector of length ``len(self.terms)``.
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

        categories = category_str.split()
        for cat in categories:
            if cat not in self.nodes_idx:
                continue
            # Activate the category
            y_global[self.nodes_idx[cat]] = 1.0
            depth = nx.shortest_path_length(self.g_t, "root", cat)
            y_local[depth][self.local_nodes_idx[depth][cat]] = 1.0

            # Activate ancestors
            for ancestor in nx.ancestors(self.g_t, cat):
                y_global[self.nodes_idx[ancestor]] = 1.0
                anc_depth = nx.shortest_path_length(self.g_t, "root", ancestor)
                y_local[anc_depth][self.local_nodes_idx[anc_depth][ancestor]] = 1.0

        return y_global, y_local

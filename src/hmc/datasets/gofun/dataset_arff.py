"""
This module provides functionality to parse ARFF files and construct hierarchical datasets
for Hierarchical Multi-label Classification (HMC) models. It processes both Gene Ontology (GO)
and general hierarchical data formats, extracting features, labels, hierarchical levels,
and adjacency matrices representing the class hierarchy.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import keras
import networkx as nx
import numpy as np

from hmc.datasets.gofun import to_skip

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_depth_by_root(g_t, t, roots):
    """
    Calculates the depth of a node in a directed acyclic
    graph (DAG) by finding the shortest path to any of the specified roots.
    Args:
        g_t (nx.DiGraph): The transpose of the DAG.
        t (str): The node for which to calculate the depth.
        roots (list): A list of root nodes.
    Returns:
        int: The depth of the node.
    """
    for root in roots:
        depth = nx.shortest_path_length(g_t, t, root)
        if depth is not None:
            return depth
    return None


@dataclass
class HierarchyData:  # pylint: disable=too-many-instance-attributes
    """Holds all graph and hierarchy structures derived from the @ATTRIBUTE class line."""

    g: Any = field(default_factory=nx.DiGraph)
    g_t: Any = field(default_factory=nx.DiGraph)
    levels: Any = field(default_factory=lambda: defaultdict(list))
    levels_size: dict = field(default_factory=dict)
    nodes_idx: dict = field(default_factory=dict)
    local_nodes_idx: dict = field(default_factory=dict)
    max_depth: int = 0
    terms: list = field(default_factory=list)
    edge_index: dict = field(default_factory=dict)
    a: Any = None


@dataclass
class SampleData:
    """Holds the feature matrix and label arrays parsed from the @DATA section."""

    x: Any = None
    y: Any = None
    y_nodes: list = field(default_factory=list)
    y_local: list = field(default_factory=list)


class HMCDatasetArff:
    """
    Dataset torch para HMC local classifier.
    """

    def __init__(self, arff_file, is_go):
        self.arff_file = arff_file
        self.is_go = is_go
        self.hierarchy = HierarchyData()
        self.samples = SampleData()

        self.parse_arff()

        self.to_eval = [t not in to_skip for t in self.hierarchy.terms]
        r_, c_ = np.where(np.isnan(self.samples.x))
        m = np.nanmean(self.samples.x, axis=0)
        for i, j in zip(r_, c_):
            self.samples.x[i, j] = m[j]

    # ------------------------------------------------------------------
    # Convenience properties that preserve the original flat attribute API
    # ------------------------------------------------------------------

    @property
    def x(self):
        return self.samples.x

    @property
    def y(self):
        return self.samples.y

    @property
    def y_nodes(self):
        return self.samples.y_nodes

    @property
    def y_local(self):
        return self.samples.y_local

    @property
    def g(self):
        return self.hierarchy.g

    @property
    def g_t(self):
        return self.hierarchy.g_t

    @property
    def levels(self):
        return self.hierarchy.levels

    @property
    def levels_size(self):
        return self.hierarchy.levels_size

    @property
    def nodes_idx(self):
        return self.hierarchy.nodes_idx

    @property
    def local_nodes_idx(self):
        return self.hierarchy.local_nodes_idx

    @property
    def max_depth(self):
        return self.hierarchy.max_depth

    @property
    def terms(self):
        return self.hierarchy.terms

    @property
    def edge_index(self):
        return self.hierarchy.edge_index

    @property
    def a(self):
        return self.hierarchy.a

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def build_hierarchy_graph(self, h):
        """Parse hierarchical attribute declaration into a directed graph.

        Populates ``self.hierarchy.g`` and ``self.hierarchy.levels`` as side
        effects and also returns them for convenience.
        """
        g = nx.DiGraph()
        levels = defaultdict(list)

        for branch in h.split(","):
            branch = branch.replace("/", ".")
            terms = branch.split(".")

            if self.is_go:
                g.add_edge(terms[1], terms[0])
            else:
                level = len(terms) - 1
                levels[level].append(branch)
                if len(terms) == 1:
                    g.add_edge(terms[0], "root")
                else:
                    for i in range(2, len(terms) + 1):
                        g.add_edge(
                            ".".join(terms[:i]),
                            ".".join(terms[: i - 1]),
                        )

        self.hierarchy.g = g
        self.hierarchy.levels = levels
        return g, levels

    def build_node_structures(self):
        """Compute node indices, level sizes, and local node indices from ``self.hierarchy.g``.

        Populates the corresponding fields in ``self.hierarchy`` as side
        effects and also returns the node list for convenience.
        """
        g = self.hierarchy.g
        levels = self.hierarchy.levels

        nodes = sorted(
            g.nodes(),
            key=lambda x: (
                (nx.shortest_path_length(g, x, "root"), x)
                if self.is_go
                else (len(x.split(".")), x)
            ),
        )
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()

        if self.is_go:
            for label in nodes:
                if label != "root":
                    level = nx.shortest_path_length(g_t, "root").get(label) - 1
                    levels[level].append(label)

        levels_size = {key: len(set(value)) for key, value in levels.items()}
        max_depth = len(levels_size)
        local_nodes_idx = {
            idx: dict(zip(level_nodes, range(len(level_nodes))))
            for idx, level_nodes in levels.items()
        }

        self.hierarchy.terms = nodes
        self.hierarchy.nodes_idx = nodes_idx
        self.hierarchy.g_t = g_t
        self.hierarchy.levels_size = levels_size
        self.hierarchy.max_depth = max_depth
        self.hierarchy.local_nodes_idx = local_nodes_idx
        return nodes

    def build_edge_index(self):
        """Build parent→child adjacency matrices between consecutive hierarchy levels.

        Populates and returns ``self.hierarchy.edge_index``.
        """
        level_nodes_list = list(self.hierarchy.levels.values())
        edge_index = {}

        for idx, current_level_nodes in enumerate(level_nodes_list):
            if idx == 0:
                continue
            prev_level_nodes = level_nodes_list[idx - 1]
            shape = (len(prev_level_nodes), len(current_level_nodes))
            matrix = np.zeros(shape, dtype=np.float32)

            child_map = {node: i for i, node in enumerate(current_level_nodes)}
            parent_map = {node: i for i, node in enumerate(prev_level_nodes)}

            for c_node in current_level_nodes:
                if self.hierarchy.g.has_node(c_node):
                    for p in self.hierarchy.g.successors(c_node):
                        a = parent_map.get(p)
                        b = child_map.get(c_node)
                        if a is not None and b is not None:
                            matrix[a, b] = 1.0

            edge_index[idx] = matrix

        self.hierarchy.edge_index = edge_index
        return edge_index

    def parse_arff(self):
        """Parse ``self.arff_file`` and populate all dataset attributes."""
        with open(self.arff_file, "r", encoding="utf-8") as f:
            feature_types = self._parse_attributes(f)
            self._parse_data_lines(f, feature_types)

        self.build_edge_index()
        self.hierarchy.a = np.array(
            nx.to_numpy_array(self.hierarchy.g, nodelist=self.hierarchy.terms)
        )

        logger.info(
            "Shape of edges matrix: %s",
            {k: v.shape for k, v in self.hierarchy.edge_index.items()},
        )
        logger.info("Parsed ARFF file: %s", self.arff_file)
        logger.info("Number of matrix: %d", len(self.hierarchy.edge_index))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_feature_attribute(self, f_type, d, cats_lens):
        """Return a feature parsing function for one non-class @ATTRIBUTE line."""
        if f_type in ("numeric", "NUMERIC"):
            d.append([])
            cats_lens.append(1)
            return lambda x, i: [float(x)] if x != "?" else [np.nan]

        cats = f_type[1:-1].split(",")
        cats_lens.append(len(cats))
        d.append(
            {
                key: keras.utils.to_categorical(i, len(cats)).tolist()
                for i, key in enumerate(cats)
            }
        )
        return lambda x, i: d[i].get(x, [0.0] * cats_lens[i])

    def _parse_feature_vector(self, d_line, feature_types):
        """Parse feature columns from a data line into a flat list."""
        return list(
            chain(
                *[
                    feature_types[i](x, i)
                    for i, x in enumerate(d_line[: len(feature_types)])
                ]
            )
        )

    def _parse_sample_labels(self, lab):
        """Parse the label column for one sample into y_, y_nodes, y_local_."""
        h = self.hierarchy
        sorted_keys = sorted(h.levels_size.keys())
        y_ = np.zeros(len(h.terms))
        y_nodes = []
        y_local_ = [np.zeros(h.levels_size.get(key)) for key in sorted_keys]

        for t in lab.split("@"):
            y_node = t.replace("/", ".")
            y_nodes.append(y_node)
            y_[[h.nodes_idx.get(a) for a in nx.ancestors(h.g_t, y_node)]] = 1
            y_[h.nodes_idx[y_node]] = 1

            if self.is_go:
                depth = nx.shortest_path_length(h.g_t, "root").get(y_node) - 1
                y_local_[depth][h.local_nodes_idx[depth].get(y_node)] = 1
                for ancestor in nx.ancestors(h.g_t, y_node):
                    if ancestor != "root":
                        depth = nx.shortest_path_length(h.g_t, "root").get(ancestor) - 1
                        y_local_[depth][h.local_nodes_idx[depth].get(ancestor)] = 1
            else:
                depth = y_node.count(".") + 1
                assert depth is not None
                for index in range(depth, 0, -1):
                    local_terms = y_node.split(".")[:index]
                    local_label = ".".join(local_terms)
                    local_depth = len(local_terms) - 1
                    y_local_[local_depth][
                        h.local_nodes_idx.get(local_depth).get(local_label)
                    ] = 1

        return y_, y_nodes, y_local_

    def _parse_attributes(self, f):
        """Read @ATTRIBUTE lines from f (stopping at @DATA) and return feature_types."""
        feature_types = []
        d = []
        cats_lens = []

        for line in f:
            if line.startswith("@DATA"):
                break
            if line.startswith("@ATTRIBUTE class"):
                h = line.split("hierarchical")[1].strip()
                self.build_hierarchy_graph(h)
                self.build_node_structures()
            elif line.startswith("@ATTRIBUTE"):
                _, _, f_type = line.split()
                feature_types.append(
                    self._parse_feature_attribute(f_type, d, cats_lens)
                )

        return feature_types

    def _parse_data_lines(self, f, feature_types):
        """Read data lines from f and populate self.samples."""
        x = []
        y = []
        y_nodes = []
        y_local = []

        for line in f:
            d_line = line.split("%")[0].strip().split(",")
            lab = d_line[len(feature_types)].strip()
            x.append(self._parse_feature_vector(d_line, feature_types))
            y_, sample_y_nodes, y_local_ = self._parse_sample_labels(lab)
            y.append(y_)
            y_nodes.append(sample_y_nodes)
            y_local.append([np.stack(yy) for yy in y_local_])

        self.samples.x = np.array(x)
        self.samples.y = np.stack(y)
        self.samples.y_nodes = y_nodes
        self.samples.y_local = y_local

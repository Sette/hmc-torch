"""Utility modules for HMC training and evaluation.

The key utility for custom datasets is :func:`build_digraph_from_labels`,
which builds a child→parent DiGraph from a flat list of hierarchical label
strings — just pass your labels and the framework derives everything else
(adjacency, levels, edge indices, dimensions) automatically.
"""

from hmc.utils.graph import ancestor_closure, build_digraph_from_labels, build_r_matrix

__all__ = ["ancestor_closure", "build_digraph_from_labels", "build_r_matrix"]

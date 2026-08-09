"""HMC-Torch: A Modular Platform for Hierarchical Multi-Label Classification.

Provides a complete pipeline for HMC with R-matrix constraints, supporting
25+ datasets across 5 domains and 4 modalities.

Quick start::

    import hmc

    # List available datasets
    print(hmc.DatasetRegistry.list_available())

    # Train a model
    results = hmc.train("wos", method="globalE2E", device="cuda", epochs=5)

    # Register a custom dataset — just data + labels
    from hmc.utils import build_digraph_from_labels
    from hmc.data.hierarchy import TreeHierarchy
    from hmc.data import DatasetRegistry

    h = TreeHierarchy.from_graph(build_digraph_from_labels(my_labels))
    # ... wrap in a manager and register (see README for full example)

    @DatasetRegistry.register("my_data", defaults={"hidden_dim": 256})
    def make_my_data(**kwargs):
        ...
"""

from __future__ import annotations

import os

from hmc.arguments import Args, parse_args
from hmc.data.base import DatasetBundle, FeatureMetadata, Modality, Split
from hmc.data.hierarchy import Hierarchy
from hmc.data.protocols import DatasetManagerProtocol
from hmc.data.registry import DatasetRegistry
from hmc.datasets.dataset_manager import _ensure_builtins as _reg_builtins
from hmc.train import train

# Suppress HuggingFace progress bars by default
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ---------------------------------------------------------------------------
# Public API — configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API — data contracts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auto-register built-in datasets on import
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API — training
# ---------------------------------------------------------------------------

_reg_builtins()

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.0.9"

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    "Args",
    "DatasetBundle",
    "DatasetManagerProtocol",
    "DatasetRegistry",
    "FeatureMetadata",
    "Hierarchy",
    "Modality",
    "Split",
    "__version__",
    "parse_args",
    "train",
]

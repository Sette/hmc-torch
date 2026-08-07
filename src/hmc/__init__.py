"""HMC-Torch: A Modular Platform for Hierarchical Multi-Label Classification.

Provides a complete pipeline for HMC with R-matrix constraints, supporting
25+ datasets across 5 domains and 4 modalities.

Quick start::

    import hmc

    # List available datasets
    print(hmc.DatasetRegistry.list_available())

    # Train a model
    results = hmc.train("wos", method="globalE2E", device="cuda", epochs=5)

    # Register a custom dataset
    from hmc.data import DatasetRegistry

    @DatasetRegistry.register("my_data", defaults={"hidden_dim": 256})
    def make_my_data(**kwargs):
        ...
"""

from __future__ import annotations

import os

# Suppress HuggingFace progress bars by default
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ---------------------------------------------------------------------------
# Public API — data contracts
# ---------------------------------------------------------------------------
from hmc.data.base import DatasetBundle, FeatureMetadata, Modality, Split
from hmc.data.hierarchy import Hierarchy
from hmc.data.protocols import DatasetManagerProtocol
from hmc.data.registry import DatasetRegistry

# ---------------------------------------------------------------------------
# Public API — configuration
# ---------------------------------------------------------------------------
from hmc.arguments import Args, parse_args

# ---------------------------------------------------------------------------
# Public API — training
# ---------------------------------------------------------------------------
from hmc.train import train

# ---------------------------------------------------------------------------
# Auto-register built-in datasets on import
# ---------------------------------------------------------------------------
from hmc.datasets.dataset_manager import _ensure_builtins as _reg_builtins
_reg_builtins()

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.0.9"

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    # Data contracts
    "DatasetBundle",
    "DatasetManagerProtocol",
    "DatasetRegistry",
    "FeatureMetadata",
    "Hierarchy",
    "Modality",
    "Split",
    # Configuration
    "Args",
    "parse_args",
    # Training
    "train",
    # Version
    "__version__",
]

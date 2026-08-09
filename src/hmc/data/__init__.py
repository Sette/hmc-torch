"""HMC-Torch data contracts and dataset registry.

This module defines the canonical shapes that every dataset adapter must
produce (:class:`DatasetBundle`, :class:`Split`, :class:`Hierarchy`),
independent of modality or hierarchy type, and the :class:`DatasetRegistry`
that makes datasets discoverable by name.
"""

from hmc.data.base import DatasetBundle, FeatureMetadata, Modality, Split
from hmc.data.hierarchy import Hierarchy
from hmc.data.protocols import DatasetManagerProtocol
from hmc.data.registry import DatasetRegistry, pick_defaults

__all__ = [
    "DatasetBundle",
    "DatasetManagerProtocol",
    "DatasetRegistry",
    "FeatureMetadata",
    "Hierarchy",
    "Modality",
    "Split",
    "pick_defaults",
]

"""Common data contracts for multimodal HMC datasets.

Defines the canonical shapes that every dataset adapter must produce,
independent of modality or hierarchy type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class Modality(str, Enum):
    """Supported data modalities."""

    TABULAR = "tabular"
    TEXT = "text"
    VISION = "vision"
    PROTEIN = "protein"
    EXPRESSION = "expression"


@dataclass
class FeatureMetadata:
    """Describes the feature side of a dataset split.

    Attributes:
        modality: Primary modality of the features.
        n_features: Number of feature columns (after encoding).
        has_missing: Whether the raw data contains missing values.
        sparsity: Fraction of zero entries (0.0 = dense, 1.0 = all zeros).
        sample_ids: Optional per-sample identifiers for dedup and auditing.
        raw_available: Whether raw data (text/images/sequences) is accessible,
            or only pre-extracted features.
    """

    modality: Modality = Modality.TABULAR
    n_features: int = 0
    has_missing: bool = False
    sparsity: float = 0.0
    sample_ids: list[str] | None = None
    raw_available: bool = False


@dataclass
class Split:
    """One dataset split (train, validation, or test).

    Attributes:
        features: Feature matrix of shape ``(n_samples, n_features)``.
        labels: Binary label matrix of shape ``(n_samples, n_nodes)``.
        local_labels: Per-level label vectors, one :class:`numpy.ndarray`
            per hierarchy level (length ``max_depth``).  Each entry has
            shape ``(n_samples, n_classes_at_level)``.
        sample_ids: Optional per-sample identifiers.
        metadata: Feature metadata for this split.
    """

    features: np.ndarray
    labels: np.ndarray
    local_labels: list[np.ndarray] = field(default_factory=list)
    sample_ids: list[str] | None = None
    metadata: FeatureMetadata = field(default_factory=FeatureMetadata)

    @property
    def n_samples(self) -> int:
        """Number of samples in this split."""
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.features.shape[1])

    @property
    def n_nodes(self) -> int:
        """Number of class nodes in the hierarchy."""
        return int(self.labels.shape[1])


@dataclass
class DatasetBundle:
    """Complete dataset with all splits, hierarchy, and metadata.

    This is the canonical return type for dataset adapters.  Every loader
    (ARFF, JSONL, HDF5, …) must produce a :class:`DatasetBundle`.

    Attributes:
        train: Training split.
        valid: Validation split (may be ``None`` for datasets without one).
        test: Test split.
        hierarchy: The class hierarchy (tree or DAG).
        metadata: Feature metadata shared across splits (modality, dims, …).
    """

    train: Split
    test: Split
    hierarchy: Hierarchy  # noqa: F821  # pylint: disable=undefined-variable  # forward ref, resolved at runtime
    valid: Split | None = None
    metadata: FeatureMetadata = field(default_factory=FeatureMetadata)

    @property
    def n_nodes(self) -> int:
        """Total number of class nodes."""
        return self.hierarchy.n_nodes

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return self.train.n_features


@runtime_checkable
class FeatureEncoder(Protocol):
    """Protocol for feature encoders.

    Every encoder must support ``fit`` (on train only) and ``transform``
    (on any split).  The output is a new :class:`Split` (or subclass)
    with encoded features.
    """

    def fit(self, split: Split) -> FeatureEncoder:
        """Fit parameters (scaler, imputer, selector) on *train* only."""
        raise NotImplementedError

    def transform(self, split: Split) -> Split:
        """Apply the fitted transformation to a split."""
        raise NotImplementedError

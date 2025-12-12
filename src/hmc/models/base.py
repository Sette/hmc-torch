import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ============================================================================
# PARENT CLASS - Base Model
# ============================================================================


class HierarchicalModel(nn.Module, ABC):
    """
    Abstract parent class for hierarchical models.

    Defines common structure and interface for all hierarchical models.
    """

    def __init__(
        self,
        levels_size: List[int],
        input_size: int,
        results_path: str,
        active_levels: Optional[List[int]] = None,
    ):
        """
        Initialize the hierarchical model.

        Args:
            levels_size: Number of classes at each hierarchical level
            input_size: Dimension of input features
            results_path: Directory to save/load model checkpoints
            active_levels: Indices of levels to train (None = all levels)
        """
        super(HierarchicalModel, self).__init__()

        # Validate inputs
        self._validate_inputs(input_size, levels_size, results_path)

        # Store configuration
        self.input_size = input_size
        self.levels_size = levels_size
        self.results_path = results_path
        self.max_depth = len(levels_size)

        # Set active levels
        if active_levels is None:
            active_levels = list(range(len(levels_size)))
        self.active_levels = active_levels
        self.level_active = [i in active_levels for i in range(len(levels_size))]

        logging.info(
            "%s initialized with input_size=%d, max_depth=%d, active_levels=%s",
            self.__class__.__name__,
            input_size,
            self.max_depth,
            active_levels,
        )

    @staticmethod
    def _validate_inputs(input_size: int, levels_size: List[int], results_path: str):
        """Validate input parameters."""
        if not input_size or input_size <= 0:
            raise ValueError(f"Invalid input_size: {input_size}")
        if not levels_size:
            raise ValueError("levels_size cannot be empty")
        if not results_path:
            raise ValueError("results_path is required")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping level indices to output tensors
        """
        pass

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level."""
        pass

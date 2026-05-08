"""
Local classifier model for Hierarchical Multi-label Classification (HMC).

This module provides :class:`HMCLocalModel`, a hierarchical model that trains
an independent classifier at each level of the label hierarchy.
"""

import logging
import os
from typing import Dict, List, Optional

import torch
from torch import nn

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import BuildClassification


class HMCLocalModel(HierarchicalModel):
    """
    Hierarchical model where each level makes independent predictions.

    Child class that implements local classification at each level.
    Optionally supports residual connections between levels.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        levels_size: List[int],
        input_size: int,
        hidden_dims: List[int],
        results_path: str,
        num_layers: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        active_levels: Optional[List[int]] = None,
    ):
        """
        Initialize local classification model.

        Args:
            levels_size: Number of classes at each level.
            input_size: Dimension of input features.
            hidden_dims: Hidden layer sizes for each level.
            results_path: Path to save model checkpoints.
            num_layers: Number of layers for each level (default: 2).
            dropouts: Dropout rates for each level (default: 0.0).
            active_levels: Indices of levels to train (default: all levels).
        """
        super().__init__(
            levels_size=levels_size,
            input_size=input_size,
            results_path=results_path,
            active_levels=active_levels,
        )

        if num_layers is None:
            num_layers = [2] * len(levels_size)
        if dropouts is None:
            dropouts = [0.0] * len(levels_size)

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.levels = nn.ModuleDict()
        self.level_active = [True] * len(levels_size)

        self._build_levels()

    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:
            level_classifier = BuildClassification(
                {
                    "input_size": self.input_size,
                    "output_size": self.levels_size[level_idx],
                    "num_layers": self.num_layers[level_idx],
                    "dropout": self.dropouts[level_idx],
                    "hidden_dims": self.hidden_dims[level_idx],
                    "level": level_idx,
                    "device": "cpu",
                }
            )
            self.levels[str(level_idx)] = level_classifier
            logging.info(
                "Level %d: input_size=%d, output_size=%d",
                level_idx,
                self.input_size,
                self.levels_size[level_idx],
            )

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass through all active levels.

        Args:
            x: Input tensor of shape ``(batch_size, input_size)``.

        Returns:
            Dictionary mapping level index strings to output tensors.
        """
        outputs = {}
        current_input = x

        for level_idx, level_module in self.levels.items():
            level_idx = int(level_idx)

            if not self.level_active[level_idx]:
                self._load_checkpoint(level_idx)

            outputs[str(level_idx)] = level_module(current_input)

        return outputs

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level.

        Args:
            level_idx: Index of the level whose checkpoint to load.

        Returns:
            ``True`` if the checkpoint was found and loaded, ``False`` otherwise.
        """
        checkpoint_path = os.path.join(
            self.results_path, f"best_model_level_{level_idx}.pth"
        )
        if os.path.exists(checkpoint_path):
            self.levels[str(level_idx)].load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            return True
        return False

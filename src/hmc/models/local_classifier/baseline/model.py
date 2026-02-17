import os
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import BuildClassification


class HMCLocalModel(HierarchicalModel):
    """
    Hierarchical model where each level makes independent predictions.

    Child class that implements local classification at each level.
    Optionally supports residual connections between levels.
    """

    def __init__(
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
            levels_size: Number of classes at each level
            input_size: Dimension of input features
            hidden_dims: Hidden layer sizes for each level
            results_path: Path to save model checkpoints
            num_layers: Number of layers for each level (default: 2)
            dropouts: Dropout rates for each level (default: 0.0)
            active_levels: Indices of levels to train
            residual: Whether to use residual connections
        """
        super(HMCLocalModel, self).__init__(
            levels_size=levels_size,
            input_size=input_size,
            results_path=results_path,
            active_levels=active_levels,
        )

        # Set defaults
        if num_layers is None:
            num_layers = [2] * len(levels_size)
        if dropouts is None:
            dropouts = [0.0] * len(levels_size)

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropouts = dropouts
        # Create level modules
        self.levels = {}  # dict {level_idx: {'encoder': ModuleList, 'level_classifier': BuildClassification}}
        self.level_active = [True] * len(levels_size)
        
       
        self._build_levels()

    # noinspection PyGlobalUndefined
    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:
            # Input size depends on residual connections
            current_input_size = self.input_size
            
            level_classifier = BuildClassification(
                input_size=current_input_size,
                output_size=self.levels_size[level_idx],
                num_layers=self.num_layers[level_idx],
                dropout=self.dropouts[level_idx],
                hidden_dim=self.hidden_dims[level_idx],
                level=level_idx,
            )

            self.levels[level_idx] = {
                'level_classifier': level_classifier
            }

            logging.info(
                "Level %d: input_size=%d, output_size=%d",
                level_idx, current_input_size, self.levels_size[level_idx]
            )

    def forward(
            self,
            x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass with optional residual connections.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Dictionary {level_idx: output_tensor}
        """
        outputs = {}
        current_input = x

        for level_idx, level_module in self.levels.items():
            level_idx = int(level_idx)

            # Load checkpoint if level is inactive
            if not self.level_active[level_idx]:
                self._load_checkpoint(level_idx)

            # Forward through level
            level_output = level_module['level_classifier'](current_input)
            outputs[level_idx] = level_output

        return outputs

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level."""

        checkpoint_name = f"best_model_level_{level_idx}.pth"
        checkpoint_path = os.path.join(self.results_path, checkpoint_name)

        if os.path.exists(checkpoint_path):
            # logging.info(f"Loading checkpoint: {checkpoint_path}")
            self.levels[level_idx]['level_classifier'].load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            return True
        return False

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import (
    ClassificationNetwork,
    BuildClassification,
)


# ============================================================================
# CHILD CLASS 2 - Local Classification Model constraint
# ============================================================================


def build_edge_index_from_r(r_global):
    # R: (num_labels_total x num_labels_total)
    src, dst = torch.nonzero(r_global, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index.long()


def slice_level_edge_index(edge_index_global, level_indices):
    level_mask = torch.zeros(edge_index_global.max() + 1, dtype=torch.bool)
    level_mask[level_indices] = True

    src = edge_index_global[0]
    dst = edge_index_global[1]

    mask = level_mask[src] & level_mask[dst]
    return edge_index_global[:, mask]


class HMCLocalModelConstraint(HierarchicalModel):
    """
    Hierarchical model where each level makes independent predictions.

    Child class that implements local classification at each level.
    Optionally supports residual connections between levels.
    This variant includes hierarchical constraints specific to the model.
    """

    def __init__(
        self,
        levels_size: List[int],
        input_size: int,
        hidden_dims: List[int],
        results_path: str,
        level_model_type: Optional[str],
        class_indices_per_level: Optional[List[int]] = None,
        num_layers: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        active_levels: Optional[List[int]] = None,
        residual: bool = False,
        num_heads: int = 1,
        r_global: Optional[List[float]] = None,
        constraint_alpha: float = 0.8,
    ):
        """
        Initialize local classification model with constraints.

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
        super(HMCLocalModelConstraint, self).__init__(
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

        # ===== model_type por nível =====
        if isinstance(level_model_type, dict):
            self.model_type_map = level_model_type
        else:
            # default = mlp
            self.model_type_map = {
                lvl: (level_model_type or "mlp") for lvl in active_levels
            }

        if level_model_type is not None:
            self.level_model_type = level_model_type

        if num_heads is not None:
            self.num_heads = num_heads

        if r_global is not None:
            self.r_global = r_global

        if constraint_alpha:
            self.constraint_alpha = constraint_alpha

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.residual = residual

        # Edge index global
        self.edge_index_global = build_edge_index_from_r(r_global)

        # Edge index por nível
        self.edge_index_levels = []
        for lvl_idxs in class_indices_per_level:
            e = slice_level_edge_index(self.edge_index_global, lvl_idxs)
            self.edge_index_levels.append(e)

        # Create level modules
        self.levels = nn.ModuleDict()
        self._build_levels()

    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:

            model_type = self.model_type_map.get(level_idx, "mlp").lower()

            # flags explícitos
            self.use_mlp = model_type == "mlp"
            self.use_attention = model_type == "attention"
            self.use_gcn = model_type == "gcn"
            self.use_gat = model_type == "gat"

            # Input size depends on residual connections
            if self.residual and level_idx > 0:
                in_dim = self.input_size + self.levels_size[level_idx - 1]
            else:
                in_dim = self.input_size

            # Create classification network
            self.levels[str(level_idx)] = BuildClassification(
                input_size=in_dim,
                hidden_dims=self.hidden_dims[level_idx],
                output_size=self.levels_size[level_idx],
                num_layers=self.num_layers[level_idx],
                dropout=self.dropouts[level_idx],
                attention=self.use_attention if not self.use_mlp else False,
                gatconv=self.use_gat if not self.use_mlp else False,
                gcnconv=self.use_gcn if not self.use_mlp else False,
                num_heads=self.num_heads,
            )

            logging.debug(
                f"Level {level_idx}: input_size={in_dim}, "
                f"output_size={self.levels_size[level_idx]}"
            )

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
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

            edge_index = self.edge_index_levels[level_idx].to(x.device)

            # Load checkpoint if level is inactive
            if not self.level_active[level_idx]:
                self._load_checkpoint(level_idx)

            # Add residual connection
            if self.residual and level_idx > 0:
                previous_output = outputs[level_idx - 1]
                # Thresholding: treat as binary (0 or 1)
                previous_output_binary = (previous_output > 0.5).float()
                current_input = torch.cat((x, previous_output_binary), dim=1)

            # Forward through level
            if self.use_gat or self.use_gcn:
                level_output = level_module(current_input, edge_index)
            else:
                level_output = level_module(current_input)
            outputs[level_idx] = level_output

        return outputs

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level."""
        import os

        checkpoint_name = f"best_model_level_{level_idx}.pth"
        checkpoint_path = os.path.join(self.results_path, checkpoint_name)

        if os.path.exists(checkpoint_path):
            # logging.info(f"Loading checkpoint: {checkpoint_path}")
            self.levels[str(level_idx)].load_state_dict(torch.load(checkpoint_path))
            return True
        return False

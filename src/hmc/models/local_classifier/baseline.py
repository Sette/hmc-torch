import os
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import BuildClassification


class EncoderBlock(nn.Module):
    """Encoder block with multi-head self-attention and a feed-forward sublayer.

    This block follows the Transformer-style residual pattern:
    1) Multi-head self-attention + dropout + residual + layer norm
    2) Feed-forward network + dropout + residual + layer norm

    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        ff_dim: Inner dimension of the feed-forward network.
        dropout: Dropout probability applied after attention and in FFN.

    Forward shapes:
        x: (batch_size, seq_len, embed_dim)
        key_padding_mask: optional mask of shape (batch_size, seq_len) with True for padded positions
        attn_mask: optional attention mask (seq_len, seq_len) or (batch_size, seq_len, seq_len)

    Returns:
        Tensor of shape (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        # batch_first=True: inputs are (batch, seq, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the encoder block.

        Note: key_padding_mask uses True for positions that should be ignored (padded).
        """
        # Self-attention residual
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ============================================================================
# CHILD CLASS 1 - Local Classification Model
# ============================================================================


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
        residual: bool = False,
        attention: bool = False,
        gcn: bool = False,
        gat: bool = False,
        num_heads: int = 1,
        edges_index=None,
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

        self.attention = attention
        self.gcn = gcn
        self.gat = gat
        self.num_heads = num_heads
        self.edges_index = edges_index

        # Set defaults
        if num_layers is None:
            num_layers = [2] * len(levels_size)
        if dropouts is None:
            dropouts = [0.0] * len(levels_size)

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.residual = residual

        # Create level modules
        self.levels = nn.ModuleDict()
        self._build_levels()

    def _build_levels(self):
        """Build classification networks for each active level."""
        for index in self.active_levels:
            # Input size depends on residual connections
            current_input_size = self.input_size

            # Create classification network
            self.levels[str(index)] = BuildClassification(
                input_size=current_input_size,
                hidden_dims=self.hidden_dims[index],
                output_size=self.levels_size[index],
                num_layers=self.num_layers[index],
                dropout=self.dropouts[index],
                attention=self.attention and index > 3,  # attention from level 4 onwards
                gcn=self.gcn,
                gat=self.gat,
                num_heads=self.num_heads,
                level=index,
            )

            logging.debug(
                "Level %d: input_size=%d, output_size=%d",
                index, current_input_size, self.levels_size[index]
            )

    def forward(
            self,
            x: torch.Tensor,
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

            # Add residual connection
            if self.residual and level_idx > 0:
                previous_output = outputs[level_idx - 1]
                # Thresholding: treat as binary (0 or 1)
                previous_output_binary = (previous_output > 0.5).float()
                current_input = torch.cat((x, previous_output_binary), dim=1)

            # Forward through level
            level_output = level_module(current_input, edge_index=self.edges_index)
            outputs[level_idx] = level_output

        return outputs

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level."""

        checkpoint_name = f"best_model_level_{level_idx}.pth"
        checkpoint_path = os.path.join(self.results_path, checkpoint_name)

        if os.path.exists(checkpoint_path):
            # logging.info(f"Loading checkpoint: {checkpoint_path}")
            self.levels[str(level_idx)].load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            return True
        return False

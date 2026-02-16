import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import BuildClassification


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention for GO classification"""

    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, input_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, input_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        context = context.view(batch_size, seq_len, self.input_dim)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights

# ============================================================================
# CHILD CLASS 1 - Local Classification Model
# ============================================================================

def _get_divisible_dim(dim: int, divisor: int) -> int:
    """Próximo múltiplo de divisor >= dim. Ex: _get_divisible_dim(529, 8) → 536"""
    return ((dim + divisor - 1) // divisor) * divisor

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
        encoder_block: bool = False,
        num_heads: int = 8,
        encoder_heads: Optional[List[int]] = None,  # heads por nível para EncoderBlock
        edges_index: Optional[Dict[int, torch.Tensor]] = None,  # dict {level: edge_index}
        encoder_ff_dim: int = 1024,  # FFN dim no EncoderBlock
        encoder_layers: int = 1,  # quantos EncoderBlock empilhados por nível
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
        self.encoder_block = encoder_block
        self.num_heads = num_heads
        self.encoder_heads = encoder_heads or [num_heads // 2] * len(levels_size)  # default menor que GAT
        self.edges_index = edges_index or {}
        self.encoder_ff_dim = encoder_ff_dim
        self.encoder_layers = encoder_layers
        self.residual = residual

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
        self.levels = {}  # dict {level_idx: {'encoder': ModuleList, 'classifier': BuildClassification}}
        
        if self.encoder_block:
            self.effective_embed_dim = _get_divisible_dim(self.input_size, self.num_heads)
            self.embed_proj = nn.Linear(self.input_size, self.effective_embed_dim)
            self.input_size = self.effective_embed_dim
        self._build_levels()

    # noinspection PyGlobalUndefined
    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:
            global multihead_attention_list
            # Input size depends on residual connections
            current_input_size = self.input_size
            if self.encoder_block:
                multihead_attention_list = nn.ModuleList()
                for encoder_index in range(self.encoder_layers):
                    multihead_attention_list.append(
                        MultiHeadAttentionLayer(current_input_size,
                                                self.encoder_heads[encoder_index],
                                                self.dropouts[encoder_index])
                    )

            level_classifier = BuildClassification(
                input_size=current_input_size,
                output_size=self.levels_size[level_idx],
                num_layers=self.num_layers[level_idx],
                dropout=self.dropouts[level_idx],
                level=level_idx,
            )

            
            self.levels[level_idx] = {
                'multihead_attention': multihead_attention_list if self.encoder_block else None,
                'level_classifier': level_classifier
            }

            logging.info(
                "Level %d: input_size=%d, output_size=%d",
                level_idx, current_input_size, self.levels_size[level_idx]
            )

    def forward(
            self,
            x: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with optional residual connections.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Dictionary {level_idx: output_tensor}
        """
        outputs = {}
        attention_maps = []
        current_input = x
        if hasattr(self, 'embed_proj'):
            current_input = self.embed_proj(current_input)  # 529 → 536


        for level_idx, level_module in self.levels.items():
            level_idx = int(level_idx)

            # Load checkpoint if level is inactive
            if not self.level_active[level_idx]:
                self._load_checkpoint(level_idx)

            # === RESIDUAL CONNECTION ===
            if self.residual and level_idx > 0:
                previous_output = outputs[level_idx - 1]
                # Thresholding: treat as binary (0 or 1)
                previous_output_binary = (previous_output > 0.5).float()
                current_input = torch.cat((x, previous_output_binary), dim=1)
            if self.encoder_block:
                # Multi-head attention
                attn_output, attn_weights = level_module['multihead_attention'](current_input)
                attention_maps.append(attn_weights)

                # Residual connection + Layer norm
                current = level_module['level_classifier'](current_input + attn_output)

                # Feed-forward network
                ff_output = level_module['level_classifier'](current)
                current = level_module['level_classifier'](current + ff_output)

                # Average pooling over sequence dimension for classification
                pooled = current.mean(dim=1)  # (batch_size, input_dim)

                # Classification for this level
                logits = level_module['level_classifier'](pooled)
                outputs[level_idx] = logits

            else:
                # Forward through level
                level_output = level_module['level_classifier'](current_input)
                outputs[level_idx] = level_output

        return outputs, attention_maps

    def _load_checkpoint(self, level_idx: int) -> bool:
        """Load a saved checkpoint for a specific level."""

        checkpoint_name = f"best_model_level_{level_idx}.pth"
        checkpoint_path = os.path.join(self.results_path, checkpoint_name)

        if os.path.exists(checkpoint_path):
            # logging.info(f"Loading checkpoint: {checkpoint_path}")
            self.levels[level_idx]['classifier'].load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )
            return True
        return False

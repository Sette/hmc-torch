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
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True acima da diagonal principal → bloqueia futuro
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask  # [L, L], bool

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the encoder block.

        Note: key_padding_mask uses True for positions that should be ignored (padded).
        """
        # Se não veio key_padding_mask:
        if key_padding_mask is None:
            key_padding_mask = (x == 0).all(dim=-1).to(x.device)   # [B, L], bool
        else:
            key_padding_mask = key_padding_mask.to(x.device)

        # Se não veio attn_mask e você quer causal:
        if attn_mask is None:
            attn_mask = self._generate_causal_mask(x.size(0), x.device).to(x.device)
        else:
            attn_mask = attn_mask.to(x.device)
        # Self-attention residual
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        attn_out = attn_out.to(x.device)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward residual
        ff_out = self.ff(x)
        ff_out = ff_out.to(x.device)
        x = self.norm2(x + self.dropout(ff_out))
        return x


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
        self.classifiers = nn.ModuleDict()  # dict {level_idx: {'encoder': ModuleList, 'classifier': BuildClassification}}
        self.encoders = nn.ModuleDict()  # dict {level_idx: ModuleList of EncoderBlocks}
        self.levels = {}  # dict {level_idx: {'encoder': ModuleList, 'classifier': BuildClassification}}
        
        if self.encoder_block:
            self.effective_embed_dim = _get_divisible_dim(self.input_size, self.num_heads)
            self.embed_proj = nn.Linear(self.input_size, self.effective_embed_dim)
            self.input_size = self.effective_embed_dim
        self._build_levels()

    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:
            # Input size depends on residual connections
            current_input_size = self.input_size
            if self.encoder_block:
                # EncoderBlock(s) para refinamento local (self-attention)
                encoder_blocks = nn.ModuleList()
                for encoder_index in range(self.encoder_layers):
                    encoder_blocks.append(
                        EncoderBlock(
                            embed_dim=current_input_size,
                            num_heads=self.encoder_heads[encoder_index],
                            ff_dim=self.encoder_ff_dim,
                            dropout=self.dropouts[encoder_index],
                        )
                    )

            classifier = BuildClassification(
                input_size=current_input_size,
                hidden_dims=self.hidden_dims[level_idx],
                output_size=self.levels_size[level_idx],
                num_layers=self.num_layers[level_idx],
                dropout=self.dropouts[level_idx],
                attention=self.attention and level_idx > 3,  # attention from level 4 onwards
                gcn=self.gcn,
                gat=self.gat,
                num_heads=self.num_heads,
                level=level_idx,
            )

            # Create classification network
            self.classifiers[str(level_idx)] = classifier
            if self.encoder_block:
                self.encoders[str(level_idx)] = encoder_blocks
            
            self.levels[level_idx] = {
                'encoder': encoder_blocks if self.encoder_block else None,
                'classifier': classifier
            }

            logging.info(
                "Level %d: input_size=%d, output_size=%d",
                level_idx, current_input_size, self.levels_size[level_idx]
            )

    def forward(
            self,
            x: torch.Tensor, 
            key_padding_mask: Optional[torch.Tensor] = None, 
            attn_mask: Optional[torch.Tensor] = None
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
                # === ENCODERBLOCK LOCAL (self-attention refinamento) ===
                encoder_out = current_input
                for encoder_block in level_module['encoder']:
                    encoder_out = encoder_block(encoder_out, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                    current_input = encoder_out  # output do último EncoderBlock é a entrada para o classificador

            # Forward through level
            level_output = level_module['classifier'](current_input, edge_index=self.edges_index)
            outputs[level_idx] = level_output

        return outputs

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

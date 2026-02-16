import os
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmc.models.base import HierarchicalModel
from hmc.models.local_classifier.networks import BuildClassification


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention for GO classification"""

    def __init__(self,
                d_model: int, 
                num_heads: int, 
                dropout: float = 0.2, 
                device = 'cuda',
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model).to(device)
        self.W_k = nn.Linear(d_model, d_model).to(device)
        self.W_v = nn.Linear(d_model, d_model).to(device)
        self.W_o = nn.Linear(d_model, d_model).to(device)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, d_model)
        returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        assert x.dim() == 3, f"esperado (batch, seq_len, d_model), veio {x.shape}"
        batch_size, seq_len, _ = x.shape

        # Projeções lineares
        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # (B, L, H, Dh) -> (B, H, L, Dh)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Atenção scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Contexto
        context = torch.matmul(attention_weights, V)  # (B, H, L, Dh)

        # Junta cabeças
        context = context.transpose(1, 2).contiguous()  # (B, L, H, Dh)
        context = context.view(batch_size, seq_len, self.d_model)  # (B, L, D)

        # Projeção final
        output = self.W_o(context)

        return output, attention_weights

class TabularMultiHeadAttention(nn.Module):
    """
    Atenção multi-head para vetores tabulares (batch, num_features).

    Passos:
    - Cada feature vira um "token" com embedding próprio (feature_embeddings).
    - O valor numérico da feature escala esse embedding.
    - Rodamos self-attention entre features.
    - Fazemos pooling sobre as features para obter um vetor por amostra.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = 'cuda',
        pooling: str = "mean",  # "mean", "sum" ou "flatten"
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        assert pooling in {"mean", "sum", "flatten"}

        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.pooling = pooling

        # Um embedding por feature (F, D)
        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features, d_model) * 0.01
        )

        # Bloco de atenção "normal" em cima dos tokens de features
        self.attention = MultiHeadAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        ).to(device)

        # Opcional: pequena MLP depois da atenção
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        ).to(device)

    @property
    def output_dim(self) -> int:
        if self.pooling == "flatten":
            return self.d_model * self.num_features
        else:
            return self.d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, num_features)
        returns:
            out: (batch_size, output_dim)
            attn_weights: (batch_size, num_heads, num_features, num_features)
        """
        assert x.dim() == 2, f"esperado (batch, num_features), veio {x.shape}"
        batch_size, num_features = x.shape
        assert num_features == self.num_features, (
            f"num_features={num_features} diferente do esperado {self.num_features}"
        )

        # Garante que embeddings estejam no mesmo device/dtype de x
        feature_emb = self.feature_embeddings.to(device=x.device, dtype=x.dtype)  # (F, D)

        # x: (B, F) -> (B, F, 1)
        x_expanded = x.unsqueeze(-1)  # (B, F, 1)

        # Embedding "gated" pelas features: (B, F, 1) * (1, F, D) -> (B, F, D)
        feature_emb = feature_emb.unsqueeze(0)  # (1, F, D)
        tokens = x_expanded * feature_emb       # (B, F, D)

        # Self-attention entre features
        attn_out, attn_weights = self.attention(tokens)  # (B, F, D), (B, H, F, F)

        # Pequena FFN + residual por token
        ffn_out = self.ffn(attn_out)  # (B, F, D)
        tokens = tokens + ffn_out     # residual

        # Pooling sobre as features para obter um vetor por amostra
        if self.pooling == "mean":
            out = tokens.mean(dim=1)  # (B, D)
        elif self.pooling == "sum":
            out = tokens.sum(dim=1)   # (B, D)
        else:  # "flatten"
            out = tokens.reshape(batch_size, -1)  # (B, F * D)

        return out, attn_weights


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
        encoder_block: bool = False,
        num_heads: int = 8,
        encoder_heads: Optional[List[int]] = None,  # heads por nível para EncoderBlock
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

        self.encoder_block = encoder_block
        self.num_heads = num_heads
        self.encoder_heads = encoder_heads or [num_heads // 2] * len(levels_size)  # default menor que GAT
        self.encoder_ff_dim = encoder_ff_dim
        self.encoder_layers = encoder_layers


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
        
        # if self.encoder_block:
        #     self.effective_embed_dim = _get_divisible_dim(self.input_size, self.num_heads)
        #     self.embed_proj = nn.Linear(self.input_size, self.effective_embed_dim)
        #     self.input_size = self.effective_embed_dim

        self._build_levels()

    # noinspection PyGlobalUndefined
    def _build_levels(self):
        """Build classification networks for each active level."""
        for level_idx in self.active_levels:
            # Input size depends on residual connections
            current_input_size = self.input_size
            if self.encoder_block:
                multihead_attention = TabularMultiHeadAttention(
                    num_features=current_input_size,
                    d_model = 32,
                    num_heads=4,
                    dropout=0.2,
                )
                level_classifier = nn.Linear(
                    multihead_attention.output_dim,
                    self.levels_size[level_idx],
                )

                self.levels[level_idx] = {
                    'multihead_attention': multihead_attention.to('cuda'),
                    'level_classifier': level_classifier.to('cuda')
                }

            else:
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
        attention_maps = []
        current_input = x
        if hasattr(self, 'embed_proj'):
            current_input = self.embed_proj(current_input)  # 529 → 536


        for level_idx, level_module in self.levels.items():
            level_idx = int(level_idx)

            # Load checkpoint if level is inactive
            if not self.level_active[level_idx]:
                self._load_checkpoint(level_idx)

            if self.encoder_block:
                # Multi-head attention
                attn_output, _ = level_module['multihead_attention'](current_input)

                # Classification for this level
                logits = level_module['level_classifier'](attn_output)
                outputs[level_idx] = logits

            else:
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

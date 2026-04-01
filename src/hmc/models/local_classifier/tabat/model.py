"""
This module contains the TabAT model implementation.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

from hmc.models.local_classifier.networks import BuildClassification


class TabularAttention(nn.Module):
    """
    Tabular Attention module.

    Args:
        num_features: Number of input features.
        embed_dim: Dimension of the embedding.
        num_heads: Number of attention heads.
    """

    def __init__(self, num_features: int, embed_dim: int = 64, num_heads: int = 1):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Projeção das features para um espaço de embedding (latente)
        # Cada feature bruta vira um vetor de tamanho embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)

        # 2. Multi-Head Attention padrão (PyTorch)
        # batch_first=True para lidar com (batch, seq, feature)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # 3. Layer Norm para estabilidade (essencial em redes profundas)
        self.ln = nn.LayerNorm(embed_dim)

        # 4. Projeção de saída para voltar ao tamanho original ou condensado
        self.output_proj = nn.Linear(num_features * embed_dim, num_features)

    def forward(self, x: torch.Tensor):
        """
        x shape: (batch_size, num_features)
        """
        batch_size = x.shape[0]

        # Step 1: Reshape para (batch, num_features, 1) para projetar cada feature individualmente
        x_reshaped = x.unsqueeze(-1)

        # Step 2: Embed features -> (batch, num_features, embed_dim)
        # Isso cria uma "sequência" onde cada elemento é uma feature do seu dataset
        h = self.feature_embed(x_reshaped)

        # Step 3: Self-Attention
        # attn_output: representação rica em contexto entre features
        # attn_weights: matriz de afinidade (batch, num_heads, num_features, num_features)
        attn_out, attn_weights = self.mha(h, h, h)

        # Step 4: Residual + Norm
        h = self.ln(h + attn_out)

        # Step 5: Flatten e Projeção Final
        # Transformamos os embeddings de volta em um vetor de contexto
        h_flat = h.view(batch_size, -1)
        context_vector = self.output_proj(h_flat)

        return context_vector, attn_weights


class TabATModel(nn.Module):
    """
    TabAT model implementation.

    Args:
        input_size: Number of input features.
        levels_size: Dictionary containing the number of classes for each level.
        num_layers: List containing the number of layers for each level.
        dropouts: List containing the dropout rates for each level.
        hidden_dims: List containing the hidden dimensions for each level.
        embed_dim: Dimension of the embedding.
        num_heads: Number of attention heads.
        pooling: Pooling method.
        device: Device to use for training.
    """

    def __init__(
        self,
        input_size: int,
        levels_size: Dict[str, int],
        num_layers: List[int],
        dropouts: List[float],
        hidden_dims: List[int],
        embed_dim: int = 64,
        num_heads: int = 8,
        pooling: str = "mean",
        device: str = "cpu",
    ):
        super().__init__()
        self.levels_size = levels_size  # Ex: {0: 10, 1: 50, 2: 100}
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.hidden_dims = hidden_dims
        self.level_active = [True] * len(self.levels_size)

        # Uma cabeça de atenção para cada nível da hierarquia
        self.attn_layers = nn.ModuleDict(
            {
                str(lvl): TabularAttention(
                    num_features=input_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                ).to(device)
                for lvl in levels_size
            }
        )

        # 3) Cabeças locais por nível GO
        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(levels_size.values()):
            level_classifier = BuildClassification(
                input_size=input_size * 2,
                output_size=num_classes,
                num_layers=self.num_layers[i],
                dropout=self.dropouts[i],
                hidden_dims=self.hidden_dims[i],
                level=i,
            ).to(device)
            self.heads.append(level_classifier)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for the TabAT model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Tuple of logits and attention weights.
        """
        logits = {}
        all_attn_weights = {}

        for lvl_idx, attn_layer in self.attn_layers.items():
            # 1. Atenção específica para este nível
            # Cada nível "olha" para o input de forma diferente
            context, weights = attn_layer(x)

            # 2. Concatena feature original com o contexto filtrado pela atenção
            combined = torch.cat([x, context], dim=1)

            # 3. Predição do nível
            logits[lvl_idx] = self.heads[int(lvl_idx)](combined)
            all_attn_weights[lvl_idx] = weights

        return logits

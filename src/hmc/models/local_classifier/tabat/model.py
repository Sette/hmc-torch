import math
import torch
from typing import Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from hmc.models.local_classifier.networks import BuildClassification


class TabularAttention(nn.Module):
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
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
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
    def __init__(self,
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
        self.attn_layers = nn.ModuleDict({
            str(lvl): TabularAttention(
                num_features=input_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
            ).to(device)
            for lvl in levels_size
        })

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

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

        return logits, all_attn_weights


# class MultiHeadAttentionLayer(nn.Module):
#     """Multi-head self-attention genérico: (batch, seq_len, d_model)."""

#     def __init__(self, d_model: int, num_heads: int, dropout: float = 0.2):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)

#         self.scale = math.sqrt(self.head_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: (batch_size, seq_len, d_model)
#         returns:
#             output: (batch_size, seq_len, d_model)
#             attention_weights: (batch_size, num_heads, seq_len, seq_len)
#         """
#         assert x.dim() == 3, f"esperado (batch, seq_len, d_model), veio {x.shape}"
#         batch_size, seq_len, _ = x.shape

#         # Projeções lineares
#         Q = self.W_q(x)  # (B, L, D)
#         K = self.W_k(x)
#         V = self.W_v(x)

#         # (B, L, H, Dh) -> (B, H, L, Dh)
#         Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # Atenção scaled dot-product
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
#         attention_weights = F.softmax(scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)

#         # Contexto
#         context = torch.matmul(attention_weights, V)  # (B, H, L, Dh)

#         # Junta cabeças
#         context = context.transpose(1, 2).contiguous()  # (B, L, H, Dh)
#         context = context.view(batch_size, seq_len, self.d_model)  # (B, L, D)

#         # Projeção final
#         output = self.W_o(context)

#         return output, attention_weights


# class TabularMultiHeadAttention(nn.Module):
#     """
#     Atenção multi-head para vetores tabulares (batch, input_size).

#     - Cada feature (coluna) vira um token com embedding próprio.
#     - O valor numérico escala o embedding (gating).
#     - Self-attention entre features (tipo TabTransformer/FT-Transformer).
#     """

#     def __init__(
#         self,
#         input_size: int,
#         d_model: int = 32,
#         num_heads: int = 4,
#         dropout: float = 0.2,
#         pooling: str = "mean",  # "mean", "sum" ou "flatten"
#     ):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
#         assert pooling in {"mean", "sum", "flatten"}

#         self.input_size = input_size
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.pooling = pooling

#         # Um embedding por feature (F, D)
#         self.feature_embeddings = nn.Parameter(
#             torch.randn(input_size, d_model) * 0.01
#         )

#         # Bloco de atenção em cima dos tokens de features
#         self.attention = MultiHeadAttentionLayer(
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#         )

#         # Pequena FFN por token + residual
#         self.ffn = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model * 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 2, d_model),
#         )

#     @property
#     def output_dim(self) -> int:
#         if self.pooling == "flatten":
#             return self.d_model * self.num_features
#         else:
#             return self.d_model

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: (batch_size, input_size)

#         returns:
#             out: (batch_size, output_dim)
#             attn_weights: (batch_size, num_heads, input_size, input_size)
#         """
#         assert x.dim() == 2, f"esperado (batch, input_size), veio {x.shape}"
#         batch_size, input_size = x.shape
#         assert input_size == self.input_size, (
#             f"input_size={input_size} diferente do esperado {self.input_size}"
#         )

#         # Garante device/dtype corretos
#         feat_emb = self.feature_embeddings.to(device=x.device, dtype=x.dtype)  # (F, D)

#         # x: (B, F) -> (B, F, 1)
#         x_expanded = x.unsqueeze(-1)  # (B, F, 1)

#         # Embedding "gated" pelas features: (B, F, 1) * (1, F, D) -> (B, F, D)
#         feat_emb = feat_emb.unsqueeze(0)  # (1, F, D)
#         tokens = x_expanded * feat_emb    # (B, F, D)

#         # Self-attention entre features
#         attn_out, attn_weights = self.attention(tokens)  # (B, F, D), (B, H, F, F)

#         # FFN + residual
#         ffn_out = self.ffn(attn_out)  # (B, F, D)
#         tokens = tokens + ffn_out     # (B, F, D)

#         # Pooling sobre as features
#         if self.pooling == "mean":
#             out = tokens.mean(dim=1)       # (B, D)
#         elif self.pooling == "sum":
#             out = tokens.sum(dim=1)        # (B, D)
#         else:  # "flatten"
#             out = tokens.reshape(batch_size, -1)  # (B, F*D)

#         return out, attn_weights

# class TabularLocalAttentionClassifier(nn.Module):
#     def __init__(
#         self,
#         input_size: int,          # dimensão do vetor de entrada
#         levels_size: Dict[str, int],   # ex.: {"go_l1": 20, "go_l2": 150, ...}
#         num_layers: List[int],
#         dropouts: List[float],
#         hidden_dims: List[int],
#         d_model: int = 64,
#         num_heads: int = 8,
#         attn_dropout: float = 0.2,
#         mlp_hidden_dim: int = 128,
#         mlp_dropout: float = 0.2,
#         pooling: str = "mean",
#     ):
#         super().__init__()
#         self.levels_size = levels_size
#         self.level_active = [True] * len(levels_size)
#         if num_layers is None:
#             num_layers = [2] * len(levels_size)
#         if dropouts is None:
#             dropouts = [0.0] * len(levels_size)
#         if hidden_dims is None:
#             hidden_dims = [64] * len(levels_size)
#         self.num_layers = num_layers
#         self.dropouts = dropouts
#         self.hidden_dims = hidden_dims
#         d_model = input_size * 2
#         if d_model % num_heads != 0:
#             d_model = d_model - (d_model % num_heads)

#         # 1) Bloco de atenção tabular (tipo TabTransformer)
#         self.tab_attn = TabularMultiHeadAttention(
#             input_size=input_size,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=attn_dropout,
#             pooling=pooling,
#         )

#         mlp_hidden_dim = input_size * 2

#         attn_out_dim = self.tab_attn.output_dim  # d_model ou d_model * num_features

#         # 2) MLP compartilhado para extrair representação global
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(input_size + attn_out_dim),
#             nn.Linear(input_size + attn_out_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(mlp_dropout),
#             nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(mlp_dropout),
#         )

#         # 3) Cabeças locais por nível GO
#         self.heads = nn.ModuleList()
#         for i, num_classes in enumerate(levels_size.values()):
#             level_classifier = BuildClassification(
#                 input_size=mlp_hidden_dim,
#                 output_size=num_classes,
#                 num_layers=self.num_layers[i],
#                 dropout=self.dropouts[i],
#                 hidden_dim=self.hidden_dims[i],
#                 level=i,
#             )
#             self.heads.append(level_classifier)

#     def forward(self, x: torch.Tensor, mode: str = "levels") -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
#         """
#         x: (batch_size, num_features)

#         returns:
#             logits: dict[level_name] -> (batch, num_classes_nivel)
#             attn_weights: (batch, num_heads, num_features, num_features)
#         """
#         h = x
#         attn_out = None
#         if mode == "attention" or mode == "levels":
#             logits = {
#                 level_idx: 0
#                 for level_idx in self.levels_size
#             }
#             # 1) Atenção entre features
#             attn_out, attn_weights = self.tab_attn(x)  # attn_out: (batch, attn_out_dim)

#         if mode == "levels" or mode == "attention":
#             # 2) Representação global compartilhada
#             combined = torch.cat([x, attn_out], dim=-1)
#             h = self.mlp(combined)  # (batch, mlp_hidden_dim)
#             # 3) Saídas por nível
#             logits = {
#                 level_idx: head(h)
#                 for level_idx, head in enumerate(self.heads)
#             }
#         return logits, attn_weights

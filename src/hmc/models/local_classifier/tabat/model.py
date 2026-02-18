import math
import torch
from typing import Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head self-attention genérico: (batch, seq_len, d_model)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

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
    Atenção multi-head para vetores tabulares (batch, input_size).

    - Cada feature (coluna) vira um token com embedding próprio.
    - O valor numérico escala o embedding (gating).
    - Self-attention entre features (tipo TabTransformer/FT-Transformer).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",  # "mean", "sum" ou "flatten"
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        assert pooling in {"mean", "sum", "flatten"}

        self.input_size = input_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.pooling = pooling

        # Um embedding por feature (F, D)
        self.feature_embeddings = nn.Parameter(
            torch.randn(input_size, d_model) * 0.01
        )

        # Bloco de atenção em cima dos tokens de features
        self.attention = MultiHeadAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Pequena FFN por token + residual
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    @property
    def output_dim(self) -> int:
        if self.pooling == "flatten":
            return self.d_model * self.num_features
        else:
            return self.d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, input_size)

        returns:
            out: (batch_size, output_dim)
            attn_weights: (batch_size, num_heads, input_size, input_size)
        """
        assert x.dim() == 2, f"esperado (batch, input_size), veio {x.shape}"
        batch_size, input_size = x.shape
        assert input_size == self.input_size, (
            f"input_size={input_size} diferente do esperado {self.input_size}"
        )

        # Garante device/dtype corretos
        feat_emb = self.feature_embeddings.to(device=x.device, dtype=x.dtype)  # (F, D)

        # x: (B, F) -> (B, F, 1)
        x_expanded = x.unsqueeze(-1)  # (B, F, 1)

        # Embedding "gated" pelas features: (B, F, 1) * (1, F, D) -> (B, F, D)
        feat_emb = feat_emb.unsqueeze(0)  # (1, F, D)
        tokens = x_expanded * feat_emb    # (B, F, D)

        # Self-attention entre features
        attn_out, attn_weights = self.attention(tokens)  # (B, F, D), (B, H, F, F)

        # FFN + residual
        ffn_out = self.ffn(attn_out)  # (B, F, D)
        tokens = tokens + ffn_out     # (B, F, D)

        # Pooling sobre as features
        if self.pooling == "mean":
            out = tokens.mean(dim=1)       # (B, D)
        elif self.pooling == "sum":
            out = tokens.sum(dim=1)        # (B, D)
        else:  # "flatten"
            out = tokens.reshape(batch_size, -1)  # (B, F*D)

        return out, attn_weights


class LocalAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,          # dimensão do vetor de entrada
        levels_size: Dict[str, int],   # ex.: {"go_l1": 20, "go_l2": 150, ...}
        d_model: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.2,
        mlp_hidden_dim: int = 128,
        mlp_dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        # Create level modules
        self.levels = {}  # dict {level_idx: {'encoder': ModuleList, 'level_classifier': BuildClassification}}
        self.levels_size = levels_size
        self.level_active = [True] * len(levels_size)
        
        # 1) Bloco de atenção tabular (tipo TabTransformer)
        self.tab_attn = TabularMultiHeadAttention(
            input_size=input_size,
            d_model=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            pooling=pooling,
        )

        attn_out_dim = self.tab_attn.output_dim  # d_model ou d_model * num_features

        # 2) MLP compartilhado para extrair representação global
        self.mlp = nn.Sequential(
            nn.LayerNorm(attn_out_dim),
            nn.Linear(attn_out_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
        )

        # 3) Cabeças locais por nível GO
        self.heads = nn.ModuleList()
        for num_classes in levels_size.values():
            level_classifier = nn.Linear(mlp_hidden_dim, num_classes)
            self.heads.append(level_classifier)


    def forward(self, x: torch.Tensor, mode: str = "levels") -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        x: (batch_size, num_features)

        returns:
            go_logits: dict[level_name] -> (batch, num_classes_nivel)
            attn_weights: (batch, num_heads, num_features, num_features)
        """
        h = x
        if mode == "attention":
            logits = {
                    level_idx: 0
                    for level_idx in self.levels_size
                }
            # 1) Atenção entre features
            attn_out, attn_weights = self.tab_attn(x)  # attn_out: (batch, attn_out_dim)

            # 2) Representação global compartilhada
            h = self.mlp(attn_out)  # (batch, mlp_hidden_dim)

        if mode == "levels":
            # 3) Saídas por nível
            logits = {
                level_idx: head(h)
                for level_idx, head in enumerate(self.heads)
            }
        return logits, attn_weights

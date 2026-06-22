"""Local classifier: one MLP per hierarchy level.

Two variants:
  - LocalModel: frozen transformer embeddings + per-level MLPs
  - LocalE2EModel: fine-tuned transformer + per-level MLPs (no R-matrix)
"""

import torch
import torch.nn as nn

_CLS_POOL_MODELS = ("specter",)


class LocalE2EModel(nn.Module):
    """Fine-tuned transformer with per-level MLP heads — no R-matrix.

    Architecture:
        Transformer (fine-tuned) → CLS / mean-pool → shared embedding
          ├── MLP head level 0
          └── MLP head level 1

    Trained end-to-end with per-level BCE loss.
    """

    def __init__(
        self,
        model_name: str,
        levels_size: dict,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        self.levels_size = levels_size

        from transformers import AutoModel  # pylint: disable=import-outside-toplevel

        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad_(False)

        embed_dim = self.transformer.config.hidden_size
        self._cls_pool = any(k in model_name.lower() for k in _CLS_POOL_MODELS)

        self.heads = nn.ModuleDict()
        for lvl, n_classes in sorted(levels_size.items()):
            layers = []
            in_dim = embed_dim
            for _ in range(num_layers):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_classes))
            self.heads[str(lvl)] = nn.Sequential(*layers)

    def _pool(self, out, mask):
        if self._cls_pool:
            return out.last_hidden_state[:, 0, :]
        return (out.last_hidden_state * mask.unsqueeze(-1).float()).sum(1) / mask.sum(
            1
        ).float().clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, **_kw):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        emb = self._pool(out, attention_mask)
        return {lvl: torch.sigmoid(head(emb)) for lvl, head in self.heads.items()}


import torch
import torch.nn as nn


class LevelMLP(nn.Module):
    """Single-level MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[
            non_lin
        ]

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class LocalModel(nn.Module):
    """One MLP per hierarchy level, trained jointly.

    Args:
        input_dim: Feature dimension (e.g. 768 for SPECTER2).
        levels_size: dict {level_idx: n_classes}.
        hidden_dim: Hidden size for each level's MLP.
        num_layers: Number of hidden layers per MLP.
        dropout: Dropout probability.
        non_lin: Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        levels_size: dict,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        non_lin: str = "relu",
    ):
        super().__init__()
        self.levels_size = levels_size
        self.max_depth = len(levels_size)

        self.classifiers = nn.ModuleDict()
        for level, n_classes in sorted(levels_size.items()):
            self.classifiers[str(level)] = LevelMLP(
                input_dim=input_dim,
                output_dim=n_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                non_lin=non_lin,
            )

    def forward(self, x: torch.Tensor) -> dict:
        """Return {level_str: tensor(N, n_classes_level)}."""
        return {lvl: clf(x) for lvl, clf in self.classifiers.items()}

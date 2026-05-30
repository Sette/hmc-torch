"""End-to-end transformer models for hierarchical classification (globalE2E / globalSOTA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from hmc.models.global_classifier.constraint.utils import get_constr_out

# Models that encode document meaning in the [CLS] token rather than mean-pool.
_CLS_POOL_MODELS = ("specter",)


class E2EConstrainedModel(nn.Module):
    """HuggingFace transformer fine-tuned end-to-end with R-matrix constraint.

    Architecture:
        Transformer → CLS / mean-pool embedding → MLP head → sigmoid
        Inference only: get_constr_out() enforces hierarchy constraints.

    Discriminative fine-tuning is handled by the optimizer in the training loop
    (small LR for transformer layers, larger LR for the MLP head).
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        r_matrix: torch.Tensor,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        freeze_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad_(False)

        embed_dim: int = self.transformer.config.hidden_size
        self._cls_pool: bool = any(k in model_name.lower() for k in _CLS_POOL_MODELS)

        layers: list = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, output_dim), nn.Sigmoid()]
        self.classifier = nn.Sequential(*layers)

        self.register_buffer("r_matrix", r_matrix)

    def _pool(self, transformer_out, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._cls_pool:
            return transformer_out.last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        return (transformer_out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(
            min=1e-9
        )

    def head_parameters(self):
        return self.classifier.parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        emb = self._pool(out, attention_mask)
        scores = self.classifier(emb)
        if not self.training:
            scores = get_constr_out(scores, self.r_matrix)
        return scores


class E2EGNNModel(nn.Module):
    """HuggingFace transformer + label-hierarchy GCN for HMC (globalSOTA).

    Architecture (HiAGM-style):
        Text encoder  : fine-tuned transformer → CLS/mean-pool → hidden_dim
        Label encoder : 2-layer GCN on label hierarchy → hidden_dim per node
        Prediction    : sigmoid(doc_emb @ label_emb.T)
        Inference     : get_constr_out() enforces hierarchy constraints

    Compared with E2EConstrainedModel (globalE2E), the label representations
    are graph-aware — adjacent nodes in the hierarchy share information — which
    consistently improves F1 on datasets with rich hierarchy structure.
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        r_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        freeze_transformer: bool = False,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import (  # pylint: disable=import-outside-toplevel
            GCNConv,
        )

        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad_(False)

        embed_dim: int = self.transformer.config.hidden_size
        self._cls_pool: bool = any(k in model_name.lower() for k in _CLS_POOL_MODELS)

        # Document projection: embed_dim → hidden_dim
        proj_layers: list = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            proj_layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.doc_proj = nn.Sequential(*proj_layers)

        # Label hierarchy encoder: learnable embeddings + 2-layer GCN
        self.label_emb = nn.Embedding(output_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.register_buffer("r_matrix", r_matrix)
        self.register_buffer("edge_index", edge_index)

    def _pool(self, transformer_out, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._cls_pool:
            return transformer_out.last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        return (transformer_out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(
            min=1e-9
        )

    def _label_representations(self) -> torch.Tensor:
        x = self.label_emb.weight
        x = F.relu(self.gcn1(x, self.edge_index))
        x = self.drop(x)
        x = self.gcn2(x, self.edge_index)
        return x  # (N, hidden_dim)

    def head_parameters(self):
        import itertools  # pylint: disable=import-outside-toplevel

        return itertools.chain(
            self.doc_proj.parameters(),
            self.label_emb.parameters(),
            self.gcn1.parameters(),
            self.gcn2.parameters(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        doc_emb = self.doc_proj(self._pool(out, attention_mask))  # (B, hidden_dim)
        label_emb = self._label_representations()  # (N, hidden_dim)
        scores = torch.sigmoid(doc_emb @ label_emb.T)  # (B, N)
        if not self.training:
            scores = get_constr_out(scores, self.r_matrix)
        return scores

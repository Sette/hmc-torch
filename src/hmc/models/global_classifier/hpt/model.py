"""Hierarchy-aware prompt tuning (HPT, Wang et al., EMNLP'22) for the global pipeline.

The document is followed by one virtual level token and one ``[PRED]`` slot per
hierarchy level::

    [CLS] text ... V_0 [PRED] V_1 [PRED] ... V_{L-1} [PRED] [SEP]

Every label is a new vocabulary row whose embedding starts as the mean word
embedding of the label's name.  Label and level rows go through a GAT over the
taxonomy on every forward pass (edges parent -> child and level -> label);
the graph output replaces the level-token inputs.  The MLM head scores every
label at the ``[PRED]`` slot of its own level against the raw label rows (the
verbalizer), giving one logit per label.  An auxiliary MLM loss on the text
is computed during training.

Differences from the reference code (``HPT/models/prompt.py``):

* ``graph_direction="both"`` (default) also adds the reverse edges.  With
  HPT's one-way edges a level token has no incoming edge, so the GAT sees only
  its self-loop and the hierarchy never reaches the encoder input;
  ``"down"`` keeps HPT's behaviour.
* Label names are embedded without the ``[CLS]``/``[SEP]`` tokens; a label
  without a name starts as the mean of its children.
* The MLM loss is over the original vocabulary only.
* The R-matrix is applied by the pipeline (losses and reconciliation), not
  here: the model returns raw label logits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GraphLayer(nn.Module):
    """GAT + dropout + residual + LayerNorm (HPT's ``GraphLayer``, GAT branch)."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gat = GATConv(hidden, hidden, heads=1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One residual GAT step over the label/level graph."""
        h = F.dropout(self.gat(x, edge_index), p=self.dropout, training=self.training)
        return self.norm(x + h)


class HierarchyCrossAttentionHead(nn.Module):
    """Cross-attention + FFN scoring head over GAT-contextualized hierarchy nodes (Spec 10 M2)."""

    def __init__(self, hidden: int, dropout: float, heads: int = 4) -> None:
        super().__init__()
        n_heads = heads if hidden % heads == 0 else 1
        self.attn = nn.MultiheadAttention(
            hidden, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, h_level: torch.Tensor, graph_nodes: torch.Tensor) -> torch.Tensor:
        """Contextualize level [PRED] states (B, L, H) against hierarchy nodes (N+L, H)."""
        kv = graph_nodes.to(h_level.dtype).unsqueeze(0).expand(h_level.shape[0], -1, -1)
        attn_out, _ = self.attn(h_level, kv, kv, need_weights=False)
        h = self.norm1(h_level + attn_out)
        return self.norm2(h + self.ffn(h))


def build_edge_index(
    edges: np.ndarray, label_depth: Sequence[int], direction: str = "both"
) -> torch.Tensor:
    """Graph over ``N`` label nodes followed by ``L`` level nodes.

    ``edges`` holds ``(parent, child)`` pairs in label-index space.  Every
    label is linked from the level node of its depth.  ``direction="both"``
    adds the reverse of every edge.
    """
    n = len(label_depth)
    src = [int(p) for p, _ in edges] + [n + int(d) for d in label_depth]
    dst = [int(c) for _, c in edges] + list(range(n))
    if direction == "both":
        src, dst = src + dst, dst + src
    elif direction != "down":
        raise ValueError(f"graph direction must be 'both' or 'down', got {direction!r}")
    return torch.tensor([src, dst], dtype=torch.long)


class HPTPromptModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Masked-LM prompt classifier with graph-encoded label and level tokens.

    Args:
        mlm: a BERT-family ``*ForMaskedLM`` (exposes ``base_model`` and
            ``cls.predictions.transform``).
        tokenizer: its tokenizer.
        label_texts: text used to initialise each label row (``None`` = mean
            of the children, or the code as a last resort; see ``label_codes``).
        label_codes: fallback text per label.
        label_depth: 0-based level of every label (the ``[PRED]`` slot that
            scores it).
        edges: ``(parent, child)`` label-index pairs.
        graph_layers: number of GAT layers (HPT: 1).
        graph_direction: ``"both"`` or ``"down"`` (HPT); see the module doc.
        use_mlm: add the auxiliary MLM loss in training.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mlm: nn.Module,
        tokenizer,
        label_texts: Sequence[str | None],
        label_codes: Sequence[str],
        label_depth: Sequence[int],
        edges: np.ndarray,
        graph_layers: int = 1,
        graph_direction: str = "both",
        use_mlm: bool = True,
        label_encoder: str = "mean",
        label_prompt_tokens: int = 1,
        score_head: str = "verbalizer",
    ) -> None:
        super().__init__()
        if label_encoder not in ("mean", "sentence", "random"):
            raise ValueError(
                f"label_encoder must be 'mean', 'sentence', or 'random', got {label_encoder!r}"
            )
        if label_prompt_tokens < 1:
            raise ValueError(
                f"label_prompt_tokens must be >= 1, got {label_prompt_tokens}"
            )
        if score_head not in ("verbalizer", "crossattn"):
            raise ValueError(
                f"score_head must be 'verbalizer' or 'crossattn', got {score_head!r}"
            )
        self.mlm = mlm
        self.tokenizer = tokenizer
        self.use_mlm = use_mlm
        self.label_encoder = label_encoder
        self.label_prompt_tokens = int(label_prompt_tokens)
        self.score_head = score_head
        config = mlm.config
        hidden = config.hidden_size
        word = mlm.get_input_embeddings()
        self.vocab_size = word.num_embeddings
        self.n_labels = len(label_depth)
        self.n_levels = int(max(label_depth)) + 1
        k_tok = self.label_prompt_tokens
        # Prompt-token ids live past the vocabulary:
        # level i, slot k -> V + i*K + k; [PRED] -> V + L*K.
        self.n_virtual = self.n_levels * k_tok
        self.pred_id = self.vocab_size + self.n_virtual
        self.prompt_ids = [
            t
            for i in range(self.n_levels)
            for t in (
                *(self.vocab_size + i * k_tok + k for k in range(k_tok)),
                self.pred_id,
            )
        ] + [tokenizer.sep_token_id]
        self._special_ids = torch.tensor(
            sorted(set(tokenizer.all_special_ids)), dtype=torch.long
        )

        with torch.no_grad():
            init = _label_init(
                word.weight,
                tokenizer,
                label_texts,
                label_codes,
                edges,
                label_encoder=label_encoder,
                base_model=mlm.base_model,
                init_range=config.initializer_range,
            )
        self.label_emb = nn.Parameter(init)
        self.level_emb = nn.Parameter(
            torch.randn(self.n_levels, hidden) * config.initializer_range
        )
        self.pred_emb = nn.Parameter(
            word.weight[tokenizer.mask_token_id].detach().clone().unsqueeze(0)
        )
        bound = 1.0 / hidden**0.5
        self.label_bias = nn.Parameter(
            torch.empty(self.n_labels).uniform_(-bound, bound)
        )
        self.graph = nn.ModuleList(
            GraphLayer(hidden, config.attention_probs_dropout_prob)
            for _ in range(graph_layers)
        )
        if k_tok > 1:
            self.prompt_token_offset = nn.Parameter(
                torch.randn(k_tok, hidden) * config.initializer_range
            )
        else:
            self.register_parameter("prompt_token_offset", None)
        if score_head == "crossattn":
            self.cross_head: HierarchyCrossAttentionHead | None = (
                HierarchyCrossAttentionHead(hidden, config.attention_probs_dropout_prob)
            )
        else:
            self.cross_head = None
        self.register_buffer(
            "edge_index",
            build_edge_index(edges, label_depth, graph_direction),
            persistent=False,
        )
        self.register_buffer(
            "label_depth",
            torch.as_tensor(label_depth, dtype=torch.long),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls, model_name: str, model_cache_dir: str = "./models", **kwargs
    ) -> HPTPromptModel:
        """Load ``model_name`` as a masked LM from the project model cache.

        Checkpoints without a pretrained MLM head (SPECTER2) get a randomly
        initialised transform; the decoder stays tied to the word embeddings.
        """
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForMaskedLM,
            AutoTokenizer,
        )

        from hmc.utils.model_cache import (  # pylint: disable=import-outside-toplevel
            ensure_transformer_model_cached,
        )

        path = ensure_transformer_model_cached(
            model_name, model_cache_dir, auto_class="AutoModelForMaskedLM"
        )
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mlm = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
        return cls(mlm, tokenizer, **kwargs)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def prompt_rows(
        self, texts: Sequence[str], max_length: int = 512
    ) -> list[list[int]]:
        """Token ids of each text followed by the prompt, at most ``max_length``."""
        room = max_length - len(self.prompt_ids) + 1  # the text's own [SEP] is dropped
        rows = self.tokenizer(list(texts), truncation=True, max_length=room)[
            "input_ids"
        ]
        return [r[:-1] + self.prompt_ids for r in rows]

    def collate(self, rows: Sequence[Sequence[int]]) -> dict:
        """Right-pad ``prompt_rows`` output into ``input_ids``/``attention_mask``."""
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def graph_nodes(self) -> torch.Tensor:
        """Label and level rows after the GAT, ``(N + L, H)``."""
        x = torch.cat([self.label_emb, self.level_emb], 0)
        for layer in self.graph:
            x = layer(x, self.edge_index)
        return x

    def _mask_text(self, ids: torch.Tensor, attention_mask: torch.Tensor):
        """HPT's MLM corruption: 15% of text tokens are targets (80/10/10-like)."""
        candidate = (
            (ids < self.vocab_size)
            & attention_mask.bool()
            & ~torch.isin(ids, self._special_ids.to(ids.device))
        )
        u = torch.rand(ids.shape, device=ids.device)
        target = candidate & (u >= 0.85)
        corrupted = ids.masked_fill(target & (u >= 0.865), self.tokenizer.mask_token_id)
        swap = target & (u >= 0.985)
        random_ids = torch.randint_like(ids, 0, self.vocab_size)
        corrupted = torch.where(swap, random_ids, corrupted)
        return corrupted, target

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(label logits (B, N), MLM loss or None)``."""
        target = None
        ids = input_ids
        if self.training and self.use_mlm:
            ids, target = self._mask_text(input_ids, attention_mask)

        word = self.mlm.get_input_embeddings()
        prompt = ids >= self.vocab_size
        embeds = word(ids.masked_fill(prompt, 0))
        nodes = self.graph_nodes()
        level_nodes = nodes[self.n_labels :]
        if self.prompt_token_offset is not None:
            level_nodes = (
                level_nodes.unsqueeze(1) + self.prompt_token_offset.unsqueeze(0)
            ).reshape(self.n_virtual, -1)
        table = torch.cat([level_nodes, self.pred_emb], 0).to(embeds.dtype)
        prompt_rows = table[(ids - self.vocab_size).clamp(0, self.n_virtual)]
        embeds = torch.where(prompt.unsqueeze(-1), prompt_rows, embeds)

        hidden = self.mlm.base_model(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state

        # One [PRED] per level, in order: (B, L, H) -> scores (B, L, N).
        batch = ids.shape[0]
        slots = (ids == self.pred_id).nonzero()[:, 1].view(batch, self.n_levels)
        h = hidden.gather(1, slots.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        if self.cross_head is not None:
            h = self.cross_head(h, nodes)
            label_proto = nodes[: self.n_labels]
        else:
            label_proto = self.label_emb
        h = self.mlm.cls.predictions.transform(h)
        scores = h @ label_proto.to(h.dtype).T + self.label_bias.to(h.dtype)
        depth = self.label_depth.view(1, 1, -1).expand(batch, 1, -1)
        logits = scores.gather(1, depth).squeeze(1)

        mlm_loss = None
        if target is not None:
            if target.any():
                mlm_logits = self.mlm.cls(hidden[target])
                mlm_loss = F.cross_entropy(mlm_logits.float(), input_ids[target])
            else:
                mlm_loss = logits.new_zeros((), dtype=torch.float32)
        return logits, mlm_loss


def _label_init(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    word_weight: torch.Tensor,
    tokenizer,
    label_texts: Sequence[str | None],
    label_codes: Sequence[str],
    edges: np.ndarray,
    label_encoder: str = "mean",
    base_model: nn.Module | None = None,
    init_range: float = 0.02,
) -> torch.Tensor:
    """Initialise each label row via mean word embeddings, contextual sentence encoding, or random.

    Labels without a name take the mean of their children's rows (deepest
    first, so a chain of unnamed nodes resolves bottom-up); childless unnamed
    labels fall back to their code.
    """
    n = len(label_codes)
    if label_encoder == "random":
        return (
            torch.randn(
                n, word_weight.shape[1], device=word_weight.device, dtype=torch.float32
            )
            * init_range
        )

    def embed(text: str) -> torch.Tensor:
        if label_encoder == "sentence" and base_model is not None:
            tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            tok = {k: v.to(word_weight.device) for k, v in tok.items()}
            out = base_model(**tok).last_hidden_state[0]
            if out.shape[0] > 2:
                return out[1:-1].mean(0).detach()
            return out.mean(0).detach()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"] or [
            tokenizer.unk_token_id
        ]
        return word_weight[ids].mean(0)

    rows: list[torch.Tensor | None] = [embed(t) if t else None for t in label_texts]
    children: dict[int, list[int]] = {}
    for p, c in edges:
        children.setdefault(int(p), []).append(int(c))
    pending = [i for i in range(n) if rows[i] is None]
    while pending:
        progressed = False
        for i in list(pending):
            kids = children.get(i, [])
            if not kids:
                rows[i] = embed(label_codes[i])
            elif all(rows[k] is not None for k in kids):
                rows[i] = torch.stack([rows[k] for k in kids]).mean(0)
            else:
                continue
            pending.remove(i)
            progressed = True
        if not progressed:  # cycle among unnamed nodes: use the codes
            for i in pending:
                rows[i] = embed(label_codes[i])
            pending = []
    return torch.stack(rows).float()

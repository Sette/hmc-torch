"""Label-embedding heads: score each class by a dot product with a class vector.

A standard sigmoid head learns one weight vector per class independently.
:class:`LabelEmbeddingHead` instead builds the class vectors from knowledge
about the labels, so that related classes share parameters:

* ``"free"``  -- one learned vector per class (equivalent to a linear head);
* ``"graph"`` -- learned vectors smoothed by a two-layer GCN over the label
  hierarchy, so a class borrows strength from its parents and children;
* ``"text"``  -- per-class vectors plus a projection of frozen text embeddings
  of each class's name and definition;
* ``"text+graph"`` -- the text vectors, then the GCN;
* ``"gae"`` -- Surj-style (Yang & Howe, WWW'22): a projection of frozen
  graph-autoencoder embeddings (:func:`~hmc.models.hierarchical.graph_autoencoder.pretrain_label_gae`),
  scored by scaled cosine similarity instead of a dot product;
* ``"gae+free"`` -- the same, plus a learned per-class residual;
* ``"gae-init"`` -- the ``"free"`` head with its class vectors initialized
  from a projection of the graph-autoencoder embeddings;
* ``"gae-reg"`` -- the ``"free"`` head with its class vectors pulled towards
  a learned projection of the graph-autoencoder embeddings
  (:meth:`LabelEmbeddingHead.extra_loss`);
* ``"gae-2stage"`` -- Surj's two stages taken literally: the sample encoder
  is first trained alone to regress the mean graph-autoencoder embedding of
  each sample's labels (:func:`pretrain_two_stage`), then frozen; only the
  cosine scale and the per-class biases are trained on the classification
  loss.

The graph autoencoder smooths neighbours, so siblings end up with nearly
identical embeddings; the last two modes use it as a prior on the class
vectors instead of as the class vectors themselves.

Text embeddings for Gene Ontology terms come from :func:`go_term_texts`.
"""

from __future__ import annotations

import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn

GO_OBO_URL = "http://purl.obolibrary.org/obo/go.obo"
MODES = (
    "free",
    "graph",
    "text",
    "text+graph",
    "gae",
    "gae+free",
    "gae-init",
    "gae-reg",
    "gae-2stage",
)


def normalized_adjacency(edges: np.ndarray, n: int) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops, D^-1/2 (A + I) D^-1/2.

    Args:
        edges: ``(E, 2)`` array of (child, parent) column indices.
        n: number of classes.
    """
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    vals = 1.0 / np.sqrt(deg[rows] * deg[cols])
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals), (n, n)
    ).coalesce()


class LabelEmbeddingHead(nn.Module):  # pylint: disable=too-many-instance-attributes
    """MLP sample encoder + class vectors built from label knowledge.

    Args:
        input_dim: feature dimension.
        hidden_dim: width of the sample encoder and of the class vectors.
        output_dim: number of classes.
        mode: one of :data:`MODES`.
        adjacency: normalized label adjacency (required for graph modes).
        text_embeddings: ``(output_dim, T)`` frozen embeddings (text modes).
        label_features: ``(output_dim, K)`` frozen graph-autoencoder
            embeddings (gae modes).
        dropout: dropout of the sample encoder.
        prior_weight: weight of the ``"gae-reg"`` penalty.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mode: str = "free",
        adjacency: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        label_features: torch.Tensor | None = None,
        dropout: float = 0.3,
        prior_weight: float = 1e-3,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
        if "graph" in mode and adjacency is None:
            raise ValueError(f"mode {mode!r} needs an adjacency matrix")
        if "text" in mode and text_embeddings is None:
            raise ValueError(f"mode {mode!r} needs text embeddings")
        if "gae" in mode and label_features is None:
            raise ValueError(f"mode {mode!r} needs label features")
        self.mode = mode
        self.cosine = mode in ("gae", "gae+free", "gae-2stage")
        self.frozen = False
        self.prior_weight = prior_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Per-class vectors, as in a plain linear head.  The knowledge
        # branches are added through scalar gates that start at zero, so the
        # model starts out identical to the "free" head and only uses the
        # label text or the hierarchy if that lowers the training loss.
        self.class_vectors = nn.Parameter(torch.empty(output_dim, hidden_dim))
        nn.init.normal_(self.class_vectors, std=hidden_dim**-0.5)
        if "text" in mode:
            self.register_buffer("text", text_embeddings.float())
            self.text_proj = nn.Linear(text_embeddings.shape[1], hidden_dim)
            self.text_gate = nn.Parameter(torch.zeros(()))
        if "graph" in mode:
            self.register_buffer("adjacency", adjacency.float())
            self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
            self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
            self.graph_gate = nn.Parameter(torch.zeros(()))
        if "gae" in mode:
            self.register_buffer("label_features", label_features.float())
        if "gae" in mode and mode != "gae-2stage":
            self.label_proj = nn.Linear(label_features.shape[1], hidden_dim)
        if mode == "gae-2stage":
            # Samples are mapped into the autoencoder space itself.
            self.sample_proj = nn.Linear(hidden_dim, label_features.shape[1])
        if mode == "gae-init":
            with torch.no_grad():
                prior = self.label_proj(self.label_features.cpu())
                self.class_vectors.copy_(prior * hidden_dim**-0.5 / prior.std())
        if self.cosine:
            # Cosine scores live in [-1, 1]; a learned scale (init 10) lets the
            # logits reach the range BCE needs.
            self.log_scale = nn.Parameter(
                torch.tensor(np.log(10.0), dtype=torch.float32)
            )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def label_embeddings(self) -> torch.Tensor:
        """The ``(output_dim, hidden_dim)`` class vectors."""
        if self.mode == "gae-2stage":
            return self.label_features
        if self.cosine:
            emb = self.label_proj(self.label_features)
            return emb + self.class_vectors if self.mode == "gae+free" else emb
        emb = self.class_vectors
        if "text" in self.mode:
            emb = emb + self.text_gate * self.text_proj(self.text)
        if "graph" in self.mode:
            h = torch.relu(self.adjacency @ self.gcn1(emb))
            emb = emb + self.graph_gate * (self.adjacency @ self.gcn2(h))
        return emb

    def extra_loss(self) -> torch.Tensor:
        """Penalty added to the training loss (non-zero only for ``"gae-reg"``)."""
        if self.mode != "gae-reg":
            return self.bias.new_zeros(())
        prior = self.label_proj(self.label_features)
        return self.prior_weight * (self.class_vectors - prior).pow(2).sum(1).mean()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-norm sample embeddings, in the space the classes are scored in."""
        h = self.encoder(x)
        if self.mode == "gae-2stage":
            h = self.sample_proj(h)
        return nn.functional.normalize(h, dim=1)

    def freeze_encoder(self) -> None:
        """Freeze the sample encoder (stage 2 of ``"gae-2stage"``); it stays in eval mode."""
        self.frozen = True
        for module in (self.encoder, getattr(self, "sample_proj", None)):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True) -> LabelEmbeddingHead:
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities, ``(batch, output_dim)``."""
        if self.cosine:
            h = self.embed(x)
            emb = nn.functional.normalize(self.label_embeddings(), dim=1)
            return torch.sigmoid(self.log_scale.exp() * (h @ emb.T) + self.bias)
        return torch.sigmoid(self.encoder(x) @ self.label_embeddings().T + self.bias)


def pretrain_two_stage(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    head: LabelEmbeddingHead,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    batch: int = 32,
    max_epochs: int = 200,
    patience: int = 10,
) -> int:
    """Stage 1 of ``"gae-2stage"``: regress each sample's label-set embedding.

    The target of a sample is the normalized mean of the autoencoder
    embeddings of its positive labels; the loss is ``1 - cos``.  Early
    stopping on the validation loss; the best encoder is kept and frozen.

    Returns:
        int: the selected number of epochs.
    """
    device = head.label_features.device
    z = head.label_features

    def targets(labels):
        t = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device) @ z
        return nn.functional.normalize(t, dim=1)

    x_t = torch.from_numpy(x).to(device)
    t_t = targets(y)
    x_v, t_v = torch.from_numpy(x_val).to(device), targets(y_val)
    params = list(head.encoder.parameters()) + list(head.sample_proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(len(x_t), device=device)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            loss = (1 - (head.embed(x_t[idx]) * t_t[idx]).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val = float((1 - (head.embed(x_v) * t_v).sum(1)).mean())
        if val < best:
            best, best_epoch, bad = val, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.freeze_encoder()
    return best_epoch


def load_go_obo(path: str) -> dict[str, str]:
    """Map every GO id (including ``alt_id``s) to ``"name: definition"``.

    Downloads the ontology from :data:`GO_OBO_URL` if ``path`` does not exist.
    Obsolete terms keep their text, without the ``"obsolete "`` prefix.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urlretrieve(GO_OBO_URL, path)
    texts: dict[str, str] = {}
    term: dict[str, str] = {}
    aliases: list[str] = []
    in_term = False

    def flush():
        if "id" in term:
            text = term.get("name", "").removeprefix("obsolete ")
            if term.get("def"):
                text = f"{text}: {term['def']}"
            for key in [term["id"], *aliases]:
                texts[key] = text

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush()
                term, aliases, in_term = {}, [], line == "[Term]"
            elif in_term and ": " in line:
                key, value = line.split(": ", 1)
                if key in ("id", "name"):
                    term[key] = value
                elif key == "def":
                    term["def"] = value.split('" [')[0].strip('"')
                elif key == "alt_id":
                    aliases.append(value)
    flush()
    return texts


def go_term_texts(node_names: list[str], obo_path: str) -> list[str]:
    """Text for each class of a GO dataset (``"GO0003674"``-style names).

    Nodes that are not GO terms (the root) get their own name.
    """
    texts = load_go_obo(obo_path)
    out = []
    for name in node_names:
        go_id = f"GO:{name[2:]}" if name.startswith("GO") and ":" not in name else name
        out.append(texts.get(go_id, name))
    return out

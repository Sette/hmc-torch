"""Label-embedding heads: score each class by a dot product with a class vector.

A standard sigmoid head learns one weight vector per class independently.
:class:`LabelEmbeddingHead` instead builds the class vectors from knowledge
about the labels, so that related classes share parameters:

* ``"free"``  -- one learned vector per class (equivalent to a linear head);
* ``"graph"`` -- learned vectors smoothed by a two-layer GCN over the label
  hierarchy, so a class borrows strength from its parents and children;
* ``"text"``  -- per-class vectors plus a projection of frozen text embeddings
  of each class's name and definition;
* ``"text+graph"`` -- the text vectors, then the GCN.

Text embeddings for Gene Ontology terms come from :func:`go_term_texts`.
"""

from __future__ import annotations

import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn

GO_OBO_URL = "http://purl.obolibrary.org/obo/go.obo"
MODES = ("free", "graph", "text", "text+graph")


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
        dropout: dropout of the sample encoder.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mode: str = "free",
        adjacency: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
        if "graph" in mode and adjacency is None:
            raise ValueError(f"mode {mode!r} needs an adjacency matrix")
        if "text" in mode and text_embeddings is None:
            raise ValueError(f"mode {mode!r} needs text embeddings")
        self.mode = mode
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
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def label_embeddings(self) -> torch.Tensor:
        """The ``(output_dim, hidden_dim)`` class vectors."""
        emb = self.class_vectors
        if "text" in self.mode:
            emb = emb + self.text_gate * self.text_proj(self.text)
        if "graph" in self.mode:
            h = torch.relu(self.adjacency @ self.gcn1(emb))
            emb = emb + self.graph_gate * (self.adjacency @ self.gcn2(h))
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities, ``(batch, output_dim)``."""
        return torch.sigmoid(self.encoder(x) @ self.label_embeddings().T + self.bias)


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

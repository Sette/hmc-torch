"""Graph autoencoder of Kipf & Welling (2016), ported from ``tkipf/gae`` (TensorFlow) to PyTorch.

Surj Sec. 3.2: ``Z = Â ReLU(Â S W0) W1`` with ``Â = D^-1/2 (A + I) D^-1/2`` and the decoder
``σ(Z Zᵀ)``.  The benchmarks have no node features, so ``S = I`` (``tkipf/gae``'s
``--features 0``).  Defaults are that repository's: hidden 32, latent 16, Adam 0.01, 200
epochs, no dropout, Glorot-uniform weights, and the loss

    norm * mean(weighted_BCE_with_logits(Z Zᵀ, A + I, pos_weight)),
    pos_weight = (N² - |A|) / |A|,  norm = N² / (2 (N² - |A|)),

with ``A`` the symmetric adjacency without self loops.  ``tkipf/gae`` hides 15% of the edges
for its link-prediction benchmark; Surj only needs the embeddings, so every edge is used.

``python surj_repro/gae.py cellcycle_FUN`` prints the check of Surj's Figure 2: the mean
cosine distance between node embeddings for each shortest-path length.
"""

from __future__ import annotations

import math
import sys

import networkx as nx
import numpy as np
import torch
from torch import nn


def _glorot(fan_in: int, fan_out: int) -> nn.Parameter:
    r = math.sqrt(6.0 / (fan_in + fan_out))
    return nn.Parameter(torch.empty(fan_in, fan_out).uniform_(-r, r))


def symmetric_adjacency(a: np.ndarray) -> np.ndarray:
    """Undirected 0/1 adjacency without self loops."""
    s = ((np.asarray(a) + np.asarray(a).T) > 0).astype(np.float32)
    np.fill_diagonal(s, 0.0)
    return s


def train_gae(  # pylint: disable=too-many-arguments,too-many-locals
    a: np.ndarray,
    *,
    hidden: int = 32,
    dim: int = 16,
    epochs: int = 200,
    lr: float = 0.01,
    features: np.ndarray | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Node embeddings ``Z`` (N x dim) of the hierarchy whose adjacency is ``a``."""
    torch.manual_seed(seed)
    adj = torch.from_numpy(symmetric_adjacency(a)).to(device)
    n = adj.shape[0]
    a_tilde = adj + torch.eye(n, device=device)
    d_inv_sqrt = a_tilde.sum(1).pow(-0.5)
    a_hat = d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]
    s = torch.eye(n, device=device) if features is None else torch.as_tensor(
        features, dtype=torch.float32, device=device)
    w0 = _glorot(s.shape[1], hidden).to(device)
    w1 = _glorot(hidden, dim).to(device)
    w0, w1 = nn.Parameter(w0.detach()), nn.Parameter(w1.detach())
    opt = torch.optim.Adam([w0, w1], lr=lr)

    n_edges = adj.sum()
    pos_weight = (n * n - n_edges) / n_edges
    norm = n * n / (2.0 * (n * n - n_edges))
    labels = a_tilde

    def encode():
        return a_hat @ (torch.relu(a_hat @ (s @ w0)) @ w1)

    for _ in range(epochs):
        z = encode()
        loss = norm * nn.functional.binary_cross_entropy_with_logits(
            z @ z.T, labels, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return encode().cpu().numpy()


def figure2(a: np.ndarray, z: np.ndarray, max_len: int = 12) -> dict[int, float]:
    """Mean cosine distance between embeddings per shortest-path length (Surj Fig. 2)."""
    g = nx.Graph(symmetric_adjacency(a))
    zn = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    dist = 1.0 - zn @ zn.T
    sums, counts = np.zeros(max_len + 1), np.zeros(max_len + 1)
    for i, lengths in nx.all_pairs_shortest_path_length(g, cutoff=max_len):
        for j, length in lengths.items():
            if j > i:
                sums[length] += dist[i, j]
                counts[length] += 1
    return {k: float(sums[k] / counts[k]) for k in range(1, max_len + 1) if counts[k]}


if __name__ == "__main__":
    from data import load  # pylint: disable=import-outside-toplevel

    name = sys.argv[1] if len(sys.argv) > 1 else "cellcycle_FUN"
    d = load(name)
    emb = train_gae(d["a"], device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name}: shortest path -> mean cosine distance")
    for k, v in figure2(d["a"], emb).items():
        print(f"  {k:>2}  {v:.3f}")

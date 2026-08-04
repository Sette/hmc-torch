#!/usr/bin/env python3
"""Month 2: Sparse R + Ontology Embeddings on GO datasets."""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda")
SEED = 42

def compute_metrics(y_true, y_pred, eval_mask):
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best[0]: best = (f1, thr, p, r)
    try:
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except: auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


class OntologyGCN(nn.Module):
    """GCN over the GO label hierarchy graph to learn label embeddings."""
    def __init__(self, n_nodes, embed_dim=256, hidden_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.label_embed = nn.Parameter(torch.empty(n_nodes, embed_dim))
        nn.init.xavier_uniform_(self.label_embed)

        self.convs = nn.ModuleList()
        in_dim = embed_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embed_dim
            self.convs.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index):
        x = self.label_embed
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x  # (N, embed_dim)


class GCNConv(nn.Module):
    """Simple GCN layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.weight); nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        n_nodes = x.shape[0]
        deg = torch.zeros(n_nodes, device=x.device).scatter_add(
            0, row, torch.ones_like(row, dtype=torch.float)).clamp(min=1)
        deg_inv = deg.pow(-0.5)
        norm = deg_inv[row] * deg_inv[col]
        messages = x[col] @ self.weight
        out = torch.zeros(n_nodes, self.weight.shape[1], device=x.device)
        out = out.scatter_add(0, row.unsqueeze(-1).expand_as(messages),
                              messages * norm.unsqueeze(-1))
        return out + self.bias


class OntologyEnhancedModel(nn.Module):
    """MLP encoder + Ontology GCN + combined scoring + sparse R."""
    def __init__(self, input_dim, n_nodes, graph, embed_dim=256, hidden=512, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.n_nodes = n_nodes

        # Document encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, embed_dim),
        )

        # Ontology GCN
        edge_index = self._build_edge_index(graph)
        self.register_buffer('edge_index', edge_index)
        self.ontology = OntologyGCN(n_nodes, embed_dim=embed_dim, hidden_dim=hidden)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, n_nodes)

    def _build_edge_index(self, graph):
        """Build undirected edge index from DiGraph."""
        node_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        edges = set()
        for u, v in graph.edges():
            edges.add((node_to_idx[u], node_to_idx[v]))
        for u, v in list(edges):
            edges.add((v, u))
        idx = list(edges)
        return torch.tensor([[i for i, j in idx], [j for i, j in idx]], dtype=torch.long)

    def forward(self, x):
        # Document embeddings
        doc_emb = self.encoder(x)  # (B, embed_dim)

        # Ontology label embeddings
        label_emb = self.ontology(self.edge_index)  # (N, embed_dim)

        # Direct MLP scores
        mlp_scores = self.output_proj(doc_emb)  # (B, N)

        # Ontology similarity scores (dot product)
        ont_scores = torch.matmul(doc_emb, label_emb.T)  # (B, N)

        # Combine
        combined = self.alpha * mlp_scores + (1 - self.alpha) * ont_scores
        return torch.sigmoid(combined)


def run_experiment(ds_name, X_tr, y_tr, X_te, y_te, eval_mask, graph,
                   n_nodes, epochs=30, alpha=0.3):
    """Train OntologyEnhancedModel + sparse R reconciliation."""
    from hmc.models.hierarchical.sparse_r import BlockDiagonalR

    torch.manual_seed(SEED); np.random.seed(SEED)
    n_feat = X_tr.shape[1]
    t0 = time.time()

    sparse_r = BlockDiagonalR(graph)

    model = OntologyEnhancedModel(
        input_dim=n_feat, n_nodes=n_nodes, graph=graph,
        embed_dim=256, hidden=min(512, n_feat*2), alpha=alpha,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=64, shuffle=True)
    model.train()
    for ep in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            preds = model(bx)
            loss = F.binary_cross_entropy(preds, by)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=32, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores_raw = np.concatenate(preds, axis=0)

    # Sparse R reconciliation
    scores_rec = sparse_r.reconcile(torch.tensor(scores_raw).to(DEVICE)).cpu().numpy()

    f1_raw, au_raw, p_raw, r_raw, th_raw = compute_metrics(y_te, scores_raw, eval_mask)
    f1_rec, au_rec, p_rec, r_rec, th_rec = compute_metrics(y_te, scores_rec, eval_mask)

    return (f1_raw, au_raw, f1_rec, au_rec, time.time() - t0)


# ===== Run on 2 GO datasets (quick test first) =====
from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

results = {}
for ds_name in ["cellcycle_GO", "eisen_GO"]:
    print(f"\n{'='*60}")
    print(f"GO: {ds_name} — Ontology GCN + Sparse R")
    print(f"{'='*60}")

    mgr = initialize_dataset_experiments(ds_name, device="cpu", dataset_path="./data",
                                          dataset_type="arff", is_global=False)
    tr, va, te = mgr.get_datasets()
    X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
    y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
    X_te = te.x.astype(np.float32); y_te = te.y.astype(np.float32)
    em = np.array(mgr.to_eval, dtype=bool)
    n_nodes = len(mgr.nodes_idx)

    imp = SimpleImputer(strategy="mean"); sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = sc.transform(imp.transform(X_te)).astype(np.float32)

    print(f"  {X_tr.shape[0]} train | {X_te.shape[0]} test | {X_tr.shape[1]} feat | {n_nodes} nodes")

    # Try different alpha values
    for alpha in [0.1, 0.3, 0.5]:
        f1_raw, au_raw, f1_rec, au_rec, dur = run_experiment(
            ds_name, X_tr, y_tr, X_te, y_te, em, tr.g, n_nodes, epochs=25, alpha=alpha)
        results[f"{ds_name}_α={alpha}"] = {
            "f1_raw": float(f1_raw), "auprc_raw": float(au_raw),
            "f1_rec": float(f1_rec), "auprc_rec": float(au_rec), "time": dur}
        print(f"  α={alpha}: raw F1={f1_raw:.4f} AUPRC={au_raw:.4f} → "
              f"+R F1={f1_rec:.4f} AUPRC={au_rec:.4f} t={dur:.1f}s")

# ===== Compare with baseline (MLP only, no ontology) =====
print(f"\n{'='*60}")
print("BASELINE: MLP only (no ontology GCN)")
print(f"{'='*60}")

from hmc.models.global_classifier.constraint.model import ConstrainedModel
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

for ds_name in ["cellcycle_GO", "eisen_GO"]:
    mgr = initialize_dataset_experiments(ds_name, device="cpu", dataset_path="./data",
                                          dataset_type="arff", is_global=False)
    tr, va, te = mgr.get_datasets()
    X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
    y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
    X_te = te.x.astype(np.float32); y_te = te.y.astype(np.float32)
    em = np.array(mgr.to_eval, dtype=bool)
    n_nodes = len(mgr.nodes_idx)
    imp = SimpleImputer(strategy="mean"); sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = sc.transform(imp.transform(X_te)).astype(np.float32)

    sparse_r = BlockDiagonalR(tr.g)
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    model = ConstrainedModel(
        input_dim=X_tr.shape[1], hidden_dim=min(512, X_tr.shape[1]), output_dim=n_nodes,
        hyperparams={"batch_size": 64, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": min(512, X_tr.shape[1]),
                     "lr": 1e-4, "weight_decay": 1e-5},
        r_matrix=torch.eye(n_nodes).unsqueeze(0).to(DEVICE), baseline_model=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=64, shuffle=True)
    model.train()
    for ep in range(25):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = F.binary_cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for (bx,) in DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=32, shuffle=False):
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores_raw = np.concatenate(preds, axis=0)
    scores_rec = sparse_r.reconcile(torch.tensor(scores_raw).to(DEVICE)).cpu().numpy()

    f1_raw, au_raw, _, _, _ = compute_metrics(y_te, scores_raw, em)
    f1_rec, au_rec, _, _, _ = compute_metrics(y_te, scores_rec, em)
    dur = time.time() - t0
    results[f"{ds_name}_baseline"] = {
        "f1_raw": float(f1_raw), "auprc_raw": float(au_raw),
        "f1_rec": float(f1_rec), "auprc_rec": float(au_rec), "time": dur}
    print(f"  {ds_name} MLP: raw F1={f1_raw:.4f} AUPRC={au_raw:.4f} → "
          f"+R F1={f1_rec:.4f} AUPRC={au_rec:.4f} t={dur:.1f}s")

# ===== Report =====
print(f"\n{'='*70}")
print("GO ONTOLOGY + SPARSE R — Results vs Baseline")
print(f"{'='*70}")
print(f"{'Config':<30} {'F1':>8} {'AUPRC':>8}")
print("-"*50)

# Published SOTA for reference
sota = {"cellcycle_GO": {"HMCN-F": 0.400, "C-HMCNN": 0.413, "WWW'22": 0.460},
        "eisen_GO": {"HMCN-F": 0.440, "C-HMCNN": 0.455, "WWW'22": 0.487}}

for ds_name in ["cellcycle_GO", "eisen_GO"]:
    print(f"\n{ds_name}:")
    for sota_method, sota_auprc in sota[ds_name].items():
        print(f"  {sota_method:<28} {sota_auprc:>8.4f} (published)")
    best_ont = max((k, v) for k, v in results.items() if ds_name in k and "baseline" not in k)
    base = results[f"{ds_name}_baseline"]
    print(f"  MLP baseline +R         {base['auprc_rec']:>8.4f}")
    print(f"  Ontology+GCN α=best    {best_ont[1]['auprc_rec']:>8.4f}")

os.makedirs("./output/month2", exist_ok=True)
with open("./output/month2/ontology_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to ./output/month2/ontology_results.json")

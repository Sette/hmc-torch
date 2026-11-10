#!/usr/bin/env python3
"""Unified experiment protocol behind every number in the paper.

Protocol:
  * models are trained on the training split, and the validation split drives
    early stopping and the choice of the decision threshold (swept over
    0.05..0.95);
  * on the ARFF benchmarks (FunCat, GO, others) the model is then retrained on
    training + validation for the selected number of epochs, the convention of
    the published baselines (HMCN, C-HMCNN); on text it is not, following HPT;
  * the test split is touched once, for the final metrics.

Datasets without a validation file (enron/diatoms/imclef) get a fixed 20%
holdout of their training file from the dataset manager.

For each dataset and seed, three MLPs are trained, one per training objective:
  * ``bce``   -- plain binary cross-entropy;
  * ``hinge`` -- BCE + lambda * hinge penalty over all ancestor pairs (Eq. 2);
  * ``mc``    -- C-HMCNN's MCLoss (BCE on the max-constraint outputs).
Every model is evaluated twice: on its raw outputs and after bottom-up
reconciliation (Eq. 3), which gives the train-time / inference-time
decomposition.  The reconciliation is the sparse edge-list propagation,
which is exact (identical to the dense R-matrix max) at every scale.

On FunCat, a one-vs-rest GBDT baseline (with and without Platt calibration)
is also run.

Usage:
    PYTHONPATH=src python experiments/run_protocol.py --group funcat
    PYTHONPATH=src python experiments/run_protocol.py --datasets wos,arxiv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from torch import nn

sys.path.insert(0, "src")

from hmc.datasets.dataset_manager import (  # noqa: E402  pylint: disable=wrong-import-position
    initialize_dataset_experiments,
)
from hmc.datasets.registry import (  # noqa: E402  pylint: disable=wrong-import-position
    DatasetRegistry,
)
from hmc.models.hierarchical.sparse_r import (  # noqa: E402  pylint: disable=wrong-import-position
    BlockDiagonalR,
)

FUNCAT = [
    "cellcycle_FUN", "church_FUN", "derisi_FUN", "eisen_FUN", "expr_FUN",
    "gasch1_FUN", "gasch2_FUN", "pheno_FUN", "seq_FUN", "spo_FUN",
]
GO = [
    "cellcycle_GO", "derisi_GO", "eisen_GO", "expr_GO", "gasch1_GO",
    "gasch2_GO", "pheno_GO", "seq_GO", "spo_GO",
]
OTHERS = ["enron_others", "diatoms_others", "imclef07a_others", "imclef07d_others"]
TEXT = ["arxiv", "wos", "aapd", "eurlex"]
GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS, "text": TEXT}

OBJECTIVES = ("bce", "hinge", "mc")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
LAMBDA_HIER = 1.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> dict:
    """Load a dataset, standardize features on train, and build the hierarchy."""
    mgr = initialize_dataset_experiments(
        name,
        device="cpu",
        dataset_path="./data",
        is_global=True,
        # ArXiv keeps its 50k-record cap; every other text dataset is used whole.
        arxiv_max_records=50_000 if name == "arxiv" else 0,
        model_cache_dir="./models",
    )
    train, valid, test = mgr.get_datasets()
    xs = [np.asarray(s.x, dtype=np.float32) for s in (train, valid, test)]
    ys = [np.asarray(s.y, dtype=np.float32) for s in (train, valid, test)]

    # Impute (tabular features may have NaNs) and standardize with train stats.
    mean = np.nanmean(xs[0], axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    xs = [np.where(np.isnan(x), mean, x) for x in xs]
    std = xs[0].std(axis=0)
    std[std == 0] = 1.0
    xs = [((x - mean) / std).astype(np.float32) for x in xs]

    # Hierarchy: mgr.a[i, j] = 1 means j is a parent of i (child -> parent).
    adj = np.asarray(mgr.a)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    child, parent = np.nonzero(adj)
    graph.add_edges_from(zip(child.tolist(), parent.tolist()))
    sparse_r = BlockDiagonalR(graph, column_index={i: i for i in range(adj.shape[0])})

    # Ancestor pairs (for the hinge loss) and depth per column (for per-level F1).
    anc, desc = [], []
    for node in graph.nodes():
        for a in nx.descendants(graph, node):  # child->parent edges: descendants = ancestors
            anc.append(a)
            desc.append(node)
    depth = np.zeros(adj.shape[0], dtype=int)
    roots = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    lengths = {}
    for r in roots:
        for n, d in nx.single_source_shortest_path_length(graph.reverse(), r).items():
            lengths[n] = min(d, lengths.get(n, d))
    for n, d in lengths.items():
        depth[n] = d

    eval_mask = np.asarray(mgr.to_eval, dtype=bool)
    node_names = [None] * adj.shape[0]
    for node, idx in getattr(mgr, "nodes_idx", {}).items():
        if idx < len(node_names):
            node_names[idx] = node
    return {
        "name": name,
        "node_names": node_names,
        "edges": np.stack([child, parent], axis=1),
        "x": xs,
        "y": ys,
        "eval_mask": eval_mask,
        "sparse_r": sparse_r,
        "anc": np.asarray(anc, dtype=np.int64),
        "desc": np.asarray(desc, dtype=np.int64),
        "depth": depth,
        "n_edges": int(len(child)),
        "internal": int(len(set(parent.tolist()) & set(np.nonzero(eval_mask)[0].tolist()))),
    }


def hyperparams(name: str) -> dict:
    """Per-dataset hyperparameters: registry hidden size, fixed optimizer settings."""
    reg = DatasetRegistry()
    if name.endswith("_FUN") or name.endswith("_GO") or name.endswith("_others"):
        family = "FUN" if name.endswith("_FUN") else "GO" if name.endswith("_GO") else "others"
        key = name.rsplit("_", 1)[0]
        hidden = reg.hidden_dims[family][key]
        batch = 64 if family == "GO" else 32
        return {"hidden": hidden, "batch": batch, "max_epochs": 200, "patience": 20, "refit": True}
    return {"hidden": 512, "batch": 64, "max_epochs": 100, "patience": 10, "refit": False}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _counts(y_true, scores, thr):
    pred = scores >= thr
    tp = float(np.logical_and(pred, y_true == 1).sum())
    fp = float(np.logical_and(pred, y_true == 0).sum())
    fn = float(np.logical_and(~pred, y_true == 1).sum())
    return tp, fp, fn


def prf(y_true, scores, thr):
    tp, fp, fn = _counts(y_true, scores, thr)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def auprc(y_true, scores):
    """Micro-averaged average precision, identical to sklearn's (ties grouped).

    Computed with torch (on GPU when available): sklearn needs seconds per call
    on the larger label spaces, and early stopping calls this every epoch.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.as_tensor(np.asarray(y_true).ravel(), device=dev)
    s = torch.as_tensor(np.asarray(scores, dtype=np.float32).ravel(), device=dev)
    n_pos = y.sum()
    if n_pos == 0:
        return 0.0
    s, order = torch.sort(s, descending=True)
    y = y[order]
    tp = torch.cumsum(y, 0, dtype=torch.float64)
    fp = torch.cumsum(1 - y, 0, dtype=torch.float64)
    # Keep the last position of every distinct score (sklearn's thresholds).
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def pick_threshold(y_val, s_val):
    """Threshold maximizing validation micro-F1."""
    f1s = [prf(y_val, s_val, t)[2] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1s))])


def evaluate(data, s_val, s_test):
    """Validation-chosen threshold applied to test, plus threshold-free AUPRC."""
    m = data["eval_mask"]
    y_val, y_test = data["y"][1][:, m], data["y"][2][:, m]
    thr = pick_threshold(y_val, s_val[:, m])
    p, r, f1 = prf(y_test, s_test[:, m], thr)
    per_level = {}
    for lvl in sorted(set(data["depth"][m].tolist())):
        cols = m & (data["depth"] == lvl)
        if data["y"][2][:, cols].sum() == 0:
            continue
        per_level[int(lvl)] = prf(data["y"][2][:, cols], s_test[:, cols], thr)[2]
    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "auprc": auprc(y_test, s_test[:, m]),
        "threshold": thr,
        "val_f1": prf(y_val, s_val[:, m], thr)[2],
        "per_level_f1": per_level,
    }


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """3-layer MLP with sigmoid outputs (the GlobalSigmoidHead used in the paper)."""

    def __init__(self, d_in, hidden, d_out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def objective_loss(kind, probs, y, eval_cols, sparse_r, anc, desc):
    """Training loss for one of the three objectives."""
    bce = nn.functional.binary_cross_entropy
    if kind == "mc":
        # C-HMCNN MCLoss: negatives use MCM(h), positives use MCM(y * h).
        mcm = sparse_r.reconcile(probs)
        mcm_pos = sparse_r.reconcile(y * probs)
        out = (1 - y) * mcm + y * mcm_pos
        return bce(out[:, eval_cols].clamp(1e-7, 1 - 1e-7), y[:, eval_cols])
    loss = bce(probs[:, eval_cols], y[:, eval_cols])
    if kind == "hinge" and anc.numel():
        loss = loss + LAMBDA_HIER * torch.relu(probs[:, desc] - probs[:, anc]).mean()
    return loss


@torch.no_grad()
def predict(model, x, device, batch=1024):
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch]).to(device)).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def _fit(data, kind, seed, device, hp, x, y, epochs, x_val=None, model_fn=None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Train a model for up to ``epochs`` epochs.

    With ``x_val``, stop early on the validation AUPRC of the reconciled outputs
    and return the best checkpoint; otherwise train for exactly ``epochs``.
    ``model_fn(input_dim, output_dim)`` builds the network (default: the MLP).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = data["eval_mask"]
    eval_cols = torch.from_numpy(np.nonzero(m)[0]).to(device)
    anc = torch.from_numpy(data["anc"]).to(device)
    desc = torch.from_numpy(data["desc"]).to(device)
    sparse_r = data["sparse_r"]
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)

    model = (model_fn or (lambda d_in, d_out: MLP(d_in, hp["hidden"], d_out)))(
        x.shape[1], y.shape[1]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad = -1.0, None, epochs, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_t), generator=gen)
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i : i + hp["batch"]]
            xb, yb = x_t[idx].to(device), y_t[idx].to(device)
            loss = objective_loss(kind, model(xb), yb, eval_cols, sparse_r, anc, desc)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if x_val is None:
            continue
        s_val = _reconcile(data, predict(model, x_val, device), device)
        score = auprc(data["y"][1][:, m], s_val[:, m])
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best


def _reconcile(data, scores, device):
    return data["sparse_r"].reconcile(torch.from_numpy(scores).to(device)).cpu().numpy()


def train_mlp(data, kind, seed, device, hp, model_fn=None):
    """Early-stop on validation, then (optionally) refit on train + validation.

    The decision threshold always comes from the validation scores of the
    early-stopped model, so the test split never influences any choice.
    """
    x_tr, x_val, x_te = data["x"]
    y_tr, y_val = data["y"][0], data["y"][1]
    t0 = time.time()
    model, best_epoch, best = _fit(data, kind, seed, device, hp, x_tr, y_tr,
                                   hp["max_epochs"], x_val=x_val, model_fn=model_fn)
    raw_val = predict(model, x_val, device)
    if hp["refit"]:
        model, _, _ = _fit(data, kind, seed, device, hp,
                           np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val]),
                           best_epoch, model_fn=model_fn)
    train_time = time.time() - t0
    raw_test = predict(model, x_te, device)
    results = {}
    for mode, (sv, st) in {
        "raw": (raw_val, raw_test),
        "rec": (_reconcile(data, raw_val, device), _reconcile(data, raw_test, device)),
    }.items():
        results[mode] = evaluate(data, sv, st)
    return {
        "objective": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_auprc": best,
        "refit": hp["refit"],
        "train_time_s": train_time,
        "eval": results,
    }


# ---------------------------------------------------------------------------
# GBDT baseline (FunCat)
# ---------------------------------------------------------------------------


def _fit_node(x, y, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier  # pylint: disable=import-outside-toplevel

    est = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
    return est.fit(x, y)


def _gbdt_scores(x_fit, y_fit, nodes, xs, seed):
    models = Parallel(n_jobs=-1)(delayed(_fit_node)(x_fit, y_fit[:, j], seed) for j in nodes)
    prev = y_fit.mean(axis=0)
    out = []
    for x in xs:
        s = np.tile(prev, (len(x), 1)).astype(np.float32)
        for est, j in zip(models, nodes):
            s[:, j] = est.predict_proba(x)[:, 1]
        out.append(s)
    return out


def run_gbdt(data, seed=0):
    """One-vs-rest HistGradientBoosting for nodes with >= 5 training positives.

    Thresholds come from a train-only fit scored on validation; the reported
    test scores come from a refit on train + validation, as for the MLPs.
    """
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel

    x_tr, x_val, x_te = data["x"]
    y_tr, y_val = data["y"][0], data["y"][1]
    m = data["eval_mask"]
    nodes = [j for j in np.nonzero(m)[0] if y_tr[:, j].sum() >= 5]
    t0 = time.time()
    s_val, s_te_trainonly = _gbdt_scores(x_tr, y_tr, nodes, [x_val, x_te], seed)
    (s_te,) = _gbdt_scores(np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val]),
                           nodes, [x_te], seed)
    train_time = time.time() - t0

    rec = lambda s: data["sparse_r"].reconcile(torch.from_numpy(s)).numpy()  # noqa: E731
    out = {
        "raw": evaluate(data, s_val, s_te),
        "rec": evaluate(data, rec(s_val), rec(s_te)),
        "train_time_s": train_time,
        "nodes_trained": len(nodes),
    }
    # Platt scaling: calibrators fitted on validation scores of the train-only
    # model and applied to its test scores; AUPRC is threshold-free.
    s_te = s_te_trainonly
    out["trainonly_auprc"] = auprc(data["y"][2][:, m], s_te[:, m])
    s_te_platt = s_te.copy()
    for j in nodes:
        if 0 < y_val[:, j].sum() < len(y_val):
            lr = LogisticRegression().fit(s_val[:, [j]], y_val[:, j])
            s_te_platt[:, j] = lr.predict_proba(s_te[:, [j]])[:, 1]
    out["platt_auprc"] = auprc(data["y"][2][:, m], s_te_platt[:, m])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_dataset(name, seeds, device, out_dir, objectives, gbdt, max_epochs=None):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        print(f"[skip] {name}: {path} exists")
        return
    t0 = time.time()
    data = load_dataset(name)
    hp = hyperparams(name)
    if max_epochs:
        hp["max_epochs"] = max_epochs
    m = data["eval_mask"]
    record = {
        "dataset": name,
        "n_train": len(data["x"][0]),
        "n_valid": len(data["x"][1]),
        "n_test": len(data["x"][2]),
        "n_features": int(data["x"][0].shape[1]),
        "n_nodes": int(len(m)),
        "n_eval": int(m.sum()),
        "n_internal_eval": data["internal"],
        "n_edges": data["n_edges"],
        "max_depth": int(data["depth"].max()),
        "hyperparams": hp,
        "protocol": "train on train; early stopping and threshold on valid; report on test",
        "runs": [],
    }
    print(f"\n=== {name}: train {record['n_train']} valid {record['n_valid']} "
          f"test {record['n_test']} nodes {record['n_nodes']} eval {record['n_eval']}")
    for seed in seeds:
        for kind in objectives:
            run = train_mlp(data, kind, seed, device, hp)
            record["runs"].append(run)
            ev = run["eval"]
            print(f"  seed {seed:>3} {kind:<5} ep {run['best_epoch']:>3}  "
                  f"raw F1 {ev['raw']['f1']:.4f} AUPRC {ev['raw']['auprc']:.4f} | "
                  f"rec F1 {ev['rec']['f1']:.4f} AUPRC {ev['rec']['auprc']:.4f}  "
                  f"({run['train_time_s']:.0f}s)", flush=True)
    if gbdt:
        record["gbdt"] = run_gbdt(data)
        g = record["gbdt"]
        print(f"  gbdt  rec F1 {g['rec']['f1']:.4f} AUPRC {g['rec']['auprc']:.4f} "
              f"platt AUPRC {g['platt_auprc']:.4f} ({g['train_time_s']:.0f}s)")
    record["wall_time_s"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=1, default=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=[*GROUPS, "all"], default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--seeds", default="0,42,123")
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="./output/protocol")
    parser.add_argument("--no_gbdt", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()

    if args.datasets:
        names = args.datasets.split(",")
    elif args.group == "all":
        names = FUNCAT + GO + OTHERS + TEXT
    elif args.group:
        names = GROUPS[args.group]
    else:
        parser.error("pass --group or --datasets")
    seeds = [int(s) for s in args.seeds.split(",")]
    objectives = args.objectives.split(",")
    device = torch.device(args.device)
    for name in names:
        run_dataset(name, seeds, device, args.out_dir, objectives,
                    gbdt=name in FUNCAT and not args.no_gbdt, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()

"""Surj pipeline (Yang & Howe, WWW'22): ``python -m hmc.main --method surj``.

1. Load the dataset, impute and standardize features with training statistics.
2. Learn label embeddings with a graph autoencoder over the hierarchy
   (:func:`~hmc.models.hierarchical.graph_autoencoder.pretrain_label_gae`).
3. Train :class:`~hmc.models.hierarchical.surj.SurjClassifier` with BCE,
   early stopping on the validation AUPRC (``--epochs`` is the cap).
4. Choose the threshold on validation (``--best_threshold true``; 0.5
   otherwise) and score the test split once, on the raw outputs and after
   bottom-up reconciliation, with AUPRC, micro-F1 and Global Hierarchy
   Violation.

Writes the artefacts of the tabular pipelines (``scores_*.npz``,
``metrics.json``, ``run-config.json``) to ``args.results_path``.
"""

from __future__ import annotations

import logging
import time

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from hmc.models.hierarchical.graph_autoencoder import pretrain_label_gae
from hmc.models.hierarchical.sparse_r import BlockDiagonalR
from hmc.models.hierarchical.surj import SurjClassifier
from hmc.pipeline.tabular.main import _save_artefacts
from hmc.utils.datasets.scaling import standardize_splits
from hmc.utils.metrics.hierarchy_violation import global_hierarchy_violation

logger = logging.getLogger(__name__)

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
PATIENCE = 20


def _hierarchy(mgr):
    """Edges, ancestor pairs, depth and the sparse R of a dataset manager.

    ``mgr.a[i, j] = 1`` means ``j`` is a parent of ``i`` (child -> parent), so
    ``nx.descendants`` of a node are its ancestors.
    """
    adj = np.asarray(mgr.a)
    n = adj.shape[0]
    child, parent = np.nonzero(adj)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(child.tolist(), parent.tolist()))
    anc, desc = [], []
    for node in graph.nodes():
        for a in nx.descendants(graph, node):
            anc.append(a)
            desc.append(node)
    depth = np.zeros(n, dtype=int)
    reverse = graph.reverse()
    for root in (v for v in graph.nodes() if graph.out_degree(v) == 0):
        for v, d in nx.single_source_shortest_path_length(reverse, root).items():
            depth[v] = d if depth[v] == 0 else min(depth[v], d)
    return {
        "edges": np.stack([child, parent], axis=1),
        "anc": np.asarray(anc, dtype=np.int64),
        "desc": np.asarray(desc, dtype=np.int64),
        "depth": depth,
        "sparse_r": BlockDiagonalR(graph, column_index={i: i for i in range(n)}),
    }


def _hidden_dim(args, dataset_name: str) -> int:
    """Registry hidden size for ARFF datasets, 512 otherwise."""
    family = dataset_name.rsplit("_", 1)[-1] if "_" in dataset_name else ""
    key = dataset_name.rsplit("_", 1)[0]
    return args.registry.hidden_dims.get(family, {}).get(key, 512)


def _prf(y, s, thr):
    pred = s >= thr
    tp = float(np.logical_and(pred, y == 1).sum())
    fp = float(np.logical_and(pred, y == 0).sum())
    fn = float(np.logical_and(~pred, y == 1).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _auprc(y, s) -> float:
    return float(average_precision_score(y, s, average="micro")) if y.sum() else 0.0


@torch.no_grad()
def _predict(model, x, device, batch=1024):
    model.eval()
    return torch.cat(
        [
            model(torch.from_numpy(x[i : i + batch]).to(device)).cpu()
            for i in range(0, len(x), batch)
        ]
    ).numpy()


def train_surj(dataset_name: str, args) -> dict:  # pylint: disable=too-many-locals,too-many-statements
    """Train and evaluate Surj on ``dataset_name``; returns the metrics dict."""
    from hmc.datasets.dataset_manager import initialize_dataset_experiments  # pylint: disable=import-outside-toplevel

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mgr = initialize_dataset_experiments(
        dataset_name,
        device="cpu",
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_max_records=args.dataset.arxiv_max_records,
        model_cache_dir=args.dataset.model_cache_dir,
    )
    train, valid, test = mgr.get_datasets()
    x_tr, x_val, x_te = standardize_splits(train.x, valid.x, test.x)
    y_tr, y_val, y_te = (
        np.asarray(s.y, dtype=np.float32) for s in (train, valid, test)
    )
    h = _hierarchy(mgr)
    m = np.asarray(mgr.to_eval, dtype=bool)
    n_labels = y_tr.shape[1]

    t0 = time.time()
    emb = pretrain_label_gae(
        h["edges"],
        n_labels,
        dim=args.surj_dim,
        epochs=args.surj_gae_epochs,
        depth=h["depth"],
        seed=args.seed,
        device=device,
        loss=args.surj_gae_loss,
    )
    gae_time = time.time() - t0

    model = SurjClassifier(
        x_tr.shape[1],
        n_labels,
        torch.from_numpy(emb),
        hidden_dim=_hidden_dim(args, dataset_name),
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    eval_cols = torch.from_numpy(np.nonzero(m)[0]).to(device)
    x_t, y_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    gen = torch.Generator().manual_seed(args.seed)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_t), generator=gen)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            out = model(x_t[idx].to(device))[:, eval_cols]
            loss = nn.functional.binary_cross_entropy(
                out, y_t[idx].to(device)[:, eval_cols]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = _auprc(y_val[:, m], _predict(model, x_val, device)[:, m])
        logger.info("epoch %d  val AUPRC %.4f", epoch + 1, score)
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    def rec(s):
        return h["sparse_r"].reconcile(torch.from_numpy(s)).numpy()

    raw_val, raw_te = _predict(model, x_val, device), _predict(model, x_te, device)
    metrics = {
        "best_epoch": best_epoch,
        "val_auprc": best,
        "gae_time_s": gae_time,
        "train_time_s": train_time,
    }
    for mode, (s_val, s_te) in {
        "raw": (raw_val, raw_te),
        "rec": (rec(raw_val), rec(raw_te)),
    }.items():
        if args.best_threshold:
            thr = float(
                THRESHOLDS[
                    int(
                        np.argmax(
                            [_prf(y_val[:, m], s_val[:, m], t)[2] for t in THRESHOLDS]
                        )
                    )
                ]
            )
        else:
            thr = 0.5
        p, r, f1 = _prf(y_te[:, m], s_te[:, m], thr)
        metrics[mode] = {
            "micro_f1": f1,
            "micro_precision": p,
            "micro_recall": r,
            "threshold": thr,
            "auprc_micro": _auprc(y_te[:, m], s_te[:, m]),
            "ghv": global_hierarchy_violation(s_te, h["anc"], h["desc"], m),
            "ghv_relevant": global_hierarchy_violation(
                s_te, h["anc"], h["desc"], m, threshold=thr
            ),
        }
    run_config = {
        "method": "surj",
        "dataset": dataset_name,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "dropout": args.dropout,
        "gae_dim": args.surj_dim,
        "gae_epochs": args.surj_gae_epochs,
        "gae_loss": args.surj_gae_loss,
        "n_train": len(x_tr),
        "n_valid": len(x_val),
        "n_test": len(x_te),
        "n_labels": n_labels,
        "n_eval": int(m.sum()),
    }
    _save_artefacts(args.results_path, raw_te, rec(raw_te), metrics, run_config)
    logger.info(
        "Surj %s: test AUPRC raw %.4f rec %.4f, F1 rec %.4f",
        dataset_name,
        metrics["raw"]["auprc_micro"],
        metrics["rec"]["auprc_micro"],
        metrics["rec"]["micro_f1"],
    )
    args.score = metrics
    return metrics

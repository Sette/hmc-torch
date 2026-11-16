#!/usr/bin/env python3
"""Parity check between our ARFF loader and the original C-HMCNN one.

Stage 1 (data): parse every ARFF benchmark with C-HMCNN's ``utils/parser.py``
(imported unmodified from ``./C-HMCNN`` through a small compatibility shim) and
with our ``HMCDatasetManager``, then compare node order, adjacency, ``to_eval``,
features, labels and the R-matrix.

Stage 2 (model): train C-HMCNN's own model and training loop (copied verbatim
from ``C-HMCNN/main.py``: MCLoss, batch 4, dropout 0.7, Adam, fixed epochs,
train + validation merged) on four data variants:
  * ``theirs``          -- their parser + their StandardScaler/imputer (train+val);
  * ``ours``            -- our parser + the same preprocessing;
  * ``ours_noscale``    -- our parser, raw features (what ``hmc.main --method global`` does);
  * ``ours_trainscale`` -- our parser, z-score on train only (what ``run_protocol.py`` does).
The metric is C-HMCNN's: micro AUPRC on the test split over ``to_eval``.

Usage:
    PYTHONPATH=src python experiments/compare_chmcnn_loader.py --group all --skip_train
    PYTHONPATH=src python experiments/compare_chmcnn_loader.py --datasets cellcycle_FUN --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types

import networkx as nx
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch import nn

sys.path.insert(0, "src")

from hmc.datasets.dataset_manager import (  # noqa: E402  pylint: disable=wrong-import-position
    initialize_dataset_experiments,
)
from hmc.datasets.gofun.dataset_arff import (  # noqa: E402  pylint: disable=wrong-import-position
    HMCDatasetArff,
)
from hmc.models.hierarchical.sparse_r import (  # noqa: E402  pylint: disable=wrong-import-position
    BlockDiagonalR,
)
from hmc.utils.datasets.paths import (  # noqa: E402  pylint: disable=wrong-import-position
    get_dataset_paths,
)

CHMCNN_DIR = "C-HMCNN"
OUT_DIR = "output/chmcnn_parity"


# ---------------------------------------------------------------------------
# Shim: import C-HMCNN's parser unmodified (py2.7 era: keras, nx.to_numpy_matrix)
# ---------------------------------------------------------------------------


def _import_chmcnn():
    keras = types.ModuleType("keras")
    keras.utils = types.SimpleNamespace(to_categorical=lambda i, n: np.eye(n)[i])
    sys.modules.setdefault("keras", keras)
    if not hasattr(nx, "to_numpy_matrix"):
        # their call passes order=nodes, which old networkx accepted and ignored
        nx.to_numpy_matrix = lambda g, nodelist=None, order=None: nx.to_numpy_array(
            g, nodelist=nodelist
        )
    os.environ["DATA_FOLDER"] = CHMCNN_DIR
    sys.path.insert(0, CHMCNN_DIR)
    from utils import datasets as their_datasets  # pylint: disable=import-outside-toplevel
    from utils import parser as their_parser  # pylint: disable=import-outside-toplevel

    sys.path.remove(CHMCNN_DIR)
    return their_parser, their_datasets


THEIR_PARSER, THEIR_DATASETS = _import_chmcnn()

FUNCAT = [n for n in THEIR_DATASETS if n.endswith("_FUN")]
GO = [n for n in THEIR_DATASETS if n.endswith("_GO")]
OTHERS = [n for n in THEIR_DATASETS if n.endswith("_others")]
GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS, "all": FUNCAT + GO + OTHERS}

# README of C-HMCNN, mean over 10 seeds.
PUBLISHED = {
    "cellcycle_FUN": 0.255, "derisi_FUN": 0.195, "eisen_FUN": 0.306, "expr_FUN": 0.302,
    "gasch1_FUN": 0.286, "gasch2_FUN": 0.258, "seq_FUN": 0.292, "spo_FUN": 0.215,
    "cellcycle_GO": 0.413, "derisi_GO": 0.370, "eisen_GO": 0.455, "expr_GO": 0.447,
    "gasch1_GO": 0.436, "gasch2_GO": 0.414, "seq_GO": 0.446, "spo_GO": 0.382,
    "diatoms_others": 0.758, "enron_others": 0.756,
    "imclef07a_others": 0.956, "imclef07d_others": 0.927,
}

# Hyperparameters from C-HMCNN/main.py.
HIDDEN = {
    "FUN": {"cellcycle": 500, "derisi": 500, "eisen": 500, "expr": 1250, "gasch1": 1000,
            "gasch2": 500, "seq": 2000, "spo": 250},
    "GO": {"cellcycle": 1000, "derisi": 500, "eisen": 500, "expr": 4000, "gasch1": 500,
           "gasch2": 500, "seq": 9000, "spo": 500},
    "others": {"diatoms": 2000, "enron": 1000, "imclef07a": 1000, "imclef07d": 1000},
}
LRS = {"FUN": 1e-4, "GO": 1e-4, "others": 1e-5}
EPOCHS = {
    "FUN": {"cellcycle": 106, "derisi": 67, "eisen": 110, "expr": 20, "gasch1": 42,
            "gasch2": 123, "seq": 13, "spo": 115},
    "GO": {"cellcycle": 62, "derisi": 91, "eisen": 123, "expr": 70, "gasch1": 122,
           "gasch2": 177, "seq": 45, "spo": 103},
    "others": {"diatoms": 474, "enron": 133, "imclef07a": 592, "imclef07d": 588},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_theirs(name):
    """(train, val or None, test) as C-HMCNN's arff_data objects."""
    if "others" in name:
        train, test = THEIR_PARSER.initialize_other_dataset(name, THEIR_DATASETS)
        return train, None, test
    return THEIR_PARSER.initialize_dataset(name, THEIR_DATASETS)


def load_ours(name):
    """Our manager plus, for ``_others``, the full (un-held-out) training file."""
    mgr = initialize_dataset_experiments(name, device="cpu", dataset_path="./data", is_global=True)
    full_train = None
    if "others" in name:
        is_go, train_file, _ = get_dataset_paths(dataset_path="./data")[name]
        full_train = HMCDatasetArff(train_file, is_go=is_go)
    return mgr, full_train


def their_r(a):
    """R-matrix exactly as C-HMCNN/main.py builds it (before unsqueeze)."""
    r = np.zeros(a.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(a)
    for i in range(len(a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0)


def get_constr_out(x, r):
    """C-HMCNN/main.py, verbatim."""
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), r.shape[1], r.shape[1])
    r_batch = r.expand(len(x), r.shape[1], r.shape[1])
    final_out, _ = torch.max(r_batch * c_out.double(), dim=2)
    return final_out


# ---------------------------------------------------------------------------
# Stage 1: data parity
# ---------------------------------------------------------------------------


def _first_diff(a, b):
    diff = np.argwhere(~np.isclose(a, b, equal_nan=True))
    return diff[0].tolist() if len(diff) else None


def _cmp_split(theirs, x_ours, y_ours):
    res = {"n": [int(theirs.X.shape[0]), int(x_ours.shape[0])],
           "shape_x": theirs.X.shape == x_ours.shape, "shape_y": theirs.Y.shape == y_ours.shape}
    if res["shape_x"]:
        res["x"] = bool(np.allclose(theirs.X, x_ours, equal_nan=True))
        if not res["x"]:
            res["x_first_diff"] = _first_diff(theirs.X, x_ours)
            res["x_max_abs_diff"] = float(np.nanmax(np.abs(theirs.X - x_ours)))
    if res["shape_y"]:
        res["y"] = bool(np.array_equal(theirs.Y, y_ours))
        if not res["y"]:
            res["y_n_diff"] = int((theirs.Y != y_ours).sum())
    return res


def data_parity(name):
    """Compare every structure C-HMCNN feeds its model against ours."""
    t_train, t_val, t_test = load_theirs(name)
    mgr, full_train = load_ours(name)
    o_train, o_val, o_test = mgr.get_datasets()

    our_terms = [n for n, _ in sorted(mgr.nodes_idx.items(), key=lambda kv: kv[1])]
    res = {
        "terms": list(t_train.terms) == our_terms,
        "a": bool(np.array_equal(np.asarray(t_train.A), np.asarray(mgr.a))),
        "to_eval": list(t_train.to_eval) == list(mgr.to_eval),
        "n_nodes": [len(t_train.terms), len(our_terms)],
        "test": _cmp_split(t_test, np.asarray(o_test.x), np.asarray(o_test.y)),
    }
    if t_val is None:
        res["train"] = _cmp_split(t_train, np.asarray(full_train.x), np.asarray(full_train.y))
        res["holdout"] = {"their_train": int(t_train.X.shape[0]),
                          "our_train": int(o_train.x.shape[0]), "our_valid": int(o_val.x.shape[0])}
    else:
        res["train"] = _cmp_split(t_train, np.asarray(o_train.x), np.asarray(o_train.y))
        res["valid"] = _cmp_split(t_val, np.asarray(o_val.x), np.asarray(o_val.y))

    # R-matrix: theirs (dense) vs our pipeline (same code on mgr.a) vs BlockDiagonalR.
    r_t = their_r(np.asarray(t_train.A))
    r_o = their_r(np.asarray(mgr.a))
    res["r_matrix"] = bool(torch.equal(r_t, r_o))
    adj = np.asarray(mgr.a)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    child, parent = np.nonzero(adj)
    graph.add_edges_from(zip(child.tolist(), parent.tolist()))
    sparse_r = BlockDiagonalR(graph, column_index={i: i for i in range(adj.shape[0])})
    probe = torch.rand(8, adj.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    dense = get_constr_out(probe, r_t.unsqueeze(0))
    res["sparse_r_matches_dense"] = bool(torch.allclose(sparse_r.reconcile(probe).double(), dense))
    return res


def _ok(res):
    checks = [res["terms"], res["a"], res["to_eval"], res["r_matrix"], res["sparse_r_matches_dense"]]
    for split in ("train", "valid", "test"):
        if split in res:
            s = res[split]
            checks += [s["shape_x"], s["shape_y"], s.get("x", False), s.get("y", False)]
    return all(checks)


# ---------------------------------------------------------------------------
# Stage 2: C-HMCNN model on each data variant
# ---------------------------------------------------------------------------


class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h), verbatim from C-HMCNN/main.py."""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, r):
        super().__init__()
        self.nb_layers = hyperparams["num_layers"]
        self.R = r  # pylint: disable=invalid-name
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        self.drop = nn.Dropout(hyperparams["dropout"])
        self.sigmoid = nn.Sigmoid()
        self.f = nn.Tanh() if hyperparams["non_lin"] == "tanh" else nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        return get_constr_out(x, self.R)


def _preprocess(mode, x_tr, x_val, x_te):
    """Return float64 (train, val, test) features for one preprocessing variant."""
    if mode == "none":
        return x_tr, x_val, x_te
    if mode == "chmcnn":  # C-HMCNN/main.py: fit on train+val (train only for _others)
        fit = x_tr if x_val is None else np.concatenate((x_tr, x_val))
        scaler = preprocessing.StandardScaler().fit(fit.astype(float))
        imp = SimpleImputer(missing_values=np.nan, strategy="mean").fit(fit.astype(float))
        tf = lambda x: None if x is None else scaler.transform(imp.transform(x))  # noqa: E731
        return tf(x_tr), tf(x_val), tf(x_te)
    # "train": run_protocol.py's z-score with train stats
    mean = np.nanmean(x_tr, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    xs = [None if x is None else np.where(np.isnan(x), mean, x) for x in (x_tr, x_val, x_te)]
    std = xs[0].std(axis=0)
    std[std == 0] = 1.0
    return tuple(None if x is None else (x - mean) / std for x in xs)


def build_variants(name):
    """Map variant name -> (x_tr, y_tr, x_val, y_val, x_te, y_te, A, to_eval)."""
    t_train, t_val, t_test = load_theirs(name)
    mgr, full_train = load_ours(name)
    o_train, o_val, o_test = mgr.get_datasets()
    if full_train is not None:  # C-HMCNN trains on the whole file for _others
        o_train, o_val = full_train, None

    def pack(tr, va, te, a, to_eval, mode):
        x_tr, x_va, x_te = _preprocess(mode, np.asarray(tr[0], dtype=float),
                                       None if va is None else np.asarray(va[0], dtype=float),
                                       np.asarray(te[0], dtype=float))
        return {"x": (x_tr, x_va, x_te),
                "y": (np.asarray(tr[1]), None if va is None else np.asarray(va[1]), np.asarray(te[1])),
                "a": np.asarray(a), "to_eval": list(to_eval)}

    t = ((t_train.X, t_train.Y), None if t_val is None else (t_val.X, t_val.Y), (t_test.X, t_test.Y))
    o = ((o_train.x, o_train.y), None if o_val is None else (o_val.x, o_val.y), (o_test.x, o_test.y))
    return {
        "theirs": pack(*t, t_train.A, t_train.to_eval, "chmcnn"),
        "ours": pack(*o, mgr.a, mgr.to_eval, "chmcnn"),
        "ours_noscale": pack(*o, mgr.a, mgr.to_eval, "none"),
        "ours_trainscale": pack(*o, mgr.a, mgr.to_eval, "train"),
    }


def train_chmcnn(name, data, seed, device, epochs_override=None):  # pylint: disable=too-many-locals
    """C-HMCNN/main.py's training + test evaluation on pre-built arrays."""
    data_name, ontology = name.split("_")[0], name.split("_")[1]
    hidden_dim = HIDDEN[ontology][data_name]
    lr = LRS[ontology]
    num_epochs = epochs_override or EPOCHS[ontology][data_name]
    hyperparams = {"num_layers": 3, "dropout": 0.7, "non_lin": "relu"}
    batch_size, weight_decay = 4, 1e-5

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    to_eval = torch.tensor(data["to_eval"], dtype=torch.bool)
    r = their_r(data["a"]).unsqueeze(0).to(device)

    (x_tr, x_va, x_te), (y_tr, y_va, y_te) = data["x"], data["y"]
    x_train, y_train = torch.tensor(x_tr).to(device), torch.tensor(y_tr).to(device)
    train_dataset = list(zip(x_train, y_train))
    if x_va is not None:
        for x, y in zip(torch.tensor(x_va).to(device), torch.tensor(y_va).to(device)):
            train_dataset.append((x, y))
    test_dataset = list(zip(torch.tensor(x_te).to(device), torch.tensor(y_te).to(device)))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConstrainedFFNNModel(x_tr.shape[1], hidden_dim, len(data["to_eval"]), hyperparams, r).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for _ in range(num_epochs):
        model.train()
        for x, labels in train_loader:
            optimizer.zero_grad()
            output = model(x.float())
            constr_output = get_constr_out(output, r)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, r)
            train_output = (1 - labels) * constr_output.double() + labels * train_output
            loss = criterion(train_output[:, to_eval], labels[:, to_eval])
            loss.backward()
            optimizer.step()

    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for x, y in test_loader:
            outs.append(model(x.float()).cpu())
            ys.append(y.cpu())
    constr_test, y_test = torch.cat(outs), torch.cat(ys)
    return float(average_precision_score(y_test[:, to_eval], constr_test[:, to_eval], average="micro"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_summary(parity, results):
    lines = ["# C-HMCNN loader parity", "", "## Stage 1: data", "",
             "| dataset | ok | terms | A | to_eval | R | train X/Y | valid X/Y | test X/Y | n train (theirs/ours) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    mark = lambda b: "✔" if b else "✘"  # noqa: E731
    for name, res in parity.items():
        def xy(split):
            s = res.get(split)
            return "—" if s is None else f"{mark(s.get('x', False))}/{mark(s.get('y', False))}"
        n_tr = res["holdout"] if "holdout" in res else None
        n_str = (f"{n_tr['their_train']}/{n_tr['our_train']}+{n_tr['our_valid']}(holdout)"
                 if n_tr else "/".join(map(str, res["train"]["n"])))
        lines.append(f"| {name} | {mark(_ok(res))} | {mark(res['terms'])} | {mark(res['a'])} | "
                     f"{mark(res['to_eval'])} | {mark(res['r_matrix'] and res['sparse_r_matches_dense'])} | "
                     f"{xy('train')} | {xy('valid')} | {xy('test')} | {n_str} |")
    if results:
        variants = ["theirs", "ours", "ours_noscale", "ours_trainscale"]
        lines += ["", "## Stage 2: C-HMCNN model, test AUPRC (micro), mean ± std over seeds", "",
                  "| dataset | published | " + " | ".join(variants) + " |",
                  "|---|---|" + "---|" * len(variants)]
        for name, runs in results.items():
            cells = []
            for v in variants:
                vals = [r[v] for r in runs.values() if v in r]
                cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if vals else "—")
            lines.append(f"| {name} | {PUBLISHED.get(name, float('nan')):.3f} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated names")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+",
                        default=["theirs", "ours", "ours_noscale", "ours_trainscale"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_train", action="store_true", help="Stage 1 only")
    parser.add_argument("--epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()

    names = args.datasets.split(",") if args.datasets else GROUPS[args.group or "all"]
    os.makedirs(OUT_DIR, exist_ok=True)

    parity_path = os.path.join(OUT_DIR, "parity.json")
    results_path = os.path.join(OUT_DIR, "results.json")
    parity = json.load(open(parity_path, encoding="utf-8")) if os.path.exists(parity_path) else {}
    results = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else {}

    for name in names:
        print(f"[stage 1] {name}", flush=True)
        parity[name] = data_parity(name)
        print(f"  ok={_ok(parity[name])}", flush=True)
        with open(parity_path, "w", encoding="utf-8") as f:
            json.dump(parity, f, indent=2, default=float)

    if not args.skip_train:
        for name in names:
            variants = build_variants(name)
            runs = results.setdefault(name, {})
            for seed in args.seeds:
                run = runs.setdefault(str(seed), {})
                for v in args.variants:
                    t0 = time.time()
                    run[v] = train_chmcnn(name, variants[v], seed, args.device, args.epochs)
                    print(f"[stage 2] {name} seed {seed} {v:<16} AUPRC {run[v]:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, default=float)

    _write_summary({n: parity[n] for n in parity}, {n: results[n] for n in results})


if __name__ == "__main__":
    main()

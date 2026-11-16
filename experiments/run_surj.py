#!/usr/bin/env python3
"""Surj (Yang & Howe, WWW'22) reproduced with HMC-Torch.

Surj learns one vector per label offline with a graph autoencoder over the
ontology, maps samples into that space with an MLP, compares them with every
label by cosine similarity, and turns the similarities into probabilities
with a fully connected layer and a sigmoid, trained with BCE
(:class:`hmc.models.hierarchical.surj.SurjClassifier`).  Every run trains two
heads with the same network: ``surj`` (graph-autoencoder label vectors) and
``onehot`` (identity label vectors, i.e. without ontology learning).

Experiments (``--experiment``), each mirroring a part of the paper:

* ``main`` -- Tables 2, 3 and 5: the ARFF benchmarks under the protocol of
  ``run_protocol.py`` (early stopping and threshold on validation, refit on
  train + validation, three seeds) with Surj's learning rate (1e-3).  Raw and
  reconciled outputs, AUPRC, F1, Global Hierarchy Violation, and the time of
  each stage.
* ``literal`` -- the setup the paper describes: Adam at 1e-3, ten trials,
  training on the training split only, no reconciliation, epoch chosen on the
  validation AUPRC of the raw outputs.  Each run also records, as a
  diagnostic only, the best test AUPRC over all epochs (``oracle_test_auprc``):
  what choosing the epoch on the test split would report.
* ``evolving`` -- Table 4: 20% of the FunCat leaves are removed (V1); the
  model is trained and evaluated on the V1 ontology, then the autoencoder is
  retrained on the full ontology (V2) and the model retrained on the same V1
  labels and evaluated on every label.
* ``datasize`` -- Figure 5: 20/40/60/80% of the Gene Ontology training split
  (fixed subset, no refit).

Writes ``output/surj/<experiment>/<dataset>[_frac<f>].json``; existing files
are skipped.

Usage:
    PYTHONPATH=src python experiments/run_surj.py --experiment main --group funcat
    PYTHONPATH=src python experiments/run_surj.py --experiment main --datasets cellcycle_FUN --seeds 0 --max_epochs 3
"""

import argparse
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402  pylint: disable=wrong-import-position
    pretrain_label_gae,
)
from hmc.models.hierarchical.sparse_r import (  # noqa: E402  pylint: disable=wrong-import-position
    BlockDiagonalR,
)
from hmc.models.hierarchical.surj import (  # noqa: E402  pylint: disable=wrong-import-position
    SurjClassifier,
)
from hmc.utils.metrics.hierarchy_violation import (  # noqa: E402  pylint: disable=wrong-import-position
    global_hierarchy_violation,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    GO,
    OTHERS,
    auprc,
    hyperparams,
    load_dataset,
    predict,
    train_mlp,
)

# Surj's Table 2 (AUPRC), for the summary printed at the end of each dataset.
SURJ_REPORTED = {
    "cellcycle_FUN": 0.269, "derisi_FUN": 0.231, "eisen_FUN": 0.392, "expr_FUN": 0.382,
    "gasch1_FUN": 0.369, "gasch2_FUN": 0.273, "seq_FUN": 0.341, "spo_FUN": 0.241,
    "cellcycle_GO": 0.460, "derisi_GO": 0.445, "eisen_GO": 0.487, "expr_GO": 0.477,
    "gasch1_GO": 0.481, "gasch2_GO": 0.473, "seq_GO": 0.478, "spo_GO": 0.439,
    "enron_others": 0.743, "diatoms_others": 0.772,
    "imclef07a_others": 0.943, "imclef07d_others": 0.917,
}
# The 8 FunCat / 8 GO datasets of Surj's tables (church and pheno are not in it).
FUNCAT8 = [n for n in FUNCAT if n not in ("church_FUN", "pheno_FUN")]
GO8 = [n for n in GO if n != "pheno_GO"]
GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS, "surj": FUNCAT8 + GO8 + OTHERS}
HEADS = ("surj", "onehot")
SURJ_LR = 1e-3
EVOLVE_FRAC = 0.2
DATA_FRACS = (0.2, 0.4, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Pieces shared by every experiment
# ---------------------------------------------------------------------------


def label_vectors(data, args, device):
    """Graph-autoencoder label embeddings of ``data``'s hierarchy, and their time."""
    features = None
    if args.gae_features == "text":
        from run_label_embeddings import go_text_embeddings  # pylint: disable=import-outside-toplevel

        features = go_text_embeddings(data)
    t0 = time.time()
    emb = pretrain_label_gae(
        data["edges"], len(data["eval_mask"]), dim=args.gae_dim, epochs=args.gae_epochs,
        depth=data["depth"], device=device, loss=args.gae_loss, features=features,
    )
    return torch.from_numpy(emb).to(device), time.time() - t0


def model_fn_for(head, gae, hp, device):
    def build(d_in, d_out):
        return SurjClassifier(
            d_in, d_out, label_embeddings=gae if head == "surj" else None,
            hidden_dim=hp["hidden"], dropout=hp.get("dropout", 0.3),
        ).to(device)
    return build


def violations(data, scores, threshold):
    m = data["eval_mask"]
    return {
        "all": global_hierarchy_violation(scores, data["anc"], data["desc"], m),
        "relevant": global_hierarchy_violation(scores, data["anc"], data["desc"], m,
                                               threshold=threshold),
    }


def run_head(data, head, gae, seed, device, hp, on_epoch=None):
    """Train one head, add hierarchy violations, drop the score arrays."""
    run = train_mlp(data, "bce", seed, device, hp, model_fn=model_fn_for(head, gae, hp, device),
                    keep_scores=True, on_epoch=on_epoch)
    for mode in ("raw", "rec"):
        scores = run.pop(f"{mode}_test_scores")
        run["eval"][mode]["ghv"] = violations(data, scores, run["eval"][mode]["threshold"])
    run["head"] = head
    return run


def log_run(run, t0, extra=""):
    ev = run["eval"]
    print(f"  seed {run['seed']:>3} {run['head']:<7} ep {run['best_epoch']:>3}  "
          f"raw AUPRC {ev['raw']['auprc']:.4f} GHV {ev['raw']['ghv']['all']['violations']:>8} | "
          f"rec AUPRC {ev['rec']['auprc']:.4f} F1 {ev['rec']['f1']:.4f}{extra}  "
          f"({time.time() - t0:.0f}s)", flush=True)


def summary(record, mode):
    """Mean test AUPRC per head, printed after each dataset."""
    out = []
    for head in HEADS:
        vals = [r["eval"][mode]["auprc"] for r in record["runs"] if r["head"] == head]
        if vals:
            out.append(f"{head} {np.mean(vals):.4f}±{np.std(vals):.4f}")
    ref = SURJ_REPORTED.get(record["dataset"])
    return " | ".join(out) + (f" | Surj reported {ref:.3f}" if ref else "")


def base_record(data, name, hp, experiment, args, gae_time):
    return {
        "dataset": name, "experiment": experiment, "hyperparams": hp,
        "gae": {"loss": args.gae_loss, "features": args.gae_features, "dim": args.gae_dim,
                "epochs": args.gae_epochs, "time_s": gae_time},
        "n_train": len(data["y"][0]), "n_eval": int(data["eval_mask"].sum()),
        "surj_reported": SURJ_REPORTED.get(name), "runs": [],
    }


def dump(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=1, default=float)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def exp_main(name, args, device, path):
    data = load_dataset(name)
    hp = hyperparams(name) | {"lr": SURJ_LR}
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    gae, gae_time = label_vectors(data, args, device)
    record = base_record(data, name, hp, "main", args, gae_time)
    print(f"\n=== {name}: {record['n_eval']} labels, GAE {gae_time:.1f}s", flush=True)
    for seed in args.seeds:
        for head in HEADS:
            t0 = time.time()
            run = run_head(data, head, gae, seed, device, hp)
            record["runs"].append(run)
            log_run(run, t0)
    print(f"  rec: {summary(record, 'rec')}")
    dump(record, path)


def exp_literal(name, args, device, path):
    data = load_dataset(name)
    hp = hyperparams(name) | {"lr": SURJ_LR, "optimizer": "adam", "refit": False,
                              "select": "raw", "batch": 32, "dropout": 0.3}
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    gae, gae_time = label_vectors(data, args, device)
    record = base_record(data, name, hp, "literal", args, gae_time)
    m = data["eval_mask"]
    x_te, y_te = data["x"][2], data["y"][2][:, m]
    print(f"\n=== {name} (literal): {record['n_eval']} labels", flush=True)
    for seed in args.seeds:
        for head in HEADS:
            curve = []

            def on_epoch(_epoch, model, curve=curve):
                # Diagnostic only: never feeds back into training or selection.
                curve.append(auprc(y_te, predict(model, x_te, device)[:, m]))

            t0 = time.time()
            run = run_head(data, head, gae, seed, device, hp, on_epoch=on_epoch)
            run["test_auprc_curve"] = curve
            run["oracle_test_auprc"] = max(curve) if curve else None
            run["oracle_epoch"] = int(np.argmax(curve)) + 1 if curve else None
            record["runs"].append(run)
            log_run(run, t0, f" oracle {run['oracle_test_auprc']:.4f}@{run['oracle_epoch']}")
    print(f"  raw: {summary(record, 'raw')}")
    dump(record, path)


def subset_labels(data, keep):
    """``data`` restricted to the label columns ``keep`` (removed leaves only)."""
    keep = np.asarray(keep)
    remap = -np.ones(len(data["eval_mask"]), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    edges = remap[data["edges"]]
    edges = edges[(edges >= 0).all(1)]
    pairs = (remap[data["anc"]] >= 0) & (remap[data["desc"]] >= 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(keep)))
    graph.add_edges_from(map(tuple, edges.tolist()))
    return data | {
        "edges": edges,
        "y": [y[:, keep] for y in data["y"]],
        "eval_mask": data["eval_mask"][keep],
        "depth": data["depth"][keep],
        "anc": remap[data["anc"][pairs]],
        "desc": remap[data["desc"][pairs]],
        "sparse_r": BlockDiagonalR(graph, column_index={i: i for i in range(len(keep))}),
        "node_names": [data["node_names"][i] for i in keep],
    }


def exp_evolving(name, args, device, path):
    data = load_dataset(name)
    hp = hyperparams(name) | {"lr": SURJ_LR}
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    n = len(data["eval_mask"])
    parents = set(data["edges"][:, 1].tolist())
    leaves = np.array([j for j in np.nonzero(data["eval_mask"])[0] if j not in parents])
    removed = np.sort(np.random.default_rng(0).choice(
        leaves, int(round(EVOLVE_FRAC * len(leaves))), replace=False))
    keep = np.setdiff1d(np.arange(n), removed)

    # V1: the ontology without the removed leaves, and the labels that go with it.
    v1 = subset_labels(data, keep)
    # V2: the full ontology, but the training data are still those of V1.
    y_tr, y_val = data["y"][0].copy(), data["y"][1].copy()
    y_tr[:, removed] = 0
    y_val[:, removed] = 0
    v2 = data | {"y": [y_tr, y_val, data["y"][2]]}

    gae_v1, t_v1 = label_vectors(v1, args, device)
    gae_v2, t_v2 = label_vectors(v2, args, device)
    record = base_record(data, name, hp, "evolving", args, t_v1)
    record |= {"n_leaves": int(len(leaves)), "n_removed": int(len(removed)),
               "removed": removed.tolist(), "gae_v2_time_s": t_v2}
    print(f"\n=== {name} (evolving): {len(removed)} of {len(leaves)} leaves removed", flush=True)
    for seed in args.seeds:
        for version, d, gae in (("v1", v1, gae_v1), ("v2", v2, gae_v2)):
            for head in HEADS:
                t0 = time.time()
                run = run_head(d, head, gae, seed, device, hp) | {"ontology": version}
                record["runs"].append(run)
                log_run(run, t0, f" [{version}]")
    for head in HEADS:
        a = {v: np.mean([r["eval"]["rec"]["auprc"] for r in record["runs"]
                         if r["head"] == head and r["ontology"] == v]) for v in ("v1", "v2")}
        print(f"  {head}: V1 {a['v1']:.4f} -> V2 {a['v2']:.4f} ({100 * (a['v2'] / a['v1'] - 1):+.1f}%)")
    dump(record, path)


def exp_datasize(name, args, device, frac, path):
    data = load_dataset(name)
    hp = hyperparams(name) | {"lr": SURJ_LR, "refit": False}
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    n_train = len(data["y"][0])
    keep = np.sort(np.random.default_rng(0).choice(
        n_train, max(1, int(round(frac * n_train))), replace=False))
    data["x"] = [data["x"][0][keep], *data["x"][1:]]
    data["y"] = [data["y"][0][keep], *data["y"][1:]]
    gae, gae_time = label_vectors(data, args, device)
    record = base_record(data, name, hp, "datasize", args, gae_time) | {"train_frac": frac}
    print(f"\n=== {name} ({frac:.0%} of train: {len(keep)} samples)", flush=True)
    for seed in args.seeds:
        for head in HEADS:
            t0 = time.time()
            run = run_head(data, head, gae, seed, device, hp)
            record["runs"].append(run)
            log_run(run, t0)
    print(f"  rec: {summary(record, 'rec')}")
    dump(record, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_NAMES = {"main": GROUPS["surj"], "literal": GROUPS["surj"],
                 "evolving": FUNCAT8, "datasize": GO8}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=list(DEFAULT_NAMES), default="main")
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default 0,42,123; literal: 0..9)")
    parser.add_argument("--fracs", default=",".join(map(str, DATA_FRACS)),
                        help="Training fractions for --experiment datasize")
    parser.add_argument("--gae_loss", choices=["full", "sampled"], default="full")
    parser.add_argument("--gae_features", choices=["identity", "text"], default="identity",
                        help="Node features of the autoencoder ('text' needs GO datasets)")
    parser.add_argument("--gae_dim", type=int, default=128)
    parser.add_argument("--gae_epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="./output/surj")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()

    if args.datasets:
        names = args.datasets.split(",")
    elif args.group:
        names = GROUPS[args.group]
    else:
        names = DEFAULT_NAMES[args.experiment]
    default_seeds = "0,1,2,3,4,5,6,7,8,9" if args.experiment == "literal" else "0,42,123"
    args.seeds = [int(s) for s in (args.seeds or default_seeds).split(",")]
    device = torch.device(args.device)
    tag = "" if (args.gae_loss, args.gae_features) == ("full", "identity") else \
        f"_{args.gae_loss}_{args.gae_features}"
    out_dir = os.path.join(args.out_dir, args.experiment + tag)

    for name in names:
        if args.experiment == "datasize":
            for frac in (float(f) for f in args.fracs.split(",")):
                path = os.path.join(out_dir, f"{name}_frac{frac:g}.json")
                if os.path.exists(path):
                    print(f"[skip] {path} exists")
                    continue
                exp_datasize(name, args, device, frac, path)
            continue
        path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        {"main": exp_main, "literal": exp_literal, "evolving": exp_evolving}[args.experiment](
            name, args, device, path)


if __name__ == "__main__":
    main()

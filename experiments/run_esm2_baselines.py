#!/usr/bin/env python3
"""Spec 29: baselines with frozen ESM-2 features on ``seq_FUN`` / ``seq_GO``.

Methods (``--methods``): ``chmcnn`` (phase 1), ``hmcnf`` and ``gbdt`` (phase 2).

Our own arms (BCE / MCLoss, raw and reconciled, ESM-2 only) already live in
``output/protein_lm/<model>/<dataset>.json`` (``run_protein_lm.py``).  This script adds
the C-HMCNN arm under the *same* protocol and the *same* features:

  * identical ESM-2 matrices (embedding cache of ``run_protein_lm.py``, same ORF mapping,
    same train-mean imputation + indicator column, z-score on train);
  * C-HMCNN's own ``ConstrainedFFNNModel`` and R-matrix, imported from the local
    ``./C-HMCNN`` clone through ``compare_chmcnn_loader`` (GPL code is never copied here);
  * MCLoss (``mc`` objective of ``run_protocol.train_mlp``), early stopping and threshold
    on validation, refit on train + validation, test used once;
  * same hidden size / dropout / batch / lr as our arms (``run_protocol.hyperparams``).

It then writes ``output/protein_lm_baselines/<dataset>.json`` and ``summary.md`` with the
decision gate of the spec: our BCE + rec. must beat C-HMCNN by more than one sample
standard deviation (ddof = 1) on both datasets before Phase 2 is worth running.

Phase 2 adds HMCN-F (Wehrmann et al., ICML'18; written here from the paper: a global
highway over levels plus one local head per level, mixed with beta = 0.5) and the
one-vs-rest GBDT of ``run_protocol.run_gbdt`` (seed 0, Platt AUPRC on validation scores).

Usage:
    PYTHONPATH=src python experiments/run_esm2_baselines.py --datasets seq_FUN,seq_GO
    PYTHONPATH=src python experiments/run_esm2_baselines.py --methods hmcnf,gbdt
    PYTHONPATH=src python experiments/run_esm2_baselines.py --datasets seq_FUN --seeds 0 --max_epochs 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from compare_chmcnn_loader import (  # noqa: E402  pylint: disable=wrong-import-position
    ConstrainedFFNNModel,
    their_r,
)
from hmc.features.protein import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_PROTEIN_MODEL,
    ProteinFeatureEncoder,
    orf_embedding_features,
)
from hmc.features.yeast_orf import (  # noqa: E402  pylint: disable=wrong-import-position
    read_orf_ids,
    read_proteome,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    hyperparams,
    load_dataset,
    run_gbdt,
    train_mlp,
)

OUT_DIR = "output/protein_lm_baselines"
OURS_DIR = "output/protein_lm"
OBJECTIVE = {"ours": "bce", "chmcnn": "mc", "hmcnf": "bce"}
METHODS = {
    "chmcnn": "C-HMCNN(h): ConstrainedFFNNModel + MCLoss (./C-HMCNN, not copied)",
    "hmcnf": "HMCN-F (Wehrmann et al., ICML'18), written from the paper; BCE on P_G + per-level local BCE",
    "gbdt": "one-vs-rest HistGradientBoosting (run_protocol.run_gbdt), seed 0",
}


def adjacency(data):
    """Child -> parent adjacency (``a[i, j] = 1``: j is a parent of i) from the edge list."""
    n = len(data["eval_mask"])
    adj = np.zeros((n, n), dtype=np.float64)
    adj[data["edges"][:, 0], data["edges"][:, 1]] = 1.0
    return adj


class ChunkedEval(torch.nn.Module):
    """Runs C-HMCNN's model unchanged, but in small chunks outside training.

    Its ``get_constr_out`` expands the dense R to ``(batch, C, C)``; with 1024-row
    evaluation batches that is >100 GiB on ``seq_GO`` (C ~ 4,100).  Chunking changes the
    memory footprint only, not the outputs.
    """

    def __init__(self, model, chunk=8):
        super().__init__()
        self.model, self.chunk = model, chunk

    def forward(self, x):
        if self.training or len(x) <= self.chunk:
            return self.model(x)
        return torch.cat([self.model(x[i : i + self.chunk]) for i in range(0, len(x), self.chunk)])


def chmcnn_model_fn(data, hp, device):
    """Factory for C-HMCNN's constrained MLP (``ConstrainedFFNNModel``) with this data's R."""
    r = their_r(adjacency(data)).unsqueeze(0).to(device)
    cfg = {"num_layers": 3, "dropout": hp["dropout"], "non_lin": "relu"}
    return lambda d_in, d_out: ChunkedEval(ConstrainedFFNNModel(d_in, hp["hidden"], d_out, cfg, r))


class HMCNF(torch.nn.Module):
    """HMCN-F (Wehrmann et al., 2018), written from the paper.

    Level ``l`` has a global layer ``A_G^l = relu(W [A_G^{l-1}, x])`` and a local head
    ``P_L^l = sigmoid(W_L relu(W_T A_G^l))`` over the columns of that level; the global head
    ``P_G`` reads the last global layer.  Prediction: ``(1 - beta) P_G + beta P_L``.
    Training: BCE on ``P_G`` (the forward output) plus BCE on every ``P_L^l`` (``aux_loss``),
    which is the sum of losses of the original model.
    """

    def __init__(self, d_in, d_out, depth, hidden=384, dropout=0.5, beta=0.5):
        super().__init__()
        levels = sorted(set(int(d) for d in depth))
        self.cols = [torch.from_numpy(np.nonzero(depth == lv)[0]) for lv in levels]
        self.beta, self.drop = beta, torch.nn.Dropout(dropout)
        self.glob = torch.nn.ModuleList(
            torch.nn.Linear(d_in if i == 0 else hidden + d_in, hidden) for i in range(len(levels)))
        self.trans = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in levels)
        self.local = torch.nn.ModuleList(torch.nn.Linear(hidden, len(c)) for c in self.cols)
        self.head = torch.nn.Linear(hidden, d_out)
        self._pl = None

    def forward(self, x):
        a, pl = None, []
        for g, t, loc in zip(self.glob, self.trans, self.local):
            a = self.drop(torch.relu(g(x if a is None else torch.cat([a, x], dim=1))))
            pl.append(torch.sigmoid(loc(torch.relu(t(a)))))
        pg = torch.sigmoid(self.head(a))
        self._pl = pl
        if self.training:
            return pg
        local = torch.zeros_like(pg)
        for cols, p in zip(self.cols, pl):
            local[:, cols.to(pg.device)] = p
        return (1 - self.beta) * pg + self.beta * local

    def aux_loss(self, _xb, yb):
        """Sum of the per-level local BCE of the preceding training forward pass."""
        return sum(torch.nn.functional.binary_cross_entropy(p, yb[:, c.to(yb.device)])
                   for c, p in zip(self.cols, self._pl))


def hmcnf_model_fn(data, hp):
    """Factory for HMCN-F with the level of each column (``data["depth"]``)."""
    return lambda d_in, d_out: HMCNF(d_in, d_out, data["depth"], hidden=hp.get("hmcnf_hidden", 384),
                                     dropout=hp["dropout"])


def sd(vals):
    """Sample standard deviation (ddof = 1); 0 for a single value."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def ours_runs(name, model_tag):
    """Our ESM-only runs from ``run_protein_lm.py`` (``None`` when not run yet)."""
    path = os.path.join(OURS_DIR, model_tag, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f)["runs"] if r["features"] == "esm"]


def gate(ours, theirs):
    """Spec gate: ours (BCE + rec.) beats C-HMCNN by more than one sd of the better spread."""
    diff = np.mean(ours) - np.mean(theirs)
    spread = max(sd(ours), sd(theirs))
    return {"diff": float(diff), "spread": spread, "passes": bool(diff > spread)}


def _load(name, method):
    path = os.path.join(OUT_DIR, f"{name}.json" if method == "chmcnn" else f"{name}_{method}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(vals):
    return f"{np.mean(vals):.4f} ± {sd(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"


def summarize(names, model_tag):
    """Markdown table (all methods found on disk) and the phase-1 gate."""
    lines = ["# ESM-2 baselines: test AUPRC (micro), mean ± sd (ddof = 1) over seeds", "",
             "| dataset | MLP flat (BCE raw) | ours BCE + rec. | ours MCLoss + rec. | C-HMCNN | "
             "HMCN-F raw / rec. | GBDT raw / rec. / Platt (1 seed) | ours − C-HMCNN |", "|---|---|---|---|---|---|---|---|"]
    gates = {}
    for name in names:
        ours, ch = ours_runs(name, model_tag), _load(name, "chmcnn")
        if ours is None or ch is None:
            continue
        cell = lambda kind, mode: [r["eval"][mode]["auprc"] for r in ours if r["objective"] == kind]  # noqa: E731
        theirs = [r["eval"]["rec"]["auprc"] for r in ch["runs"]]
        g = gates[name] = gate(cell("bce", "rec"), theirs)
        hm, gb = _load(name, "hmcnf"), _load(name, "gbdt")
        hm_c = " / ".join(_fmt([r["eval"][m]["auprc"] for r in hm["runs"]]) for m in ("raw", "rec")) if hm else "—"
        gb_c = (" / ".join(f"{gb['runs'][0][k]['auprc']:.4f}" for k in ("raw", "rec"))
                + f" / {gb['runs'][0]['platt_auprc']:.4f}") if gb else "—"
        lines.append(f"| {name} | {_fmt(cell('bce', 'raw'))} | {_fmt(cell('bce', 'rec'))} | "
                     f"{_fmt(cell('mc', 'rec'))} | {_fmt(theirs)} | {hm_c} | {gb_c} | "
                     f"{g['diff']:+.4f} (sd {g['spread']:.4f}) |")
    if gates:
        ok = len(gates) == 2 and all(g["passes"] for g in gates.values())
        lines += ["", f"Phase-1 gate (both datasets, diff > 1 sd): **{'PASS' if ok else 'NOT passed'}**"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def run_dataset(name, args, encoder, proteome):
    """Train the requested baselines on the ESM-2 matrices of one dataset (all seeds)."""
    t0 = time.time()
    todo = {m: os.path.join(OUT_DIR, f"{name}.json" if m == "chmcnn" else f"{name}_{m}.json")
            for m in args.methods}
    records = {}
    for m, path in list(todo.items()):
        if os.path.exists(path):
            print(f"[skip] {name}/{m}: {path} exists")
            with open(path, encoding="utf-8") as f:
                records[m] = json.load(f)
            del todo[m]
    if not todo:
        return records
    data = load_dataset(name)
    ids = read_orf_ids(os.path.join(args.ids_dir, f"{name}_ids.tsv"))
    data["x"] = orf_embedding_features(data["x"], ids, encoder, proteome)
    hp = hyperparams(name)
    if args.max_epochs:
        hp["max_epochs"] = args.max_epochs
    base = {
        "dataset": name,
        "protein_model": encoder.model_name,
        "features": "esm",
        "n_features": int(data["x"][0].shape[1]),
        "unmatched_frac": {s: float(np.mean([o is None for o in ids[s]])) for s in ("train", "valid", "test")},
        "hyperparams": hp,
        "protocol": "train on train; early stopping and threshold on valid; refit; report on test",
    }
    for m, path in todo.items():
        t1 = time.time()
        record = base | {"method": METHODS[m], "runs": []}
        print(f"\n=== {name}: {m} on ESM-2 ({record['n_features']} features)", flush=True)
        if m == "gbdt":
            run = run_gbdt(data, seed=0)
            run |= {"seed": 0, "features": "esm", "method": m}
            record["runs"].append(run)
            print(f"  raw AUPRC {run['raw']['auprc']:.4f} rec {run['rec']['auprc']:.4f} "
                  f"platt {run['platt_auprc']:.4f} ({run['train_time_s']:.0f}s)", flush=True)
        else:
            model_fn = chmcnn_model_fn(data, hp, args.device) if m == "chmcnn" else hmcnf_model_fn(data, hp)
            kind = "mc" if m == "chmcnn" else "bce"
            for seed in args.seeds:
                run = train_mlp(data, kind, seed, args.device, hp, model_fn=model_fn)
                run["features"], run["method"] = "esm", m
                record["runs"].append(run)
                ev = run["eval"]
                print(f"  seed {seed:>3} ep {run['best_epoch']:>3}  raw AUPRC {ev['raw']['auprc']:.4f} "
                      f"rec {ev['rec']['auprc']:.4f}  F1 {ev['rec']['f1']:.4f} ({run['train_time_s']:.0f}s)",
                      flush=True)
        record["wall_time_s"] = time.time() - t1
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, default=float)
        records[m] = record
    print(f"[{name}] done in {time.time() - t0:.0f}s")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="seq_FUN,seq_GO")
    parser.add_argument("--model", default=DEFAULT_PROTEIN_MODEL)
    parser.add_argument("--methods", default="chmcnn", help="chmcnn,hmcnf,gbdt")
    parser.add_argument("--seeds", default="0,42,123")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--proteome", default="data/yeast/raw/orf_trans_all.fasta.gz")
    parser.add_argument("--ids_dir", default="data/yeast")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.methods = args.methods.split(",")
    args.device = torch.device(args.device)

    os.makedirs(OUT_DIR, exist_ok=True)
    proteome = read_proteome(args.proteome)
    encoder = ProteinFeatureEncoder(args.model)
    for name in args.datasets.split(","):
        run_dataset(name, args, encoder, proteome)
    summarize(args.datasets.split(","), encoder.model_name.split("/")[-1])


if __name__ == "__main__":
    main()

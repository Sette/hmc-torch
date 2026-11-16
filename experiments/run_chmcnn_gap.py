#!/usr/bin/env python3
"""Ablation from our ``global`` recipe to C-HMCNN's, and past it.

Every step adds to the previous one; nothing is chosen on the test split:
  * ``A`` -- our recipe: dropout 0.3, batch 32, train only, early stopping on validation;
  * ``B`` -- + dropout 0.7 (C-HMCNN);
  * ``C`` -- + batch 4 (C-HMCNN);
  * ``D`` -- + refit on train + validation for the epochs chosen on validation;
  * ``E`` -- D, averaging the scores of every seed (ensemble);
  * ``F`` -- D with hidden/lr/dropout picked by a small grid on validation (seed 0).

The model, R-matrix and MCLoss are C-HMCNN's (from ``compare_chmcnn_loader.py``);
features are z-scored with the statistics of the split the model is fit on.
Metric: micro AUPRC on the test split over ``to_eval``, as in C-HMCNN.

Usage:
    PYTHONPATH=src python experiments/run_chmcnn_gap.py --datasets cellcycle_FUN --seeds 0 --max_epochs 5
    PYTHONPATH=src python experiments/run_chmcnn_gap.py --group funcat
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from compare_chmcnn_loader import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    HIDDEN,
    PUBLISHED,
    ConstrainedFFNNModel,
    get_constr_out,
    load_ours,
    their_r,
)
from hmc.utils.datasets.scaling import (  # noqa: E402  pylint: disable=wrong-import-position
    standardize_splits,
)

OUT_DIR = "output/chmcnn_gap"
LADDER = {
    "A": {"dropout": 0.3, "batch": 32, "refit": False},
    "B": {"dropout": 0.7, "batch": 32, "refit": False},
    "C": {"dropout": 0.7, "batch": 4, "refit": False},
    "D": {"dropout": 0.7, "batch": 4, "refit": True},
}
GRID = {"hidden_mult": (0.5, 1.0, 2.0), "lr": (1e-4, 3e-4), "dropout": (0.5, 0.7)}


def load(name):
    """Raw (unscaled) arrays plus the hierarchy tensors."""
    mgr, _ = load_ours(name)
    train, valid, test = mgr.get_datasets()
    xs = [np.asarray(s.x, dtype=np.float32) for s in (train, valid, test)]
    ys = [np.asarray(s.y, dtype=np.float32) for s in (train, valid, test)]
    return {"x": xs, "y": ys, "r": their_r(np.asarray(mgr.a)).unsqueeze(0),
            "to_eval": torch.tensor(list(mgr.to_eval), dtype=torch.bool)}


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _predict(model, x, device, batch=1024):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i : i + batch].to(device)).cpu() for i in range(0, len(x), batch)])


def _auprc(y, scores, to_eval):
    return float(average_precision_score(y[:, to_eval], scores[:, to_eval], average="micro"))


def fit(x, y, hp, seed, device, epochs, r, to_eval, x_val=None, y_val=None, patience=20):
    """C-HMCNN's training step; early stopping on validation AUPRC when ``x_val`` is given.

    Returns ``(model, best_epoch, best_val_auprc)``.
    """
    _seed(seed)
    r = r.to(device)
    model = ConstrainedFFNNModel(x.shape[1], hp["hidden"], y.shape[1],
                                 {"num_layers": 3, "dropout": hp["dropout"], "non_lin": "relu"}, r).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    criterion = nn.BCELoss()
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    mask = to_eval.to(device)
    best, best_state, best_epoch, bad = -1.0, None, epochs, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i : i + hp["batch"]]
            labels = y[idx].double()
            opt.zero_grad()
            output = model(x[idx])
            constr_output = get_constr_out(output, r)
            train_output = get_constr_out(labels * output.double(), r)
            train_output = (1 - labels) * constr_output + labels * train_output
            loss = criterion(train_output[:, mask], labels[:, mask])
            loss.backward()
            opt.step()
        if x_val is None:
            continue
        score = _auprc(y_val, _predict(model, torch.from_numpy(x_val), device).numpy(), to_eval)
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best


def run_recipe(data, hp, seed, device, max_epochs):
    """Train one recipe; return validation-selected epoch/score and test scores."""
    (x_tr, x_va, _), (y_tr, y_va, y_te) = data["x"], data["y"]
    r, to_eval = data["r"], data["to_eval"]
    xs = standardize_splits(x_tr, x_va, data["x"][2])
    model, best_epoch, best_val = fit(xs[0], y_tr, hp, seed, device, max_epochs, r, to_eval,
                                      x_val=xs[1], y_val=y_va)
    x_test = xs[2]
    if hp["refit"]:
        x_full, x_test = standardize_splits(np.concatenate((x_tr, x_va)), data["x"][2])
        model, _, _ = fit(x_full, np.concatenate((y_tr, y_va)), hp, seed, device, best_epoch, r, to_eval)
    scores = _predict(model, torch.from_numpy(x_test), device).numpy()
    return {"best_epoch": best_epoch, "val": best_val, "test": _auprc(y_te, scores, to_eval)}, scores


def run_dataset(name, seeds, device, max_epochs):
    """Run the whole ladder (A-F) for one dataset."""
    data_name, family = name.rsplit("_", 1)
    base = {"hidden": HIDDEN[family][data_name], "lr": 1e-4}
    data = load(name)
    y_te, to_eval = data["y"][2], data["to_eval"]
    res = {}
    d_scores = []
    for step, recipe in LADDER.items():
        res[step] = []
        for seed in seeds:
            t0 = time.time()
            run, scores = run_recipe(data, base | recipe, seed, device, max_epochs)
            res[step].append(run)
            if step == "D":
                d_scores.append(scores)
            print(f"  {name} {step} seed {seed}: ep {run['best_epoch']:>3} val {run['val']:.4f} "
                  f"test {run['test']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res["E"] = [{"test": _auprc(y_te, np.mean(d_scores, axis=0), to_eval), "n_models": len(d_scores)}]
    print(f"  {name} E ensemble of {len(d_scores)}: test {res['E'][0]['test']:.4f}", flush=True)

    # F: pick hidden/lr/dropout on validation (seed 0, train only), then refit like D.
    grid = []
    for mult, lr, dropout in itertools.product(*GRID.values()):
        hp = base | LADDER["D"] | {"hidden": int(base["hidden"] * mult), "lr": lr, "dropout": dropout}
        run, _ = run_recipe(data, hp | {"refit": False}, seeds[0], device, max_epochs)
        grid.append({"hp": hp, "val": run["val"]})
        print(f"  {name} F grid h{hp['hidden']} lr{lr:g} d{dropout}: val {run['val']:.4f}", flush=True)
    best_hp = max(grid, key=lambda g: g["val"])["hp"]
    res["F"] = [run_recipe(data, best_hp, seed, device, max_epochs)[0] for seed in seeds]
    res["F_hp"] = best_hp
    res["F_grid"] = grid
    print(f"  {name} F {best_hp}: test {np.mean([r['test'] for r in res['F']]):.4f}", flush=True)
    return res


def write_summary(results):
    """Write ``summary.md`` with mean ± std of the test AUPRC per step."""
    steps = ["A", "B", "C", "D", "E", "F"]
    lines = ["# From our recipe to C-HMCNN's: test AUPRC (micro), mean ± std over seeds", "",
             "| dataset | C-HMCNN | " + " | ".join(steps) + " |", "|---|---|" + "---|" * len(steps)]
    for name, res in results.items():
        cells = []
        for s in steps:
            vals = [r["test"] for r in res.get(s, [])]
            cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1
                         else f"{vals[0]:.3f}" if vals else "—")
        lines.append(f"| {name} | {PUBLISHED.get(name, float('nan')):.3f} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--group", choices=["funcat"], default="funcat")
    parser.add_argument("--datasets", default=None, help="Comma-separated names")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_epochs", type=int, default=300)
    args = parser.parse_args()

    names = args.datasets.split(",") if args.datasets else FUNCAT
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for name in names:
        path = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results[name] = json.load(f)
            print(f"[skip] {name} (delete {path} to redo)")
            continue
        print(f"[run] {name}", flush=True)
        results[name] = run_dataset(name, args.seeds, args.device, args.max_epochs)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results[name], f, indent=2, default=float)
    write_summary(results)


if __name__ == "__main__":
    main()

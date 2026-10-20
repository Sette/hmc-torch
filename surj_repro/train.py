#!/usr/bin/env python3
"""Train one recipe (``recipes.json``) on C-HMCNN's benchmarks; one JSON + NPZ per seed.

Data: C-HMCNN's parser and preprocessing (``data.py``).  Networks: C-HMCNN's, or Surj's head
with one-hot or GAE label embeddings (``model.py``).  Every knob is in the recipe; ``--set``
overrides fields for one-off runs, and the output folder name records the overrides.

Outputs, under ``surj_repro/results/<run name>/<dataset>/``:
  * ``seed<k>.json`` -- recipe, provenance (commit, environment, ARFF checksums), epochs,
    timings, test metrics (C-HMCNN's micro AP and the metric hypotheses of ``metrics.py``),
    Global Hierarchy Violation, and two diagnostics that are never results: the test AP over
    the train-only epochs (``test_curve``) and its maximum (epoch picked on test);
  * ``seed<k>.npz`` -- the test scores (float16) for recomputing any metric.
Existing seed files are skipped, so runs resume and can be split across processes.

Usage (from the repository root):
    PYTHONPATH=src python surj_repro/train.py --recipe chmcnn_fixed --datasets cellcycle_FUN --seeds 0
    PYTHONPATH=src python surj_repro/train.py --recipe L4_surj_lr3 --group funcat --seeds 0,1,2
    PYTHONPATH=src python surj_repro/train.py --recipe selected --group funcat --evolve
    PYTHONPATH=src python surj_repro/train.py --recipe selected --group go --frac 0.4
    PYTHONPATH=src python surj_repro/train.py --recipe L4_surj_lr3 --set batch=32 --set lr=3e-4 \
        --datasets eisen_FUN --seeds 0 --no_test --name hp_eisen
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import argparse
import dataclasses
import json
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_recipe, parse_sets, provenance  # noqa: E402
from data import (  # noqa: E402
    FUNCAT,
    GO,
    OTHERS,
    arff_files,
    leaves,
    load,
    published_epochs,
    subsample,
    subset_labels,
)
from gae import train_gae  # noqa: E402
from metrics import hypotheses, micro_ap  # noqa: E402
from model import ChmcnnNet, Constraint, SurjNet  # noqa: E402

from hmc.utils.metrics.hierarchy_violation import global_hierarchy_violation  # noqa: E402

GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS, "all": FUNCAT + OTHERS + GO}
EVOLVE_FRAC = 0.2
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def seed_all(seed: int) -> None:
    """C-HMCNN/main.py's seeding, plus deterministic kernels."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Trainer:
    """One recipe on one (possibly modified) dataset: fit, select epochs, score."""

    def __init__(self, data: dict, recipe, device, z: np.ndarray | None):
        self.data, self.recipe, self.device = data, recipe, device
        self.z = None if z is None else torch.from_numpy(z)
        self.constraint = Constraint(data["a"], device, sparse=data["is_go"])
        self.batch_eval = 256 if data["is_go"] else 32
        self.mask = data["to_eval"]
        self.mask_t = torch.from_numpy(self.mask).to(device)
        self.x = [torch.from_numpy(x).to(device) for x in data["x"]]
        self.y = [torch.from_numpy(y).to(device) for y in data["y"]]
        r = recipe
        self.hidden = data["hidden_dim"] if r.hidden == "chmcnn" else int(r.hidden)

    def _net(self, seed: int) -> nn.Module:
        seed_all(seed)
        n_in, n_out, r = self.x[0].shape[1], self.y[0].shape[1], self.recipe
        if r.net == "chmcnn":
            net = ChmcnnNet(n_in, self.hidden, n_out, r.dropout)
        elif r.net in ("surj", "onehot"):
            net = SurjNet(n_in, self.hidden, n_out, self.z if r.net == "surj" else None, r.dropout)
        else:
            raise ValueError(f"unknown net {r.net!r}")
        return net.to(self.device)

    def _train_epoch(self, net, opt, x, y, gen):
        net.train()
        perm = torch.randperm(len(x), generator=gen).to(self.device)
        for i in range(0, len(perm), self.recipe.batch):
            idx = perm[i:i + self.recipe.batch]
            out, labels = net(x[idx]), y[idx]
            if self.recipe.mcloss:
                labels = labels.double()
                out = self.constraint.mcloss_output(out, labels)
            loss = nn.functional.binary_cross_entropy(out[:, self.mask_t], labels[:, self.mask_t])
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(self, net, split: int) -> np.ndarray:
        net.eval()
        out = []
        for i in range(0, len(self.x[split]), self.batch_eval):
            s = net(self.x[split][i:i + self.batch_eval])
            out.append((self.constraint(s) if self.recipe.mcm else s).float().cpu())
        return torch.cat(out).numpy()

    def fit(self, seed: int, splits: tuple[int, ...], epochs: int, watch: bool = False,
            watch_test: bool = True):
        """Train on ``splits``; with ``watch``, early-stop on validation AP."""
        net = self._net(seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.recipe.lr,
                               weight_decay=self.recipe.weight_decay)
        gen = torch.Generator().manual_seed(seed)
        x, y = torch.cat([self.x[s] for s in splits]), torch.cat([self.y[s] for s in splits])
        best, best_epoch, best_state, bad = -1.0, epochs, None, 0
        val_curve, test_curve = [], []
        for epoch in range(1, epochs + 1):
            self._train_epoch(net, opt, x, y, gen)
            if not watch:
                continue
            val_curve.append(micro_ap(self.data["y"][1], self.predict(net, 1), self.mask))
            if watch_test:
                test_curve.append(micro_ap(self.data["y"][2], self.predict(net, 2), self.mask))
            if val_curve[-1] > best:
                best, best_epoch, bad = val_curve[-1], epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.recipe.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch, val_curve, test_curve

    def run(self, seed: int, name: str, with_test: bool = True) -> tuple[dict, np.ndarray | None]:
        r, out = self.recipe, {"seed": seed}
        t0 = time.time()
        if r.epochs == "published":
            out["epochs"] = published_epochs(name)
            net, *_ = self.fit(seed, (0, 1), out["epochs"])
        elif r.epochs in ("select", "select_norefit"):
            net, out["epochs"], val_curve, test_curve = self.fit(
                seed, (0,), r.max_epochs, watch=True, watch_test=with_test)
            out["val_ap"] = val_curve[out["epochs"] - 1]
            out["val_curve"] = val_curve
            out["time_select_s"] = time.time() - t0
            if with_test:
                out["test_ap_no_refit"] = test_curve[out["epochs"] - 1]
                out["diag_test_curve"] = test_curve
                out["diag_oracle_test_ap"] = max(test_curve)
            if r.epochs == "select":
                t0 = time.time()
                net, *_ = self.fit(seed, (0, 1), out["epochs"])
        else:
            raise ValueError(f"unknown epochs policy {r.epochs!r}")
        out["time_train_s"] = time.time() - t0
        if not with_test:
            return out, None
        y, s = self.data["y"][2], self.predict(net, 2)
        out["test_ap"] = micro_ap(y, s, self.mask)
        out["metrics"] = hypotheses(y, s, self.mask)
        anc, desc = self.data["anc"], self.data["desc"]
        out["ghv"] = global_hierarchy_violation(s, anc, desc, self.mask)
        out["ghv_relevant"] = global_hierarchy_violation(s, anc, desc, self.mask, threshold=0.5)
        return out, s


def evolve(data: dict) -> tuple[dict, dict, np.ndarray]:
    """V1 = ``data`` without 20% of its evaluated leaves (seed 0); V2 = full ontology, V1 labels."""
    cand = np.intersect1d(leaves(data["a"]), np.nonzero(data["to_eval"])[0])
    rng = np.random.default_rng(0)
    removed = np.sort(rng.choice(cand, int(round(EVOLVE_FRAC * len(cand))), replace=False))
    v1 = subset_labels(data, np.setdiff1d(np.arange(len(data["a"])), removed))
    y = list(data["y"])
    for split in (0, 1):
        y[split] = y[split].copy()
        y[split][:, removed] = 0.0
    return v1, data | {"y": tuple(y)}, removed


def run_name(args) -> str:
    if args.name:
        return args.name
    name = args.recipe
    for item in args.set or []:
        name += "__" + item.replace("=", "-")
    if args.frac < 1:
        name += f"__frac-{args.frac:g}"
    if args.evolve:
        name += "__evolve"
    return name


def main():  # pylint: disable=too-many-locals
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--set", action="append", help="recipe override, key=value (repeatable)")
    ap.add_argument("--datasets")
    ap.add_argument("--group", choices=GROUPS)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--no_test", action="store_true",
                    help="validation only (hyperparameter search): test is never scored")
    ap.add_argument("--name", help="output folder name (default: recipe + overrides)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    recipe = load_recipe(args.recipe, parse_sets(args.set))
    seeds = [int(s) for s in args.seeds.split(",")]
    names = args.datasets.split(",") if args.datasets else GROUPS[args.group or "funcat"]
    device = args.device if torch.cuda.is_available() else "cpu"
    name = run_name(args)

    for ds in names:
        folder = os.path.join(args.out_dir, name, ds)
        todo = [s for s in seeds if not os.path.exists(os.path.join(folder, f"seed{s}.json"))]
        if not todo:
            print(f"skip {folder}")
            continue
        os.makedirs(folder, exist_ok=True)
        data = load(ds)
        if args.frac < 1:
            data = subsample(data, args.frac)
        settings = {"full": data}
        if args.evolve:
            v1, v2, removed = evolve(data)
            settings = {"V1": v1, "V2": v2}
        print(f"\n=== {ds} [{name}] {dataclasses.asdict(recipe)}", flush=True)
        trainers, gae_time = {}, {}
        for key, d in settings.items():
            z = None
            if recipe.net == "surj":
                t0 = time.time()
                z = train_gae(d["a"], dim=recipe.gae_dim, epochs=recipe.gae_epochs, seed=0,
                              device=device)
                gae_time[key] = time.time() - t0
            trainers[key] = Trainer(d, recipe, device, z)
        prov = provenance(arff_files(ds))
        for seed in todo:
            record = {"dataset": ds, "run": name, "recipe_name": args.recipe,
                      "recipe": dataclasses.asdict(recipe), "frac": args.frac,
                      "provenance": prov, "gae_time_s": gae_time, "settings": {}}
            if args.evolve:
                record["removed"] = [data["terms"][i] for i in removed]
            scores = {}
            for key, trainer in trainers.items():
                res, s = trainer.run(seed, ds, with_test=not args.no_test)
                record["settings"][key] = res
                if s is not None:
                    scores[key] = s.astype(np.float16)
                msg = (f"test {res['test_ap']:.4f} (no refit {res.get('test_ap_no_refit', 0):.4f})"
                       f"  GHV {res['ghv']['violations']}" if "test_ap" in res
                       else f"val {res['val_ap']:.4f}")
                print(f"  {key:<4} seed {seed}  ep {res['epochs']:>3}  {msg}  "
                      f"{res.get('time_select_s', 0) + res['time_train_s']:.0f}s", flush=True)
            with open(os.path.join(folder, f"seed{seed}.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1, default=float)
            if scores:
                np.savez_compressed(os.path.join(folder, f"seed{seed}.npz"), **scores)


if __name__ == "__main__":
    main()

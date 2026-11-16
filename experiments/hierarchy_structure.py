#!/usr/bin/env python3
"""How much room each dataset leaves for reconciliation.

Per dataset, on the test split:
  * ``internal_label_share`` -- evaluated labels that are internal nodes;
  * ``internal_positive_share`` -- positive (sample, label) pairs on internal
    nodes, the bound that matters for micro-averaged metrics;
  * ``changed_share`` -- (sample, internal node) scores that reconciliation
    actually raises, for a BCE model trained exactly as in
    ``run_protocol.py`` (seed 0), i.e. how often the model violates the
    hierarchy; ``mean_change`` is the average size of those raises.

Writes ``output/protocol/structure.json``, read by ``make_paper_artifacts.py``.

Usage:
    PYTHONPATH=src python experiments/hierarchy_structure.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402  pylint: disable=wrong-import-position
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    GO,
    OTHERS,
    TEXT,
    _fit,
    hyperparams,
    load_dataset,
    predict,
)


def main():
    out_path = "output/protocol/structure.json"
    result = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name in FUNCAT + GO + OTHERS + TEXT:
        if "changed_share" in result.get(name, {}):
            continue
        data = load_dataset(name)
        m = data["eval_mask"]
        y = data["y"][2]
        _, parent_cols = data["sparse_r"]._edge_columns("cpu")  # pylint: disable=protected-access
        internal = np.zeros(len(m), dtype=bool)
        internal[np.unique(parent_cols.numpy())] = True
        internal &= m
        pos = y[:, m].sum()
        result[name] = {
            "internal_label_share": float(internal.sum() / m.sum()),
            "internal_positive_share": float(y[:, internal].sum() / pos),
        }
        # Violations of a BCE model trained as in run_protocol.py (seed 0).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hp = hyperparams(name)
        x_tr, x_val, x_te = data["x"]
        y_tr, y_val = data["y"][0], data["y"][1]
        model, best_epoch, _ = _fit(data, "bce", 0, device, hp, x_tr, y_tr,
                                    hp["max_epochs"], x_val=x_val)
        if hp["refit"]:
            model, _, _ = _fit(data, "bce", 0, device, hp,
                               np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val]),
                               best_epoch)
        raw = predict(model, x_te, device)
        rec = data["sparse_r"].reconcile(torch.from_numpy(raw)).numpy()
        change = (rec - raw)[:, internal]
        raised = change > 1e-6
        result[name]["changed_share"] = float(raised.mean())
        result[name]["mean_change"] = float(change[raised].mean()) if raised.any() else 0.0
        print(name, result[name], flush=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()

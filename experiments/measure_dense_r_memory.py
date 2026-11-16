#!/usr/bin/env python3
"""Peak GPU memory of dense vs sparse R-matrix reconciliation on a GO hierarchy.

Backs the scaling claims of the paper with measurements instead of estimates.
For each batch size it runs one reconciliation step, with and without a
backward pass, through:

* ``dense_f64`` -- ``get_constr_out`` as shipped (C-HMCNN's float64 broadcast);
* ``dense_f32`` -- the same broadcast kept in float32;
* ``sparse``    -- ``BlockDiagonalR.reconcile``.

It records the peak allocated memory or the out-of-memory error, and checks
that dense and sparse outputs agree.  Scores are random: memory does not
depend on their values.

Usage:
    PYTHONPATH=src python experiments/measure_dense_r_memory.py --dataset seq_GO
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402  pylint: disable=wrong-import-position
from run_protocol import load_dataset  # noqa: E402  pylint: disable=wrong-import-position

from hmc.models.global_classifier.constraint.utils import (  # noqa: E402  pylint: disable=wrong-import-position
    get_constr_out,
)


def dense_f32(x, r_matrix):
    """``get_constr_out`` without the cast to float64."""
    n = r_matrix.shape[1]
    c_out = x.unsqueeze(1).expand(len(x), n, n)
    r_batch = r_matrix.expand(len(x), n, n)
    return torch.max(r_batch * c_out, dim=2)[0]


def measure(fn, batch, n_classes, device, backward):
    """Run one step and return peak allocated bytes, or the OOM message."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    try:
        x = torch.rand(batch, n_classes, device=device, requires_grad=backward)
        with torch.set_grad_enabled(backward):
            out = fn(x)
            if backward:
                out.sum().backward()
        peak = torch.cuda.max_memory_allocated(device) - base
        result = {"ok": True, "peak_gb": peak / 1024**3}
    except torch.OutOfMemoryError as err:
        result = {"ok": False, "error": str(err).split(". ")[0]}
    for name in ("x", "out"):
        locals().pop(name, None)
    torch.cuda.empty_cache()
    return result


def main():
    """Measure every (operator, batch, backward) cell and write the JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="seq_GO")
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16, 32, 64, 128, 256])
    parser.add_argument("--out", default="output/scaling/dense_r_memory.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    data = load_dataset(args.dataset)
    sparse_r = data["sparse_r"]
    n_classes = sparse_r.n_nodes
    # R[i, j] = 1 iff i is j or an ancestor of j: reconcile one-hot rows.
    r_cpu = sparse_r.reconcile(torch.eye(n_classes)).T.contiguous()
    r32 = r_cpu.to(device).unsqueeze(0)
    child, _ = sparse_r._edge_columns("cpu")  # pylint: disable=protected-access

    x = torch.rand(4, n_classes, device=device)
    diff = (get_constr_out(x, r32).float() - sparse_r.reconcile(x)).abs().max().item()

    ops = {
        "dense_f64": lambda t: get_constr_out(t, r32),
        "dense_f32": lambda t: dense_f32(t, r32),
        "sparse": sparse_r.reconcile,
    }
    result = {
        "dataset": args.dataset,
        "n_classes": n_classes,
        "n_edges": int(len(child)),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "r_dense_f32_mb": r_cpu.numel() * 4 / 1024**2,
        "max_abs_diff_dense_vs_sparse": diff,
        "cells": [],
    }
    for name, fn in ops.items():
        for backward in (False, True):
            for batch in args.batches:
                cell = {"op": name, "batch": batch, "backward": backward}
                cell.update(measure(fn, batch, n_classes, device, backward))
                result["cells"].append(cell)
                print(cell, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=float)
    print("saved", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fine-tune SPECTER2 end-to-end on WOS under the protocol of the published baselines.

Protocol (as in HPT, EMNLP'22): train on the 30,070 training documents only,
use the 7,518 validation documents to pick the epoch and the decision
threshold, and report on the 9,397 test documents.  The model is evaluated on
its raw outputs and after bottom-up reconciliation (the R-matrix constraint).

Usage:
    PYTHONPATH=src python experiments/run_wos_e2e.py --seed 0
"""

import argparse
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from hmc.datasets.dataset_manager import (  # noqa: E402  pylint: disable=wrong-import-position
    initialize_dataset_experiments,
)
from hmc.models.global_classifier.e2e.model import (  # noqa: E402  pylint: disable=wrong-import-position
    E2EConstrainedModel,
)
from hmc.models.hierarchical.sparse_r import (  # noqa: E402  pylint: disable=wrong-import-position
    BlockDiagonalR,
)
from hmc.utils.model_cache import (  # noqa: E402  pylint: disable=wrong-import-position
    ensure_transformer_model_cached,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    auprc,
    pick_threshold,
    prf,
)


def load_texts(fname):
    with open(f"data/wos/{fname}", encoding="utf-8") as f:
        return [json.loads(line)["token"] for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default="./output/protocol")
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"wos_e2e_seed{args.seed}.json")
    lock_path = out_path + ".lock"
    if os.path.exists(out_path) or os.path.exists(lock_path):
        print(f"[skip] {out_path} exists or is being produced")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        _run(args, out_path)
    finally:
        os.remove(lock_path)


def _run(args, out_path):
    """Train, select on validation, and evaluate on test for one seed."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    mgr = initialize_dataset_experiments(
        "wos", device="cpu", dataset_path="./data", is_global=True,
        model_cache_dir="./models", arxiv_load_features=False,
    )
    train, valid, test = mgr.get_datasets()
    y = [np.asarray(s.y, dtype=np.float32) for s in (train, valid, test)]
    texts = [load_texts(f) for f in (
        "WebOfScience_train.json", "WebOfScience_dev.json", "WebOfScience_test.json")]
    assert [len(t) for t in texts] == [len(v) for v in y], "text/label split mismatch"
    m = np.asarray(mgr.to_eval, dtype=bool)
    n_nodes = mgr.output_dim

    adj = np.asarray(mgr.a)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(zip(*[v.tolist() for v in np.nonzero(adj)]))
    sparse_r = BlockDiagonalR(graph, column_index={i: i for i in range(n_nodes)})

    tokenizer = AutoTokenizer.from_pretrained(
        ensure_transformer_model_cached("allenai/specter2_base", "./models"),
        local_files_only=True,
    )

    def encode(batch):
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # Identity R inside the model: we take raw scores and reconcile ourselves.
    model = E2EConstrainedModel(
        model_name="allenai/specter2_base", output_dim=n_nodes,
        r_matrix=torch.eye(n_nodes, dtype=torch.float64).unsqueeze(0).to(device),
        hidden_dim=512, num_layers=3, dropout=0.3, model_cache_dir="./models",
    ).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.transformer.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-5,
    )
    crit = nn.BCELoss()

    @torch.no_grad()
    def scores_for(split):
        model.eval()
        out = []
        for i in range(0, len(texts[split]), 32):
            out.append(model(*encode(texts[split][i : i + 32])).float().cpu())
        raw = torch.cat(out)
        return raw.numpy(), sparse_r.reconcile(raw).numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, f".wos_e2e_seed{args.seed}.pt")
    history, best = [], None
    t0 = time.time()
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(texts[0]), generator=gen).tolist()
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            preds = model(*encode([texts[0][j] for j in idx]))
            loss = crit(preds.float(), torch.from_numpy(y[0][idx]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        _, val_rec = scores_for(1)
        thr = pick_threshold(y[1][:, m], val_rec[:, m])
        val_f1 = prf(y[1][:, m], val_rec[:, m], thr)[2]
        history.append({"epoch": epoch + 1, "val_f1": val_f1, "threshold": thr,
                        "elapsed_s": time.time() - t0})
        print(f"epoch {epoch + 1}: val micro-F1 {val_f1:.4f} (thr {thr})", flush=True)
        if best is None or val_f1 > best["val_f1"]:
            best = {"epoch": epoch + 1, "val_f1": val_f1}
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt))
    val_raw, val_rec = scores_for(1)
    test_raw, test_rec = scores_for(2)
    result = {"seed": args.seed, "best_epoch": best["epoch"], "history": history,
              "train_time_s": time.time() - t0,
              "n_train": len(texts[0]), "n_valid": len(texts[1]), "n_test": len(texts[2])}
    for mode, (sv, st) in {"raw": (val_raw, test_raw), "rec": (val_rec, test_rec)}.items():
        thr = pick_threshold(y[1][:, m], sv[:, m])
        p, r, f1 = prf(y[2][:, m], st[:, m], thr)
        result[mode] = {"f1": f1, "precision": p, "recall": r, "threshold": thr,
                        "auprc": auprc(y[2][:, m], st[:, m])}
        print(f"test {mode}: micro-F1 {f1:.4f} AUPRC {result[mode]['auprc']:.4f} (thr {thr})")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    os.remove(ckpt)


if __name__ == "__main__":
    main()

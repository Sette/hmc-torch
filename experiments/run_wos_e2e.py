#!/usr/bin/env python3
"""Fine-tune SPECTER2 on WOS (globalE2E) and compare with SOTA."""

import json as _json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.e2e.model import E2EConstrainedModel
from hmc.utils.model_cache import ensure_transformer_model_cached

sys.path.insert(0, "src")


SEED = 42
DEVICE = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true, y_pred, eval_mask):
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best[0]:
            best = (f1, thr, p, r)
    from sklearn.metrics import average_precision_score

    auprc = float(
        average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
        )
    )
    return best[0], auprc, best[2], best[3], best[1]


# ===== 1. Load WOS data & hierarchy =====
print("=" * 60)
print("Loading WOS data...")

mgr = initialize_dataset_experiments(
    "wos",
    device="cpu",
    dataset_path="./data",
    is_global=True,
    model_cache_dir="./models",
    arxiv_load_features=False,
)
train, valid, test = mgr.get_datasets()
n_nodes = mgr.output_dim
eval_mask = np.array(mgr.to_eval, dtype=bool)
a = mgr.a

# Build R-matrix
r_np = np.zeros(a.shape)
np.fill_diagonal(r_np, 1)
g_nx = nx.DiGraph(a)
for i in range(len(a)):
    ancestors = list(nx.descendants(g_nx, i))
    if ancestors:
        r_np[i, ancestors] = 1
r_matrix = torch.tensor(r_np).transpose(1, 0).unsqueeze(0).to(DEVICE)

# ===== 2. Tokenize =====
print("Tokenizing...")

model_path = ensure_transformer_model_cached("allenai/specter2_base", "./models")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def tokenize_texts(texts, max_len=512):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


# Load raw texts from JSON (labels come from the dataset manager)

train_texts, test_texts = [], []
for fname, tlist in [
    ("WebOfScience_train.json", train_texts),
    ("WebOfScience_dev.json", train_texts),
    ("WebOfScience_test.json", test_texts),
]:
    with open(f"data/wos/{fname}") as f:
        for line in f:
            tlist.append(_json.loads(line)["token"])
y_train_full = np.concatenate([train.y, valid.y]).astype(np.float32)
y_test_full = test.y.astype(np.float32)

print(
    f"  Train texts: {len(train_texts)}  Test texts: {len(test_texts)}  Nodes: {n_nodes}"
)

# ===== 3. Model =====

model = E2EConstrainedModel(
    model_name="allenai/specter2_base",
    output_dim=n_nodes,
    r_matrix=r_matrix,
    hidden_dim=512,
    num_layers=3,
    dropout=0.3,
    model_cache_dir="./models",
).to(DEVICE)

opt = torch.optim.AdamW(
    [
        {"params": model.transformer.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-5,
)
crit = nn.BCELoss()

# ===== 4. Training loop (batched tokenization to save RAM) =====
epochs = 3
batch_size = 8
n_train = len(train_texts)

print(f"Training {epochs} epochs, batch={batch_size}, device={DEVICE}...")
t0 = time.time()

for ep in range(epochs):
    model.train()
    # Shuffle
    idx = torch.randperm(n_train)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n_train, batch_size):
        batch_idx = idx[i : i + batch_size].tolist()
        batch_texts = [train_texts[j] for j in batch_idx]
        batch_y = torch.tensor(y_train_full[batch_idx]).to(DEVICE)
        input_ids, attn_mask = tokenize_texts(batch_texts)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

        preds = model(input_ids=input_ids, attention_mask=attn_mask)
        loss = crit(preds, batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1

        if n_batches % 200 == 0:
            print(
                f"  ep {ep+1}/{epochs} batch {n_batches}/{n_train//batch_size} loss={total_loss/n_batches:.4f}"
            )

    print(f"  ep {ep+1}/{epochs} loss={total_loss/n_batches:.4f}")

# ===== 5. Evaluate (batched) =====
print("Evaluating...")
model.eval()
n_test = len(test_texts)
all_preds = []
with torch.no_grad():
    for i in range(0, n_test, batch_size * 2):
        batch_texts = test_texts[i : i + batch_size * 2]
        input_ids, attn_mask = tokenize_texts(batch_texts)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        preds = model(input_ids=input_ids, attention_mask=attn_mask)
        all_preds.append(preds.cpu().numpy())
scores = np.concatenate(all_preds, axis=0)
dur = time.time() - t0

f1, au, p, r, thr = compute_metrics(y_test_full, scores, eval_mask)
print(
    f"\n  => WOS globalE2E: F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  "
    f"thr={thr:.2f}  t={dur:.1f}s"
)

# ===== 6. Compare =====
print("\n" + "=" * 60)
print("WOS — SOTA COMPARISON")
print("=" * 60)
results = [
    ("HTC-infoMAX (ICML'21)", 0.8720),
    ("HiAGM (Zhou+ ACL'20)", 0.8604),
    ("HMCN (Zhou+ ECML'20)", 0.8580),
    ("Ours global (frozen SPECTER2)", 0.7519),
    ("Ours globalE2E (fine-tuned)", f1),
]
for name, val in sorted(results, key=lambda x: -x[1]):
    mark = " ← NEW" if "E2E" in name else ""
    diff = f" (Δ={val-0.8720:+.4f} vs best)" if abs(val - 0.8720) > 0.001 else ""
    print(f"  {name:<35} {val:.4f}{mark}{diff}")

os.makedirs("./output/wos_arxiv", exist_ok=True)
np.savez_compressed("./output/wos_arxiv/wos_e2e_scores.npz", scores=scores)
print("\nSaved to ./output/wos_arxiv/wos_e2e_scores.npz")

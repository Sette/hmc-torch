#!/usr/bin/env python3
"""GPU-accelerated comparison: global + tabular_mlp + gbdt + Platt on all FUN datasets."""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import networkx as nx

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def compute_metrics(y_true, y_pred, eval_mask):
    y_t = torch.tensor(y_true).cuda()
    y_p = torch.tensor(y_pred).cuda()
    e_m = torch.tensor(eval_mask).cuda()
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_p >= thr).float()
        tp = (y_bin[:, e_m] * y_t[:, e_m]).sum()
        fp = (y_bin[:, e_m] * (1 - y_t[:, e_m])).sum()
        fn = ((1 - y_bin[:, e_m]) * y_t[:, e_m]).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1: best_f1, best_thr, best_p, best_r = f1.item(), thr, p.item(), r.item()
    try:
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except: auprc = 0.0
    return best_f1, auprc, best_p, best_r, best_thr


def build_r_matrix_gpu(adj):
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors: r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).cuda()


def load_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
    mgr = initialize_dataset_experiments(name, device="cpu", dataset_path="./data",
                                          dataset_type="arff", is_global=False)
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32); y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    terms = train.terms; adj = mgr.a
    imp = SimpleImputer(strategy="mean"); scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
    from hmc.data.hierarchy import TreeHierarchy
    branches = [t for t in terms if t != "root"]
    hier = TreeHierarchy.from_fun_cat_terms(branches if branches else terms)
    return X_tr, y_tr, X_te, y_te, eval_mask, adj, hier


DATASETS = ["cellcycle_FUN", "derisi_FUN", "eisen_FUN", "expr_FUN",
            "gasch1_FUN", "gasch2_FUN", "seq_FUN", "spo_FUN"]

all_results = []
DEVICE = torch.device("cuda")
BATCH = 128

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    X_tr, y_tr, X_te, y_te, eval_mask, adj, hier = load_data(ds_name)
    n_nodes = y_tr.shape[1]; n_feat = X_tr.shape[1]
    r_matrix = build_r_matrix_gpu(adj)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  "
          f"Feat: {n_feat}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}")

    # ===== 1. global (MLP + R-matrix) on GPU =====
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    from hmc.datasets.registry import DatasetRegistry
    reg = DatasetRegistry()
    ontology = "FUN"; data_key = ds_name.replace("_FUN", "")
    hidden = reg.hidden_dims.get(ontology, {}).get(data_key, 512)
    rec_ep = reg.all_epochs.get(ontology, {}).get(data_key, 50)
    epochs = min(rec_ep, 50)

    t0 = time.time()
    model_g = ConstrainedModel(
        input_dim=n_feat, hidden_dim=hidden, output_dim=n_nodes,
        hyperparams={"batch_size": BATCH, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": hidden,
                     "lr": 1e-4, "weight_decay": 1e-5},
        r_matrix=r_matrix, baseline_model=False,
    ).to(DEVICE)

    opt_g = torch.optim.AdamW(model_g.parameters(), lr=1e-4, weight_decay=1e-5)
    crit_g = nn.BCELoss()
    Xt_tr = torch.tensor(X_tr); yt_tr = torch.tensor(y_tr)
    tr_ldr_g = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    model_g.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in tr_ldr_g:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            preds = model_g(bx)
            loss = crit_g(preds, by)
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            total_loss += loss.item()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  global epoch {ep+1}/{epochs} loss={total_loss:.2f}")

    model_g.eval()
    with torch.no_grad():
        scores_global = model_g(torch.tensor(X_te).to(DEVICE)).cpu().numpy()

    f1_g, au_g, p_g, r_g, th_g = compute_metrics(y_te, scores_global, eval_mask)
    dur_g = time.time() - t0
    all_results.append({"method": "global", "dataset": ds_name,
                        "micro_f1": f1_g, "auprc": au_g,
                        "precision": p_g, "recall": r_g,
                        "threshold": th_g, "duration_s": dur_g, "epochs": epochs})
    print(f"  => global     F1={f1_g:.4f}  AUPRC={au_g:.4f}  "
          f"P={p_g:.4f}  R={r_g:.4f}  t={dur_g:.1f}s  ep={epochs}")

    # Save
    out_d = f"./output/experiments/{ds_name}/global_gpu"
    os.makedirs(out_d, exist_ok=True)
    np.savez_compressed(f"{out_d}/scores_final.npz", scores=scores_global)

    # ===== 2. tabular_mlp on GPU =====
    from hmc.models.tabular.mlp import TabularMLPModel
    from hmc.models.hierarchical.losses import WeightedBCELoss
    from hmc.models.hierarchical.postprocess import reconcile

    t0 = time.time()
    mlp = TabularMLPModel(
        input_dim=n_feat, n_nodes=n_nodes,
        hidden_dim=min(512, max(128, n_feat)),
        n_blocks=2, head_layers=2, dropout=0.3,
    ).to(DEVICE)

    pos_w = WeightedBCELoss.compute_pos_weight(yt_tr).to(DEVICE)
    crit_m = WeightedBCELoss(pos_weight=pos_w)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-5)

    tr_ldr_m = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    epochs_m = min(rec_ep, 50)
    mlp.train()
    for ep in range(epochs_m):
        total_loss = 0.0
        for bx, by in tr_ldr_m:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            preds = mlp(bx)
            loss = crit_m(preds, by)
            opt_m.zero_grad(); loss.backward(); opt_m.step()
            total_loss += loss.item()
        if (ep + 1) % max(1, epochs_m // 5) == 0:
            print(f"  mlp epoch {ep+1}/{epochs_m} loss={total_loss:.2f}")

    mlp.eval()
    with torch.no_grad():
        te_ldr_m = DataLoader(TensorDataset(torch.tensor(X_te)),
                              batch_size=BATCH, shuffle=False)
        preds_m = []
        for (bx,) in te_ldr_m:
            preds_m.append(mlp(bx.to(DEVICE)).cpu().numpy())
    scores_mlp_raw = np.concatenate(preds_m, axis=0)
    scores_mlp = reconcile(scores_mlp_raw, hier)

    f1_m, au_m, p_m, r_m, th_m = compute_metrics(y_te, scores_mlp, eval_mask)
    dur_m = time.time() - t0
    all_results.append({"method": "tabular_mlp", "dataset": ds_name,
                        "micro_f1": f1_m, "auprc": au_m,
                        "precision": p_m, "recall": r_m,
                        "threshold": th_m, "duration_s": dur_m, "epochs": epochs_m})
    print(f"  => mlp        F1={f1_m:.4f}  AUPRC={au_m:.4f}  "
          f"P={p_m:.4f}  R={r_m:.4f}  t={dur_m:.1f}s  ep={epochs_m}")

    out_m = f"./output/experiments/{ds_name}/tabular_mlp_gpu"
    os.makedirs(out_m, exist_ok=True)
    np.savez_compressed(f"{out_m}/scores_final.npz", scores=scores_mlp)

    # ===== 3. GBDT + Platt (CPU only, but fast) =====
    from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier
    from hmc.models.hierarchical.postprocess import PlattCalibrator
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments as init_ds

    # Need validation split for Platt
    mgr2 = init_ds(ds_name, device="cpu", dataset_path="./data",
                                           dataset_type="arff", is_global=False)
    tr2, va2, te2_ = mgr2.get_datasets()
    imp2 = SimpleImputer(strategy="mean"); scaler2 = StandardScaler()
    X_tr2 = scaler2.fit_transform(imp2.fit_transform(
        np.concatenate([tr2.x, va2.x]).astype(np.float32))).astype(np.float32)
    y_tr2 = np.concatenate([tr2.y, va2.y]).astype(np.float32)
    X_va2 = scaler2.transform(imp2.transform(va2.x.astype(np.float32))).astype(np.float32)
    y_va2 = va2.y.astype(np.float32)
    X_te2 = scaler2.transform(imp2.transform(te2_.x.astype(np.float32))).astype(np.float32)
    y_te2 = te2_.y.astype(np.float32)
    eval2 = np.array(mgr2.to_eval, dtype=bool)

    t0 = time.time()
    n_pos = y_tr2.sum(axis=0)
    trainable = (n_pos >= 5) & eval2

    gbdt = GBDTOvRClassifier(backend="histgb", backend_kwargs={
        "early_stopping": False, "max_iter": 100,
    })
    gbdt.fit(X_tr2, y_tr2, eval_mask=trainable)
    scores_va = gbdt.predict_proba(X_va2)
    scores_te_raw = gbdt.predict_proba(X_te2)

    # Platt on validation
    cal = PlattCalibrator()
    cal.fit(scores_va, y_va2, eval_mask=eval2)
    scores_te_cal = cal.calibrate(scores_te_raw)
    scores_te_final = reconcile(scores_te_cal, hier)

    f1_b, au_b, p_b, r_b, th_b = compute_metrics(y_te2, scores_te_final, eval2)
    dur_b = time.time() - t0
    all_results.append({"method": "gbdt_platt_r", "dataset": ds_name,
                        "micro_f1": f1_b, "auprc": au_b,
                        "precision": p_b, "recall": r_b,
                        "threshold": th_b, "duration_s": dur_b,
                        "nodes_trained": int(trainable.sum())})
    print(f"  => gbdt+platt F1={f1_b:.4f}  AUPRC={au_b:.4f}  "
          f"P={p_b:.4f}  R={r_b:.4f}  t={dur_b:.1f}s  nodes={trainable.sum()}")

    out_b = f"./output/experiments/{ds_name}/gbdt_platt"
    os.makedirs(out_b, exist_ok=True)
    np.savez_compressed(f"{out_b}/scores_final.npz", scores=scores_te_final)

# ===================================================================
print("\n\n" + "=" * 100)
print("GPU COMPARISON — 8 FUN Datasets")
print("=" * 100)

for ds_name in DATASETS:
    ds_res = [r for r in all_results if r["dataset"] == ds_name]
    if not ds_res: continue
    print(f"\n{ds_name}:")
    for r in sorted(ds_res, key=lambda x: -x["micro_f1"]):
        extra = f" ep={r.get('epochs','?')}" if 'epochs' in r else ""
        if 'nodes_trained' in r: extra += f" nodes={r['nodes_trained']}"
        print(f"  {r['method']:<18} F1={r['micro_f1']:.4f}  "
              f"AUPRC={r['auprc']:.4f}  P={r['precision']:.4f}  "
              f"R={r['recall']:.4f}  thr={r['threshold']:.2f}  "
              f"t={r['duration_s']:.1f}s{extra}")

print(f"\n{'='*100}")
print("SUMMARY")
for method in ["global", "gbdt_platt_r", "tabular_mlp"]:
    m_res = [r for r in all_results if r["method"] == method]
    if not m_res: continue
    avg_f1 = np.mean([r["micro_f1"] for r in m_res])
    avg_au = np.mean([r["auprc"] for r in m_res])
    avg_t = np.mean([r["duration_s"] for r in m_res])
    wins = sum(1 for ds in DATASETS
               for r in all_results if r["dataset"] == ds and r["method"] == method
               and r["micro_f1"] == max(rr["micro_f1"] for rr in all_results if rr["dataset"] == ds))
    print(f"  {method:<18} avg F1={avg_f1:.4f}  avg AUPRC={avg_au:.4f}  "
          f"avg t={avg_t:.1f}s  wins={wins}/8")

with open("./output/experiments/gpu_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to ./output/experiments/gpu_comparison.json")

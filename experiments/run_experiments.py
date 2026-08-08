#!/usr/bin/env python3
"""Full HMC experiment runner.

Runs multiple methods on WOS (text) and synthetic seq_FUN (tabular),
saving all artefacts and producing a comparison table.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, "src")
sys.path.insert(0, "tests")


# ===================================================================
# Metrics
# ===================================================================


def compute_all_metrics(y_true, y_pred, eval_mask, hierarchy=None):
    """Compute Micro-F1, AUPRC, per-level F1, and best threshold."""
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.2, 0.85, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, thr, p, r

    y_bin = (y_pred >= best_thr).astype(np.float32)
    tp_g = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
    fp_g = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
    fn_g = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()

    try:
        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except Exception:
        auprc = 0.0

    return {
        "micro_f1": float(best_f1),
        "micro_precision": float(best_p),
        "micro_recall": float(best_r),
        "auprc_micro": auprc,
        "best_threshold": float(best_thr),
        "tp": int(tp_g),
        "fp": int(fp_g),
        "fn": int(fn_g),
    }


# ===================================================================
# WOS experiments (text + transformer)
# ===================================================================


def run_wos_experiments():
    """Run global, globalE2E, and local on WOS."""
    from hmc.datasets.dataset_manager import initialize_dataset_experiments
    from hmc.datasets.registry import DatasetRegistry

    results = []
    output_base = "./output/experiments"
    os.makedirs(output_base, exist_ok=True)

    registry = DatasetRegistry()
    defaults = registry.wos_defaults
    device = torch.device("cpu")

    print("=" * 60)
    print("Loading WOS dataset (SPECTER2 embeddings)...")
    t_load = time.time()

    mgr = initialize_dataset_experiments(
        "wos",
        device="cpu",
        dataset_path="./data",
        is_global=True,
        model_cache_dir="./models",
    )
    train, valid, test = mgr.get_datasets()
    n_nodes = mgr.output_dim
    n_features = mgr.input_dim
    eval_mask = np.array(mgr.to_eval, dtype=bool)

    X_train_raw = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_train_raw = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_test = test.x.astype(np.float32)
    y_test = test.y.astype(np.float32)

    # Normalize embeddings
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test_s = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} samples, {n_features} features")
    print(f"  Test:  {X_test.shape[0]} samples, {n_nodes} nodes")
    print(f"  Load time: {time.time() - t_load:.1f}s")

    # ---- 1. global (frozen SPECTER2 + MLP + R-matrix) ----
    print("\n=== [1/2] global (frozen SPECTER2 + MLP) ===")
    t0 = time.time()
    try:
        # Build R-matrix
        import networkx as nx
        from torch.utils.data import DataLoader

        from hmc.models.global_classifier.constraint.model import ConstrainedModel

        a = mgr.a
        r_matrix_np = np.zeros(a.shape)
        np.fill_diagonal(r_matrix_np, 1)
        g_nx = nx.DiGraph(a)
        for i in range(len(a)):
            ancestors = list(nx.descendants(g_nx, i))
            if ancestors:
                r_matrix_np[i, ancestors] = 1
        r_matrix = torch.tensor(r_matrix_np).transpose(1, 0).unsqueeze(0)

        model = ConstrainedModel(
            input_dim=n_features,
            hidden_dim=defaults["hidden_dim"],
            output_dim=n_nodes,
            hyperparams={
                "batch_size": defaults["batch_size"],
                "num_layers": defaults["num_layers"],
                "dropout": defaults["dropout"],
                "non_lin": "relu",
                "hidden_dim": defaults["hidden_dim"],
                "lr": defaults["lr"],
                "weight_decay": defaults["weight_decay"],
            },
            r_matrix=r_matrix,
            baseline_model=False,
        )

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=defaults["lr"],
            weight_decay=defaults["weight_decay"],
        )
        criterion = torch.nn.BCELoss()

        train_ds = list(zip(torch.tensor(X_train), torch.tensor(y_train_raw)))
        train_loader = DataLoader(
            train_ds, batch_size=defaults["batch_size"], shuffle=True
        )

        model.train()
        epochs = 20
        for ep in range(epochs):
            total_loss = 0.0
            for bx, by in train_loader:
                preds = model(bx)
                loss = criterion(preds, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (ep + 1) % 5 == 0:
                print(f"  epoch {ep+1}/{epochs} loss={total_loss:.2f}")

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test_s)).numpy()

        metrics = compute_all_metrics(y_test, y_pred, eval_mask)
        dur = time.time() - t0

        out_dir = f"{output_base}/wos/global"
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(f"{out_dir}/scores_final.npz", scores=y_pred)
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        results.append(
            {
                "method": "global",
                "dataset": "wos",
                **metrics,
                "duration_s": dur,
                "epochs": epochs,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": n_features,
                "n_nodes": n_nodes,
            }
        )
        print(
            f"  F1={metrics['micro_f1']:.4f}  AUPRC={metrics['auprc_micro']:.4f}  time={dur:.1f}s"
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        results.append(
            {
                "method": "global",
                "dataset": "wos",
                "error": str(e),
                "duration_s": time.time() - t0,
            }
        )

    # ---- 2. local (frozen SPECTER2 + per-level MLPs) ----
    print("\n=== [2/2] local (frozen SPECTER2 + per-level MLPs) ===")
    t0 = time.time()
    try:
        # Load with is_global=False for local classifier
        mgr_loc = initialize_dataset_experiments(
            "wos",
            device="cpu",
            dataset_path="./data",
            is_global=False,
            model_cache_dir="./models",
        )
        tr_l, va_l, te_l = mgr_loc.get_datasets()

        # Use already-scaled features
        X_train_l = scaler.transform(
            np.concatenate([tr_l.x, va_l.x]).astype(np.float32)
        )
        y_train_l = np.concatenate([tr_l.y, va_l.y]).astype(np.float32)
        X_test_l = scaler.transform(te_l.x.astype(np.float32))
        y_test_l = te_l.y.astype(np.float32)

        from torch.utils.data import DataLoader

        from hmc.models.local_classifier.model import LocalModel

        model_l = LocalModel(
            input_dim=n_features,
            levels_size=mgr_loc.levels_size,
            hidden_dim=defaults["hidden_dim"],
            num_layers=defaults["num_layers"],
            dropout=defaults["dropout"],
        )

        opt_l = torch.optim.AdamW(
            model_l.parameters(),
            lr=defaults["lr"],
            weight_decay=defaults["weight_decay"],
        )
        criterion_l = torch.nn.BCELoss()

        tr_dataset = list(
            zip(
                torch.tensor(X_train_l),
                torch.tensor(y_train_l),
                [[np.stack(yy) for yy in yl] for yl in tr_l.y_local]
                + [[np.stack(yy) for yy in yl] for yl in va_l.y_local],
            )
        )
        te_dataset = list(
            zip(
                torch.tensor(X_test_l),
                torch.tensor(y_test_l),
                [[np.stack(yy) for yy in yl] for yl in te_l.y_local],
            )
        )
        tr_loader = DataLoader(
            tr_dataset, batch_size=defaults["batch_size"], shuffle=True
        )
        te_loader = DataLoader(
            te_dataset, batch_size=defaults["batch_size"], shuffle=False
        )

        model_l.train()
        epochs = 20
        for ep in range(epochs):
            total_loss = 0.0
            for bx, _, byl in tr_loader:
                preds = model_l(bx)
                loss = torch.tensor(0.0)
                for lvl in sorted(mgr_loc.levels_size):
                    loss = loss + criterion_l(preds[str(lvl)], byl[lvl].float())
                opt_l.zero_grad()
                loss.backward()
                opt_l.step()
                total_loss += loss.item()
            if (ep + 1) % 5 == 0:
                print(f"  epoch {ep+1}/{epochs} loss={total_loss:.2f}")

        # Evaluate
        model_l.eval()
        all_preds = []
        with torch.no_grad():
            for bx, yg, _ in te_loader:
                preds = model_l(bx)
                for i in range(len(bx)):
                    gp = np.zeros(n_nodes, dtype=np.float32)
                    for lvl in sorted(mgr_loc.levels_size):
                        lp = preds[str(lvl)][i].numpy()
                        for name, li in mgr_loc.local_nodes_idx[lvl].items():
                            gp[mgr_loc.nodes_idx[name]] = lp[li]
                    all_preds.append(gp)
        y_pred_l = np.stack(all_preds)

        metrics_l = compute_all_metrics(
            y_test_l, y_pred_l, np.array(mgr_loc.to_eval, dtype=bool)
        )
        dur_l = time.time() - t0

        out_dir_l = f"{output_base}/wos/local"
        os.makedirs(out_dir_l, exist_ok=True)
        np.savez_compressed(f"{out_dir_l}/scores_final.npz", scores=y_pred_l)
        with open(f"{out_dir_l}/metrics.json", "w") as f:
            json.dump(metrics_l, f, indent=2)

        results.append(
            {
                "method": "local",
                "dataset": "wos",
                **metrics_l,
                "duration_s": dur_l,
                "epochs": epochs,
                "n_train": len(X_train_l),
                "n_test": len(X_test_l),
                "n_features": n_features,
                "n_nodes": n_nodes,
            }
        )
        print(
            f"  F1={metrics_l['micro_f1']:.4f}  AUPRC={metrics_l['auprc_micro']:.4f}  time={dur_l:.1f}s"
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        results.append(
            {
                "method": "local",
                "dataset": "wos",
                "error": str(e),
                "duration_s": time.time() - t0,
            }
        )

    return results


# ===================================================================
# seq_FUN synthetic experiments
# ===================================================================


def run_seqfun_experiments():
    """Run all tabular methods on synthetic seq_FUN fixture."""
    sys.path.insert(0, "tests")
    from tests.fixtures.arff import SyntheticARFFFixture
    from tests.test_gofun_loading import _patch_gofun_paths, _restore_gofun_paths

    results = []
    output_base = "./output/experiments"

    fixture = SyntheticARFFFixture(
        name="seq_FUN",
        hierarchy="root.CC.CC01,root.CC.CC02,root.MF.MF01,root.BP.BP01",
        num_features=50,
        num_train=400,
        num_valid=50,
        num_test=50,
    )
    tmpdir = fixture.setup()
    _patch_gofun_paths(tmpdir)

    try:
        from hmc.data.hierarchy import TreeHierarchy
        from hmc.datasets.dataset_manager import initialize_dataset_experiments
        from hmc.models.hierarchical.postprocess import reconcile
        from hmc.models.tabular.preprocessing import TabularPreprocessor

        # ==== local ====
        print("\n=== [seq_FUN] local ===")
        t0 = time.time()
        from torch.utils.data import DataLoader

        from hmc.models.local_classifier.model import LocalModel

        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False
        )
        tr, va, te = mgr.get_datasets()

        pp_loc = TabularPreprocessor(with_imputation=True, with_scaling=True)
        Xtr = pp_loc.fit_transform(
            np.concatenate([tr.x, va.x]).astype(np.float32),
            np.concatenate([tr.y, va.y]).astype(np.float32),
        )
        ytr = np.concatenate([tr.y, va.y]).astype(np.float32)
        Xte = pp_loc.transform(te.x.astype(np.float32))
        yte = te.y.astype(np.float32)

        model_loc = LocalModel(
            input_dim=Xtr.shape[1],
            levels_size=mgr.levels_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
        )
        opt_loc = torch.optim.AdamW(model_loc.parameters(), lr=1e-3)
        crit_loc = torch.nn.BCELoss()

        tr_ds = list(
            zip(
                torch.tensor(Xtr),
                torch.tensor(ytr),
                [[np.stack(yy) for yy in yl] for yl in tr.y_local]
                + [[np.stack(yy) for yy in yl] for yl in va.y_local],
            )
        )
        te_ds = list(
            zip(
                torch.tensor(Xte),
                torch.tensor(yte),
                [[np.stack(yy) for yy in yl] for yl in te.y_local],
            )
        )
        tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ldr = DataLoader(te_ds, batch_size=32, shuffle=False)

        model_loc.train()
        for ep in range(30):
            for bx, _, byl in tr_ldr:
                preds = model_loc(bx)
                loss = torch.tensor(0.0)
                for lvl in sorted(mgr.levels_size):
                    loss = loss + crit_loc(preds[str(lvl)], byl[lvl].float())
                opt_loc.zero_grad()
                loss.backward()
                opt_loc.step()

        model_loc.eval()
        all_preds = []
        eval_mask_loc = np.array(mgr.to_eval, dtype=bool)
        with torch.no_grad():
            for bx, yg, _ in te_ldr:
                preds = model_loc(bx)
                for i in range(len(bx)):
                    gp = np.zeros(mgr.output_dim, dtype=np.float32)
                    for lvl in sorted(mgr.levels_size):
                        lp = preds[str(lvl)][i].numpy()
                        for name, li in mgr.local_nodes_idx[lvl].items():
                            gp[mgr.nodes_idx[name]] = lp[li]
                    all_preds.append(gp)
        yp_loc = np.stack(all_preds)

        hier = TreeHierarchy.from_fun_cat_terms([t for t in tr.terms if t != "root"])
        yp_loc_rec = reconcile(yp_loc, hier)

        metrics_loc = compute_all_metrics(yte, yp_loc_rec, eval_mask_loc)
        results.append(
            {
                "method": "local",
                "dataset": "seq_FUN",
                **metrics_loc,
                "duration_s": time.time() - t0,
                "epochs": 30,
                "n_train": len(Xtr),
                "n_test": len(Xte),
            }
        )
        print(
            f"  F1={metrics_loc['micro_f1']:.4f}  AUPRC={metrics_loc['auprc_micro']:.4f}  time={time.time()-t0:.1f}s"
        )

        # ==== tabular_gbdt ====
        print("\n=== [seq_FUN] tabular_gbdt ===")
        t0 = time.time()
        from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier

        mgr2 = initialize_dataset_experiments(
            "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False
        )
        tr2, va2, te2 = mgr2.get_datasets()

        pp_gbdt = TabularPreprocessor(with_imputation=True, with_scaling=True)
        Xgb = pp_gbdt.fit_transform(
            np.concatenate([tr2.x, va2.x]).astype(np.float32),
            np.concatenate([tr2.y, va2.y]).astype(np.float32),
        )
        ygb = np.concatenate([tr2.y, va2.y]).astype(np.float32)
        Xgb_te = pp_gbdt.transform(te2.x.astype(np.float32))
        ygb_te = te2.y.astype(np.float32)

        gbdt = GBDTOvRClassifier(backend="histgb")
        gbdt.fit(Xgb, ygb, eval_mask=np.array(mgr2.to_eval, dtype=bool))
        scores_gbdt = gbdt.predict_proba(Xgb_te)

        hier_gbdt = TreeHierarchy.from_fun_cat_terms(
            [t for t in tr2.terms if t != "root"]
        )
        scores_gbdt_rec = reconcile(scores_gbdt, hier_gbdt)

        metrics_gbdt = compute_all_metrics(
            ygb_te, scores_gbdt_rec, np.array(mgr2.to_eval, dtype=bool)
        )
        results.append(
            {
                "method": "tabular_gbdt",
                "dataset": "seq_FUN",
                **metrics_gbdt,
                "duration_s": time.time() - t0,
            }
        )
        print(
            f"  F1={metrics_gbdt['micro_f1']:.4f}  AUPRC={metrics_gbdt['auprc_micro']:.4f}  time={time.time()-t0:.1f}s"
        )

        # ==== tabular_mlp ====
        print("\n=== [seq_FUN] tabular_mlp ===")
        t0 = time.time()
        from torch.utils.data import TensorDataset

        from hmc.models.hierarchical.losses import WeightedBCELoss
        from hmc.models.tabular.mlp import TabularMLPModel

        mgr3 = initialize_dataset_experiments(
            "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False
        )
        tr3, va3, te3 = mgr3.get_datasets()

        pp_mlp = TabularPreprocessor(with_imputation=True, with_scaling=True)
        Xm = pp_mlp.fit_transform(
            np.concatenate([tr3.x, va3.x]).astype(np.float32),
            np.concatenate([tr3.y, va3.y]).astype(np.float32),
        )
        ym = np.concatenate([tr3.y, va3.y]).astype(np.float32)
        Xm_te = pp_mlp.transform(te3.x.astype(np.float32))
        ym_te = te3.y.astype(np.float32)

        device = torch.device("cpu")
        mlp_model = TabularMLPModel(
            input_dim=Xm.shape[1],
            n_nodes=ym.shape[1],
            hidden_dim=128,
            n_blocks=2,
            head_layers=2,
            dropout=0.2,
        ).to(device)

        pos_w = WeightedBCELoss.compute_pos_weight(torch.tensor(ym)).to(device)
        crit_mlp = WeightedBCELoss(pos_weight=pos_w)
        opt_mlp = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)

        tr_ds_mlp = TensorDataset(torch.tensor(Xm), torch.tensor(ym))
        te_ds_mlp = TensorDataset(torch.tensor(Xm_te), torch.tensor(ym_te))
        tr_ldr_mlp = DataLoader(tr_ds_mlp, batch_size=32, shuffle=True)
        te_ldr_mlp = DataLoader(te_ds_mlp, batch_size=32, shuffle=False)

        mlp_model.train()
        for ep in range(40):
            for bx, by in tr_ldr_mlp:
                bx, by = bx.to(device), by.to(device)
                preds = mlp_model(bx)
                loss = crit_mlp(preds, by)
                opt_mlp.zero_grad()
                loss.backward()
                opt_mlp.step()

        mlp_model.eval()
        all_preds_mlp = []
        with torch.no_grad():
            for bx, _ in te_ldr_mlp:
                all_preds_mlp.append(mlp_model(bx).cpu().numpy())
        yp_mlp = np.concatenate(all_preds_mlp, axis=0)

        hier_mlp = TreeHierarchy.from_fun_cat_terms(
            [t for t in tr3.terms if t != "root"]
        )
        yp_mlp_rec = reconcile(yp_mlp, hier_mlp)

        metrics_mlp = compute_all_metrics(
            ym_te, yp_mlp_rec, np.array(mgr3.to_eval, dtype=bool)
        )
        results.append(
            {
                "method": "tabular_mlp",
                "dataset": "seq_FUN",
                **metrics_mlp,
                "duration_s": time.time() - t0,
                "epochs": 40,
                "n_train": len(Xm),
                "n_test": len(Xm_te),
            }
        )
        print(
            f"  F1={metrics_mlp['micro_f1']:.4f}  AUPRC={metrics_mlp['auprc_micro']:.4f}  time={time.time()-t0:.1f}s"
        )

    finally:
        _restore_gofun_paths()
        fixture.teardown()

    return results


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    all_results = []

    # WOS text experiments
    print("\n" + "=" * 60)
    print("PHASE 1: WOS (text + SPECTER2 transformer)")
    print("=" * 60)
    all_results.extend(run_wos_experiments())

    # seq_FUN tabular experiments
    print("\n" + "=" * 60)
    print("PHASE 2: seq_FUN (tabular synthetic)")
    print("=" * 60)
    all_results.extend(run_seqfun_experiments())

    # Save and report
    os.makedirs("./output/experiments", exist_ok=True)
    with open("./output/experiments/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n" + "=" * 85)
    print("FINAL COMPARISON TABLE")
    print("=" * 85)
    header = f"{'Method':<18} {'Dataset':<10} {'Micro-F1':>10} {'AUPRC':>10} {'Prec':>8} {'Rec':>8} {'Thr':>6} {'Time':>8} {'Train':>7}"
    print(header)
    print("-" * 85)

    for r in sorted(all_results, key=lambda x: -(x.get("micro_f1") or 0)):
        if "error" in r:
            print(f"{r['method']:<18} {r['dataset']:<10} {'ERR: '+r['error'][:45]}")
        else:
            print(
                f"{r['method']:<18} {r['dataset']:<10} {r['micro_f1']:>10.4f} {r['auprc_micro']:>10.4f} {r['micro_precision']:>8.4f} {r['micro_recall']:>8.4f} {r['best_threshold']:>6.2f} {r['duration_s']:>7.1f}s {r.get('n_train', 0):>7d}"
            )

    print("-" * 85)
    print("\nArtefacts saved to ./output/experiments/")

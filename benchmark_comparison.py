"""Comparative benchmark of HMC methods.

Runs all applicable methods on synthetic seq_FUN (tabular) and real WOS
(text) data, producing a comparison table with Micro-F1, AUPRC, and
training time.
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import numpy as np

from tests.fixtures.arff import SyntheticARFFFixture
from tests.test_gofun_loading import _patch_gofun_paths, _restore_gofun_paths


@dataclass
class Result:
    method: str
    dataset: str
    micro_f1: float
    auprc: float
    precision: float
    recall: float
    threshold: float
    duration_s: float
    n_features: int = 0
    n_samples_train: int = 0
    n_samples_test: int = 0
    error: Optional[str] = None


def _compute_metrics(y_true, y_pred, eval_mask):
    thresholds = np.arange(0.2, 0.85, 0.05)
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in thresholds:
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, thr, p, r
    try:
        from sklearn.metrics import average_precision_score
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except ImportError:
        auprc = 0.0
    return best_f1, auprc, best_p, best_r, best_thr


# ===================================================================
# 1. Tabular methods on synthetic seq_FUN
# ===================================================================

def bench_tabular_methods():
    results = []
    fixture = SyntheticARFFFixture(
        name="seq_FUN",
        hierarchy="root.CC.CC01,root.CC.CC02,root.MF.MF01,root.BP.BP01",
        num_features=50, num_train=200, num_valid=40, num_test=40,
        is_go=False,
    )
    tmpdir = fixture.setup()
    _patch_gofun_paths(tmpdir)

    try:
        # ---- local (frozen MLP per level) ----
        print("\n=== local (seq_FUN) ===")
        t0 = time.time()
        try:
            import torch
            from hmc.datasets.dataset_manager import initialize_dataset_experiments
            from hmc.models.local_classifier.model import LocalModel
            from torch.utils.data import DataLoader

            mgr = initialize_dataset_experiments(
                "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False)
            train, valid, test = mgr.get_datasets()

            # Combine train+valid
            X_train = np.concatenate([train.x, valid.x]).astype(np.float32)
            y_train = np.concatenate([train.y, valid.y]).astype(np.float32)
            X_test = test.x.astype(np.float32)
            y_test = test.y.astype(np.float32)

            # Simple scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            eval_mask = np.array(mgr.to_eval, dtype=bool)

            # Model
            model = LocalModel(
                input_dim=X_train.shape[1],
                levels_size=mgr.levels_size,
                hidden_dim=128, num_layers=2, dropout=0.2,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = torch.nn.BCELoss()

            # DataLoaders
            train_ds = list(zip(
                torch.tensor(X_train), torch.tensor(y_train),
                [[np.stack(yy) for yy in yl] for yl in train.y_local] +
                [[np.stack(yy) for yy in yl] for yl in valid.y_local]
            ))
            test_ds = list(zip(
                torch.tensor(X_test), torch.tensor(y_test),
                [[np.stack(yy) for yy in yl] for yl in test.y_local]
            ))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            # Train
            model.train()
            for epoch in range(20):
                total_loss = 0.0
                for batch in train_loader:
                    x, _, y_local = batch
                    preds = model(x)
                    loss = torch.tensor(0.0)
                    for lvl in sorted(mgr.levels_size):
                        y_true = y_local[lvl].float()
                        loss = loss + criterion(preds[str(lvl)], y_true)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Evaluate
            model.eval()
            all_preds = []
            with torch.no_grad():
                for x, yg, _yl in test_loader:
                    preds = model(x)
                    for i in range(len(x)):
                        gp = np.zeros(mgr.output_dim, dtype=np.float32)
                        for lvl in sorted(mgr.levels_size):
                            lp = preds[str(lvl)][i].numpy()
                            lidx = mgr.local_nodes_idx[lvl]
                            for name, li in lidx.items():
                                gp[mgr.nodes_idx[name]] = lp[li]
                        all_preds.append(gp)
            y_pred = np.stack(all_preds)
            f1, auprc, p, r, thr = _compute_metrics(y_test, y_pred, eval_mask)

            results.append(Result(
                method="local", dataset="seq_FUN",
                micro_f1=float(f1), auprc=float(auprc),
                precision=float(p), recall=float(r), threshold=float(thr),
                duration_s=time.time() - t0,
                n_features=X_train.shape[1],
                n_samples_train=len(X_train), n_samples_test=len(X_test),
            ))
            print(f"  F1={f1:.4f}  AUPRC={auprc:.4f}  time={time.time()-t0:.1f}s")
        except Exception as e:
            results.append(Result(method="local", dataset="seq_FUN",
                                  micro_f1=0, auprc=0, precision=0, recall=0,
                                  threshold=0.5, duration_s=time.time()-t0,
                                  error=str(e)))
            print(f"  ERROR: {e}")

        # ---- tabular_gbdt ----
        print("\n=== tabular_gbdt (seq_FUN) ===")
        t0 = time.time()
        try:
            from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier
            from hmc.models.tabular.preprocessing import TabularPreprocessor
            from hmc.data.hierarchy import TreeHierarchy
            from hmc.models.hierarchical.postprocess import reconcile

            mgr2 = initialize_dataset_experiments(
                "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False)
            train2, valid2, test2 = mgr2.get_datasets()
            Xt = np.concatenate([train2.x, valid2.x]).astype(np.float32)
            yt = np.concatenate([train2.y, valid2.y]).astype(np.float32)
            Xt2 = test2.x.astype(np.float32)
            yt2 = test2.y.astype(np.float32)

            pp = TabularPreprocessor(with_imputation=True, with_scaling=True)
            Xt_pp = pp.fit_transform(Xt, yt)
            Xt2_pp = pp.transform(Xt2)

            gbdt = GBDTOvRClassifier(backend="histgb")
            gbdt.fit(Xt_pp, yt, eval_mask=np.array(mgr2.to_eval, dtype=bool))
            scores_raw = gbdt.predict_proba(Xt2_pp)

            branches = [t for t in train2.terms if t != "root"]
            hier = TreeHierarchy.from_fun_cat_terms(branches)
            scores_final = reconcile(scores_raw, hier)

            f1, auprc, p, r, thr = _compute_metrics(yt2, scores_final,
                                                     np.array(mgr2.to_eval, dtype=bool))
            results.append(Result(
                method="tabular_gbdt", dataset="seq_FUN",
                micro_f1=float(f1), auprc=float(auprc),
                precision=float(p), recall=float(r), threshold=float(thr),
                duration_s=time.time() - t0,
                n_features=Xt_pp.shape[1],
                n_samples_train=len(Xt), n_samples_test=len(Xt2),
            ))
            print(f"  F1={f1:.4f}  AUPRC={auprc:.4f}  time={time.time()-t0:.1f}s")
        except Exception as e:
            results.append(Result(method="tabular_gbdt", dataset="seq_FUN",
                                  micro_f1=0, auprc=0, precision=0, recall=0,
                                  threshold=0.5, duration_s=time.time()-t0,
                                  error=str(e)))
            print(f"  ERROR: {e}")

        # ---- tabular_mlp ----
        print("\n=== tabular_mlp (seq_FUN) ===")
        t0 = time.time()
        try:
            import torch
            from hmc.models.tabular.mlp import TabularMLPModel
            from hmc.models.hierarchical.losses import WeightedBCELoss
            from torch.utils.data import DataLoader, TensorDataset

            mgr3 = initialize_dataset_experiments(
                "seq_FUN", device="cpu", dataset_path=tmpdir, is_global=False)
            train3, valid3, test3 = mgr3.get_datasets()
            Xm = np.concatenate([train3.x, valid3.x]).astype(np.float32)
            ym = np.concatenate([train3.y, valid3.y]).astype(np.float32)
            Xm2 = test3.x.astype(np.float32)
            ym2 = test3.y.astype(np.float32)

            pp2 = TabularPreprocessor(with_imputation=True, with_scaling=True)
            Xm_pp = pp2.fit_transform(Xm, ym)
            Xm2_pp = pp2.transform(Xm2)

            device = torch.device("cpu")
            model_mlp = TabularMLPModel(
                input_dim=Xm_pp.shape[1], n_nodes=ym.shape[1],
                hidden_dim=128, n_blocks=2, head_layers=2, dropout=0.2,
            ).to(device)

            pos_w = WeightedBCELoss.compute_pos_weight(torch.tensor(ym)).to(device)
            criterion_mlp = WeightedBCELoss(pos_weight=pos_w)
            optimizer_mlp = torch.optim.AdamW(model_mlp.parameters(), lr=1e-3)

            train_ds = TensorDataset(torch.tensor(Xm_pp), torch.tensor(ym))
            test_ds = TensorDataset(torch.tensor(Xm2_pp), torch.tensor(ym2))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            model_mlp.train()
            for epoch in range(30):
                total_loss = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = model_mlp(bx)
                    loss = criterion_mlp(preds, by)
                    optimizer_mlp.zero_grad()
                    loss.backward()
                    optimizer_mlp.step()
                    total_loss += loss.item()

            model_mlp.eval()
            all_preds = []
            with torch.no_grad():
                for bx, _ in test_loader:
                    all_preds.append(model_mlp(bx).cpu().numpy())
            y_pred_mlp = np.concatenate(all_preds, axis=0)

            hier2 = TreeHierarchy.from_fun_cat_terms(
                [t for t in train3.terms if t != "root"]
            )
            scores_final_mlp = reconcile(y_pred_mlp, hier2)

            f1, auprc, p, r, thr = _compute_metrics(ym2, scores_final_mlp,
                                                     np.array(mgr3.to_eval, dtype=bool))
            results.append(Result(
                method="tabular_mlp", dataset="seq_FUN",
                micro_f1=float(f1), auprc=float(auprc),
                precision=float(p), recall=float(r), threshold=float(thr),
                duration_s=time.time() - t0,
                n_features=Xm_pp.shape[1],
                n_samples_train=len(Xm), n_samples_test=len(Xm2),
            ))
            print(f"  F1={f1:.4f}  AUPRC={auprc:.4f}  time={time.time()-t0:.1f}s")
        except Exception as e:
            results.append(Result(method="tabular_mlp", dataset="seq_FUN",
                                  micro_f1=0, auprc=0, precision=0, recall=0,
                                  threshold=0.5, duration_s=time.time()-t0,
                                  error=str(e)))
            print(f"  ERROR: {e}")

    finally:
        _restore_gofun_paths()
        fixture.teardown()

    return results


# ===================================================================
# 2. Text-based methods on WOS
# ===================================================================

def bench_wos_methods():
    results = []
    wos_path = "./data"

    # ---- WOS: global (frozen) ----
    print("\n=== global (WOS) ===")
    t0 = time.time()
    try:
        import torch
        from hmc.datasets.dataset_manager import initialize_dataset_experiments
        from hmc.models.global_classifier.constraint.model import ConstrainedModel
        from hmc.pipeline.global_classifier.core.train import train_step
        from hmc.utils.train.job import create_job_id_name
        from torch.utils.data import DataLoader
        import networkx as nx

        mgr = initialize_dataset_experiments(
            "wos", device="cpu", dataset_path=wos_path, is_global=True,
            model_cache_dir="./models",
        )
        train, valid, test = mgr.get_datasets()

        X_train = train.x.astype(np.float32)
        y_train = train.y.astype(np.float32)
        X_test = test.x.astype(np.float32)
        y_test = test.y.astype(np.float32)

        eval_mask = np.array(mgr.to_eval, dtype=bool)
        n_nodes = y_train.shape[1]
        n_features = X_train.shape[1]

        # Build model
        model_wos = ConstrainedModel(
            input_dim=n_features,
            hidden_dim=256,
            output_dim=n_nodes,
            hyperparams={"batch_size": 32, "num_layers": 2, "dropout": 0.3,
                         "non_lin": "relu", "hidden_dim": 256},
            r_matrix=torch.eye(n_nodes).unsqueeze(0),  # simplified - no constraint matrix
            baseline_model=False,
        )

        optimizer = torch.optim.AdamW(model_wos.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss()

        Xt_train = torch.tensor(X_train)
        yt_train = torch.tensor(y_train)
        Xt_test = torch.tensor(X_test)

        train_ds = list(zip(Xt_train, yt_train))
        test_ds = list(zip(Xt_test, yt_test))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model_wos.train()
        for epoch in range(5):
            total_loss = 0.0
            for bx, by in train_loader:
                preds = model_wos(bx)
                loss = criterion(preds, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model_wos.eval()
        with torch.no_grad():
            y_pred = model_wos(Xt_test).numpy()

        f1, auprc, p, r, thr = _compute_metrics(y_test, y_pred, eval_mask)
        results.append(Result(
            method="global", dataset="wos",
            micro_f1=float(f1), auprc=float(auprc),
            precision=float(p), recall=float(r), threshold=float(thr),
            duration_s=time.time() - t0,
            n_features=n_features,
            n_samples_train=len(X_train), n_samples_test=len(X_test),
        ))
        print(f"  F1={f1:.4f}  AUPRC={auprc:.4f}  time={time.time()-t0:.1f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(Result(method="global", dataset="wos",
                              micro_f1=0, auprc=0, precision=0, recall=0,
                              threshold=0.5, duration_s=time.time()-t0,
                              error=str(e)))
        print(f"  ERROR: {e}")

    return results


# ===================================================================
# Run and report
# ===================================================================

if __name__ == "__main__":
    all_results = []

    print("=" * 60)
    print("BENCHMARK: HMC Methods Comparison")
    print("=" * 60)

    # Tabular methods on seq_FUN
    all_results.extend(bench_tabular_methods())

    # Text methods on WOS
    all_results.extend(bench_wos_methods())

    # ---- Report ----
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    # Header
    header = f"{'Method':<18} {'Dataset':<10} {'F1':>8} {'AUPRC':>8} {'Prec':>8} {'Rec':>8} {'Thr':>6} {'Time':>7} {'Train':>7} {'Test':>6} {'Feat':>6}"
    print(header)
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: -x.micro_f1):
        if r.error:
            status = f"ERR: {r.error[:40]}"
        else:
            status = f"{r.micro_f1:>8.4f} {r.auprc:>8.4f} {r.precision:>8.4f} {r.recall:>8.4f} {r.threshold:>6.2f} {r.duration_s:>6.1f}s {r.n_samples_train:>6d} {r.n_samples_test:>6d} {r.n_features:>6d}"
        print(f"{r.method:<18} {r.dataset:<10} {status}")

    print("-" * 90)
    print(f"\nTotal: {len(all_results)} methods evaluated")

    # Save JSON
    out = []
    for r in all_results:
        d = {
            "method": r.method, "dataset": r.dataset,
            "micro_f1": r.micro_f1, "auprc": r.auprc,
            "precision": r.precision, "recall": r.recall,
            "threshold": r.threshold, "duration_s": r.duration_s,
            "n_features": r.n_features,
            "n_train": r.n_samples_train, "n_test": r.n_samples_test,
        }
        if r.error:
            d["error"] = r.error
        out.append(d)

    os.makedirs("./results", exist_ok=True)
    with open("./results/comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to ./results/comparison.json")

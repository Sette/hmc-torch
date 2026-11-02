"""HPT + R-matrix pipeline: ``python -m hmc.main --method globalHPT``.

1. Load a text dataset with its official splits and raw documents.
2. Build :class:`~hmc.models.global_classifier.hpt.HPTPromptModel` over the
   label hierarchy (one ``[PRED]`` slot per level, GAT-encoded label tokens).
3. Train with HPT's recipe (Adam 3e-5, effective batch 16, MLM auxiliary loss,
   early stopping) on ZMLCE plus the hierarchy objective of
   ``--consistency_loss``: ``none`` (HPT), ``mc`` (MCLoss on logits) or
   ``hinge`` (Eq. 2).  The epoch is chosen on the validation micro-F1 of the
   reconciled scores.
4. Score the test split once, raw and reconciled, at HPT's fixed threshold
   (logit 0, i.e. probability 0.5) and at the validation-chosen threshold:
   micro/macro F1, micro AUPRC and Global Hierarchy Violation.

``fit_hpt`` and ``evaluate_hpt`` are shared with ``experiments/run_hpt.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hmc.models.global_classifier.hpt import HPTPromptModel, hpt_objective
from hmc.pipeline.surj.main import _hierarchy
from hmc.pipeline.tabular.main import _save_artefacts
from hmc.utils.metrics.hierarchy_violation import global_hierarchy_violation
from hmc.utils.metrics.multilabel import (
    macro_f1,
    micro_auprc,
    micro_prf,
    pick_threshold,
)

logger = logging.getLogger(__name__)

# Logit given to the root column(s): always on, never trained.
ROOT_LOGIT = 30.0

BACKBONES = {"bert": "bert-base-uncased", "specter2": "allenai/specter2_base"}


@dataclass
class HPTConfig:  # pylint: disable=too-many-instance-attributes
    """Training recipe; the defaults are HPT's (``HPT/train.py``)."""

    backbone: str = "bert-base-uncased"
    loss: str = "none"
    lambda_hier: float = 1.0
    mlm: bool = True
    graph_direction: str = "both"
    graph_layers: int = 1
    lr: float = 3e-5
    batch_size: int = 8  # x grad_accum = HPT's batch of 16 (fits 12 GB in bf16)
    grad_accum: int = 2
    max_epochs: int = 20
    patience: int = 6
    max_length: int = 512
    eval_batch_size: int = 32
    bf16: bool = True
    seed: int = 0
    device: str = "cuda"
    model_cache_dir: str = "./models"
    limit: int | None = None  # smoke tests: first N documents of every split


def label_space(mgr) -> dict:
    """Hierarchy of a text manager in the model's label-index space.

    Label ``k`` is the ``k``-th column of ``to_eval``; its level is its
    shortest-path depth below the root minus one (the manager's own levels).
    """
    h = _hierarchy(mgr)
    m = np.asarray(mgr.to_eval, dtype=bool)
    cols = np.nonzero(m)[0]
    pos = {int(c): k for k, c in enumerate(cols)}
    codes = [None] * len(m)
    for term, idx in mgr.nodes_idx.items():
        codes[idx] = str(term)
    names = getattr(mgr, "label_names", {}) or {}
    child, parent = h["edges"][:, 0], h["edges"][:, 1]
    edges = np.array(
        [
            (pos[int(p)], pos[int(c)])
            for c, p in zip(child, parent)
            if int(p) in pos and int(c) in pos
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    return {
        **h,
        "eval_mask": m,
        "label_cols": cols,
        "label_depth": [int(h["depth"][c]) - 1 for c in cols],
        "label_codes": [codes[c] for c in cols],
        "label_texts": [names.get(codes[c], codes[c]) for c in cols],
        "label_edges": edges,
    }


def _split_data(mgr, limit: int | None):
    splits = mgr.get_datasets()
    texts, ys = [], []
    for s in splits:
        if s.texts is None:
            raise ValueError("dataset manager does not expose raw texts (split.texts)")
        n = len(s.texts) if limit is None else min(limit, len(s.texts))
        texts.append(list(s.texts[:n]))
        ys.append(np.asarray(s.y[:n], dtype=np.float32))
    return texts, ys


def _full_logits(
    z: torch.Tensor, label_cols: torch.Tensor, n_total: int
) -> torch.Tensor:
    full = z.new_full((z.shape[0], n_total), ROOT_LOGIT, dtype=torch.float32)
    full[:, label_cols] = z.float()
    return full


def fit_hpt(  # pylint: disable=too-many-locals,too-many-statements,logging-fstring-interpolation
    mgr,
    cfg: HPTConfig,
    log: Callable[[str], None] = logger.info,
    score_test: bool = True,
) -> dict:
    """Train on the train split with early stopping on validation.

    Returns the best model, the training history and the raw validation and
    test probabilities ``(n, n_columns)`` of the best epoch.  With
    ``score_test=False`` (screening) the test split is never scored.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    space = label_space(mgr)
    texts, ys = _split_data(mgr, cfg.limit)
    n_total = ys[0].shape[1]
    m = space["eval_mask"]

    model = HPTPromptModel.from_pretrained(
        cfg.backbone,
        cfg.model_cache_dir,
        label_texts=space["label_texts"],
        label_codes=space["label_codes"],
        label_depth=space["label_depth"],
        edges=space["label_edges"],
        graph_layers=cfg.graph_layers,
        graph_direction=cfg.graph_direction,
        use_mlm=cfg.mlm,
    ).to(device)
    rows = [model.prompt_rows(t, cfg.max_length) for t in texts]
    label_cols = torch.from_numpy(space["label_cols"]).to(device)
    eval_cols = label_cols
    anc = torch.from_numpy(space["anc"]).to(device)
    desc = torch.from_numpy(space["desc"]).to(device)
    sparse_r = space["sparse_r"]
    amp = cfg.bf16 and device.type == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def predict(split: int) -> np.ndarray:
        model.eval()
        order = np.argsort([len(r) for r in rows[split]])  # length-bucketed batches
        out = np.empty((len(order), n_total), dtype=np.float32)
        for i in range(0, len(order), cfg.eval_batch_size):
            idx = order[i : i + cfg.eval_batch_size]
            batch = model.collate([rows[split][j] for j in idx])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                z, _ = model(**{k: v.to(device) for k, v in batch.items()})
            out[idx] = torch.sigmoid(_full_logits(z, label_cols, n_total)).cpu().numpy()
        return out

    def reconcile(s: np.ndarray) -> np.ndarray:
        return sparse_r.reconcile(torch.from_numpy(s)).numpy()

    y_train = torch.from_numpy(ys[0])
    gen = torch.Generator().manual_seed(cfg.seed)
    history, best_f1, best_epoch, best_state, bad = [], -1.0, 0, None, 0
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(rows[0]), generator=gen).tolist()
        opt.zero_grad()
        n_steps = 0
        for step, i in enumerate(range(0, len(perm), cfg.batch_size)):
            idx = perm[i : i + cfg.batch_size]
            batch = model.collate([rows[0][j] for j in idx])
            yb = y_train[idx].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                z, mlm_loss = model(**{k: v.to(device) for k, v in batch.items()})
            loss = hpt_objective(
                cfg.loss,
                _full_logits(z, label_cols, n_total),
                yb,
                eval_cols,
                sparse_r,
                anc,
                desc,
                cfg.lambda_hier,
            )
            if mlm_loss is not None:
                loss = loss + mlm_loss
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0 or i + cfg.batch_size >= len(perm):
                opt.step()
                opt.zero_grad()
                n_steps += 1
        s_val = reconcile(predict(1))
        thr = pick_threshold(ys[1][:, m], s_val[:, m])
        val_f1 = micro_prf(ys[1][:, m], s_val[:, m], thr)[2]
        record = {
            "epoch": epoch + 1,
            "val_f1": val_f1,
            "val_macro_f1": macro_f1(ys[1][:, m], s_val[:, m], thr),
            "threshold": thr,
            "steps": n_steps,
            "elapsed_s": time.time() - t0,
        }
        history.append(record)
        log(
            f"epoch {epoch + 1}: val micro-F1 {val_f1:.4f} macro-F1 "
            f"{record['val_macro_f1']:.4f} (thr {thr}, {record['elapsed_s']:.0f}s)"
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch + 1, 0
            best_state = {
                k: v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop after {epoch + 1} epochs (patience {cfg.patience})")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    return {
        "model": model,
        "space": space,
        "y": ys,
        "history": history,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "n": [len(t) for t in texts],
        "raw_val": predict(1),
        "raw_test": predict(2) if score_test else None,
    }


def evaluate_hpt(fit: dict) -> dict:
    """Test metrics, raw and reconciled, at threshold 0.5 and the validation one."""
    space = fit["space"]
    m = space["eval_mask"]
    y_val, y_te = fit["y"][1][:, m], fit["y"][2][:, m]
    rec = space["sparse_r"].reconcile
    scores = {
        "raw": (fit["raw_val"], fit["raw_test"]),
        "rec": (
            rec(torch.from_numpy(fit["raw_val"])).numpy(),
            rec(torch.from_numpy(fit["raw_test"])).numpy(),
        ),
    }
    metrics: dict = {}
    for mode, (s_val, s_te) in scores.items():
        val_thr = pick_threshold(y_val, s_val[:, m])
        entry = {"auprc": micro_auprc(y_te, s_te[:, m])}
        entry["ghv"] = global_hierarchy_violation(s_te, space["anc"], space["desc"], m)
        for name, thr in (("thr0", 0.5), ("val", val_thr)):
            p, r, f1 = micro_prf(y_te, s_te[:, m], thr)
            entry[name] = {
                "threshold": thr,
                "f1": f1,
                "precision": p,
                "recall": r,
                "macro_f1": macro_f1(y_te, s_te[:, m], thr),
                "val_f1": micro_prf(y_val, s_val[:, m], thr)[2],
            }
        metrics[mode] = entry
    return metrics


def train_hpt(dataset_name: str, args) -> dict:
    """``--method globalHPT``: train, evaluate and save the artefacts."""
    from hmc.datasets.dataset_manager import (  # pylint: disable=import-outside-toplevel
        initialize_dataset_experiments,
    )

    mgr = initialize_dataset_experiments(
        dataset_name,
        device="cpu",
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_max_records=args.dataset.arxiv_max_records,
        model_cache_dir=args.dataset.model_cache_dir,
        arxiv_load_features=False,
    )
    cfg = HPTConfig(
        backbone=BACKBONES.get(args.hpt_backbone, args.hpt_backbone),
        loss=args.consistency_loss,
        lambda_hier=args.lambda_hier,
        mlm=args.hpt_mlm,
        graph_direction=args.hpt_graph,
        lr=args.hpt_lr,
        batch_size=args.hpt_batch_size,
        grad_accum=args.grad_accum,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        model_cache_dir=args.dataset.model_cache_dir,
    )
    fit = fit_hpt(mgr, cfg)
    metrics = evaluate_hpt(fit)
    metrics.update(
        best_epoch=fit["best_epoch"],
        train_time_s=fit["train_time_s"],
        history=fit["history"],
    )
    rec = fit["space"]["sparse_r"].reconcile(torch.from_numpy(fit["raw_test"])).numpy()
    run_config = {
        "method": "globalHPT",
        "dataset": dataset_name,
        **asdict(cfg),
        "n_train": fit["n"][0],
        "n_valid": fit["n"][1],
        "n_test": fit["n"][2],
        "n_labels": len(fit["space"]["label_cols"]),
    }
    _save_artefacts(args.results_path, fit["raw_test"], rec, metrics, run_config)
    logger.info(
        "globalHPT %s: test micro-F1 rec %.4f (val thr) / macro-F1 %.4f",
        dataset_name,
        metrics["rec"]["val"]["f1"],
        metrics["rec"]["val"]["macro_f1"],
    )
    args.score = metrics
    return metrics

"""Training loop for end-to-end transformer fine-tuning (--method globalE2E)."""

import logging
import time

import torch
from torch import nn
from tqdm import tqdm

from hmc.models.global_classifier.constraint.model import get_constr_out
from hmc.pipeline.global_classifier.core.train import (
    _compute_global_score,
    _compute_local_scores,
)
from hmc.utils.datasets.labels import global_to_local_predictions
from hmc.utils.path.files import create_dir
from hmc.utils.path.output import save_dict_to_json
from hmc.utils.train.job import find_global_best_threshold, log_system_info


def _to_device(token_batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in token_batch.items()}


def _run_e2e_training_loop(model, args, optimizer, criterion) -> tuple:
    """MC-loss training loop for E2E model.

    DataLoader items: (token_dict, target_dict) where target_dict["global"]
    is the global binary label tensor.
    """
    to_eval = args.to_eval.to(args.device)
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        for token_batch, target_batch in tqdm(
            args.train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False
        ):
            token_batch = _to_device(token_batch, args.device)
            labels = target_batch["global"].to(args.device)

            optimizer.zero_grad()
            output = model(**token_batch)

            constr_output = get_constr_out(output, args.r_matrix)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, args.r_matrix)
            train_output = (1 - labels) * constr_output.double() + labels * train_output
            loss = criterion(train_output[:, to_eval].float(), labels[:, to_eval])
            loss.backward()
            optimizer.step()

    return log_system_info(args.device), time.perf_counter() - start


def _collect_e2e_test_outputs(model, args) -> tuple:
    """Run inference on test set; return (constr_test, y_test, to_eval)."""
    to_eval = args.to_eval.to("cpu")
    model.eval()
    constr_parts, y_parts = [], []

    with torch.no_grad():
        for token_batch, target_batch in args.test_loader:
            token_batch = _to_device(token_batch, args.device)
            labels = target_batch["global"]
            output = model(**token_batch).cpu()
            constr_parts.append(output)
            y_parts.append(labels)

    return torch.cat(constr_parts, dim=0), torch.cat(y_parts, dim=0), to_eval


def train_e2e_step(args):
    """Train and evaluate the E2E model, saving test-scores.json to results_path."""
    model = args.model.to(args.device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.transformer.parameters(), "lr": args.lr_transformer},
            {"params": model.head_parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCELoss()

    usage, total_time = _run_e2e_training_loop(model, args, optimizer, criterion)
    logging.info("E2E training time: %.1f s", total_time)

    constr_test, y_test, to_eval = _collect_e2e_test_outputs(model, args)

    best_threshold = 0.5
    if args.best_threshold:
        best_threshold, _ = find_global_best_threshold(
            y_test[:, to_eval], constr_test[:, to_eval], args, mode="global"
        )

    y_pred_local = global_to_local_predictions(
        constr_test.data > best_threshold,
        args.hmc_dataset.local_nodes_idx,
        args.hmc_dataset.nodes_idx,
    )
    y_test_local = global_to_local_predictions(
        y_test,
        args.hmc_dataset.local_nodes_idx,
        args.hmc_dataset.nodes_idx,
    )

    scores = _compute_local_scores(y_test_local, y_pred_local)
    scores["global"] = _compute_global_score(
        constr_test,
        y_test,
        to_eval,
        best_threshold,
        {"usage": usage, "total_time": total_time},
    )

    create_dir(args.results_path)
    save_dict_to_json(scores, f"{args.results_path}/test-scores.json")
    args.score = scores["global"]
    return args

"""LLM-assisted reranking pipeline for hierarchical classification."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.llm.ollama_agent import parse_selected_labels, rerank_document_labels_result
from hmc.models.global_classifier.e2e.model import E2EConstrainedModel
from hmc.pipeline.global_classifier.e2e_train import train_e2e_step
from hmc.pipeline.global_classifier.main import _get_transformer_dataset
from hmc.utils.model_cache import ensure_transformer_model_cached

log = logging.getLogger(__name__)

_GLOBAL_LLM_DEFAULTS = {
    "llm_top_k": 8,
    "llm_margin": 0.12,
    "llm_confidence_threshold": 0.60,
    "llm_max_tokens": 256,
    "llm_temperature": 0.0,
    "llm_k_max": 0,
    "llm_expand_hierarchy": False,
    "llm_max_calls": 0,
    "llm_cache": True,
    "llm_cache_dir": "",
    "llm_fallback_on_error": True,
    "llm_preserve_scores": True,
    "llm_max_document_chars": 6000,
}

_LITE_DEFAULTS = {
    "llm_top_k": 8,
    "llm_margin": 0.20,
    "llm_confidence_threshold": 0.75,
    "llm_max_tokens": 128,
    "llm_temperature": 0.0,
    "llm_k_max": 20,
    "llm_expand_hierarchy": True,
    "llm_max_calls": 0,
    "llm_cache": True,
    "llm_cache_dir": "",
    "llm_fallback_on_error": True,
    "llm_preserve_scores": True,
    "llm_max_document_chars": 3000,
}


def _build_r_matrix(hierarchy) -> torch.Tensor:
    import networkx as nx

    r_matrix = np.zeros(hierarchy.a.shape)
    np.fill_diagonal(r_matrix, 1)
    g = nx.DiGraph(hierarchy.a)
    for i in range(len(hierarchy.a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r_matrix[i, ancestors] = 1
    return torch.tensor(r_matrix).transpose(1, 0).unsqueeze(0)


def _subset_texts(dataset, subset) -> list[str]:
    records = getattr(dataset, "records", None)
    if records is None:
        return []
    indices = getattr(subset, "indices", None)
    if indices is None:
        return []
    return [records[idx]["text"] for idx in indices]


def _build_hierarchy_path(hierarchy, label: str) -> list[str]:
    try:
        import networkx as nx

        if label not in hierarchy.nodes_idx:
            return []
        path = nx.shortest_path(hierarchy.g_t, "root", label)
        return [node for node in path if node != "root"]
    except Exception:  # pragma: no cover - best-effort metadata
        return []


def _hierarchy_neighbors(hierarchy, label: str) -> list[str]:
    """Return parent, child, and sibling labels using the root->child graph."""
    graph = getattr(hierarchy, "g_t", None)
    if graph is None or label not in graph:
        return []

    neighbors: set[str] = set()
    parents = list(graph.predecessors(label))
    children = list(graph.successors(label))
    neighbors.update(parents)
    neighbors.update(children)
    for parent in parents:
        neighbors.update(graph.successors(parent))
    neighbors.discard(label)
    neighbors.discard("root")
    return sorted(neighbors)


def _adaptive_candidate_count(
    row: torch.Tensor,
    *,
    top_k: int,
    k_max: int,
    threshold: float,
    margin: float,
) -> int:
    """Increase candidate count for low-confidence samples."""
    if k_max <= top_k or row.numel() <= top_k:
        return min(top_k, row.numel())

    top_vals, _ = torch.topk(row, k=min(2, row.numel()))
    top1 = float(top_vals[0])
    top2 = float(top_vals[1]) if top_vals.numel() > 1 else 0.0
    gap = top1 - top2

    if top1 < threshold * 0.8 or gap < margin * 0.5:
        return min(k_max, row.numel())
    if top1 < threshold or gap < margin:
        return min(max(top_k + 2, top_k), k_max, row.numel())
    return min(top_k, row.numel())


def _build_candidate_indices(
    *,
    row: torch.Tensor,
    full_row: torch.Tensor,
    to_eval: torch.Tensor,
    hierarchy,
    idx_to_node: dict[int, str],
    node_to_idx: dict[str, int],
    top_k: int,
    k_max: int,
    threshold: float,
    margin: float,
    expand_hierarchy: bool,
) -> list[int]:
    """Build global label indices for the LLM candidate set."""
    eval_indices = torch.nonzero(to_eval, as_tuple=False).flatten()
    candidate_count = _adaptive_candidate_count(
        row,
        top_k=top_k,
        k_max=k_max,
        threshold=threshold,
        margin=margin,
    )
    _, top_idx = torch.topk(row, k=candidate_count)
    candidate_indices = {int(eval_indices[local_idx].item()) for local_idx in top_idx}

    if expand_hierarchy:
        for global_idx in list(candidate_indices):
            label = idx_to_node.get(global_idx)
            if not label:
                continue
            for neighbor in _hierarchy_neighbors(hierarchy, label):
                neighbor_idx = node_to_idx.get(neighbor)
                if neighbor_idx is not None and bool(to_eval[neighbor_idx]):
                    candidate_indices.add(neighbor_idx)

    return sorted(
        candidate_indices,
        key=lambda idx: float(full_row[idx]),
        reverse=True,
    )[: max(k_max, top_k)]


def _update_candidate_recall(stats: dict[str, Any], y_row: torch.Tensor, candidate_indices: list[int]) -> None:
    true_indices = set(torch.nonzero(y_row > 0, as_tuple=False).flatten().tolist())
    true_indices.discard(0)
    if not true_indices:
        return
    stats["candidate_true_total"] += len(true_indices)
    stats["candidate_true_hits"] += len(true_indices.intersection(candidate_indices))


def _expand_selected_with_ancestors(
    *,
    selected: set[str],
    hierarchy,
    node_to_idx: dict[str, int],
    to_eval: torch.Tensor,
) -> tuple[set[str], int]:
    """Add hierarchy ancestors so LLM decisions remain valid under HMC metrics."""
    expanded = set(selected)
    for label in list(selected):
        for ancestor in _build_hierarchy_path(hierarchy, label):
            ancestor_idx = node_to_idx.get(ancestor)
            if ancestor_idx is not None and bool(to_eval[ancestor_idx]):
                expanded.add(ancestor)
    return expanded, len(expanded) - len(selected)


def _format_duration(seconds: float | None) -> str:
    """Return a compact human-readable duration."""
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _truncate_document_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep prompt size bounded while preserving the beginning and end of the document."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head_chars = max(1, int(max_chars * 0.7))
    tail_chars = max(1, max_chars - head_chars)
    truncated = (
        text[:head_chars].rstrip()
        + "\n...[truncated]...\n"
        + text[-tail_chars:].lstrip()
    )
    return truncated, True


def _estimate_remaining_llm_seconds(
    *,
    attempts: int,
    processed_samples: int,
    total_samples: int,
    max_calls: int,
    avg_call_seconds: float,
) -> tuple[float, float]:
    """Estimate remaining LLM calls and seconds from observed call density."""
    if attempts <= 0 or processed_samples <= 0:
        return 0.0, 0.0
    observed_call_rate = attempts / processed_samples
    estimated_total_calls = observed_call_rate * total_samples
    if max_calls:
        estimated_total_calls = min(float(max_calls), estimated_total_calls)
    estimated_remaining_calls = max(0.0, estimated_total_calls - attempts)
    return estimated_remaining_calls, estimated_remaining_calls * avg_call_seconds


def _make_postprocessor(args):
    hierarchy = args.hmc_dataset.hierarchy_manager
    idx_to_node = {v: k for k, v in args.hmc_dataset.nodes_idx.items()}
    node_to_idx = args.hmc_dataset.nodes_idx
    model = getattr(args, "llm_model", "qwen3:14b")
    base_url = getattr(args, "llm_base_url", "http://localhost:11434")
    num_ctx = max(0, int(getattr(args, "llm_num_ctx", 4096)))
    top_k = max(1, int(getattr(args, "llm_top_k", 8)))
    k_max = int(getattr(args, "llm_k_max", 0) or top_k)
    k_max = max(top_k, k_max)
    margin = float(getattr(args, "llm_margin", 0.12))
    threshold = float(getattr(args, "llm_confidence_threshold", 0.60))
    temperature = float(getattr(args, "llm_temperature", 0.0))
    max_tokens = int(getattr(args, "llm_max_tokens", 256))
    only_uncertain = bool(getattr(args, "llm_only_uncertain", True))
    expand_hierarchy = bool(getattr(args, "llm_expand_hierarchy", False))
    max_calls = max(0, int(getattr(args, "llm_max_calls", 0) or 0))
    timeout = max(1, int(getattr(args, "llm_timeout", 120)))
    cache_enabled = bool(getattr(args, "llm_cache", True))
    cache_dir_arg = str(getattr(args, "llm_cache_dir", "") or "").strip()
    cache_dir = (
        cache_dir_arg
        if cache_dir_arg
        else os.path.join(args.results_path, "llm-cache")
    )
    fallback_on_error = bool(getattr(args, "llm_fallback_on_error", True))
    preserve_scores = bool(getattr(args, "llm_preserve_scores", True))
    max_document_chars = max(0, int(getattr(args, "llm_max_document_chars", 6000)))
    test_texts = _subset_texts(args.text_dataset, args.test_subset)

    stats: dict[str, Any] = {
        "samples": 0,
        "top_k": top_k,
        "k_max": k_max,
        "threshold": threshold,
        "margin": margin,
        "provider": "ollama",
        "model": model,
        "base_url": base_url,
        "num_ctx": num_ctx,
        "expand_hierarchy": expand_hierarchy,
        "max_calls": max_calls,
        "cache_enabled": cache_enabled,
        "cache_dir": cache_dir if cache_enabled else "",
        "fallback_on_error": fallback_on_error,
        "preserve_scores": preserve_scores,
        "max_document_chars": max_document_chars,
        "llm_attempts": 0,
        "llm_successes": 0,
        "llm_errors": 0,
        "llm_fallbacks": 0,
        "llm_cache_hits": 0,
        "llm_cache_misses": 0,
        "llm_api_calls": 0,
        "llm_call_seconds_total": 0.0,
        "llm_avg_call_seconds": 0.0,
        "llm_eta_seconds": 0.0,
        "llm_estimated_remaining_calls": 0.0,
        "llm_elapsed_seconds": 0.0,
        "confident_skips": 0,
        "budget_skips": 0,
        "no_candidate_skips": 0,
        "empty_selections": 0,
        "selected_labels": 0,
        "rejected_labels": 0,
        "hierarchy_closure_added": 0,
        "document_chars_original_total": 0,
        "document_chars_sent_total": 0,
        "document_truncations": 0,
        "candidate_true_hits": 0,
        "candidate_true_total": 0,
        "candidate_recall": 0.0,
    }

    def _postprocess(constr_test: torch.Tensor, y_test, to_eval, inner_args):
        adjusted = constr_test.clone().detach().cpu()
        probabilities = adjusted[:, to_eval]
        sample_count = min(len(test_texts), adjusted.shape[0])
        stats["samples"] = sample_count
        llm_stage_start = time.monotonic()
        last_progress_log = llm_stage_start
        log.info(
            "Ollama reranker started: samples=%s model=%s top_k=%s k_max=%s "
            "max_calls=%s",
            sample_count,
            model,
            top_k,
            k_max,
            max_calls or "unlimited",
        )

        for i in range(sample_count):
            row = probabilities[i]
            top_vals, _ = torch.topk(row, k=min(2, row.numel()))
            candidate_indices = _build_candidate_indices(
                row=row,
                full_row=adjusted[i],
                to_eval=to_eval,
                hierarchy=hierarchy,
                idx_to_node=idx_to_node,
                node_to_idx=node_to_idx,
                top_k=top_k,
                k_max=k_max,
                threshold=threshold,
                margin=margin,
                expand_hierarchy=expand_hierarchy,
            )
            _update_candidate_recall(stats, y_test[i], candidate_indices)

            if only_uncertain and row.numel() > 1:
                top1 = float(top_vals[0])
                top2 = float(top_vals[1]) if top_vals.numel() > 1 else 0.0
                if top1 >= threshold and (top1 - top2) >= margin:
                    stats["confident_skips"] += 1
                    continue

            if max_calls and stats["llm_attempts"] >= max_calls:
                stats["budget_skips"] += 1
                continue

            candidate_labels: list[dict[str, Any]] = []
            for global_idx in candidate_indices:
                label = idx_to_node.get(global_idx)
                if not label or label == "root":
                    continue
                candidate_labels.append(
                    {
                        "label": label,
                        "score": float(adjusted[i, global_idx]),
                        "path": " > ".join(_build_hierarchy_path(hierarchy, label)),
                    }
                )

            if not candidate_labels:
                stats["no_candidate_skips"] += 1
                continue

            stats["llm_attempts"] += 1
            call_start = time.monotonic()
            document_text, was_truncated = _truncate_document_text(
                test_texts[i],
                max_document_chars,
            )
            stats["document_chars_original_total"] += len(test_texts[i])
            stats["document_chars_sent_total"] += len(document_text)
            if was_truncated:
                stats["document_truncations"] += 1

            try:
                result = rerank_document_labels_result(
                    document_text=document_text,
                    candidate_labels=candidate_labels,
                    model=model,
                    base_url=base_url,
                    num_ctx=num_ctx,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    cache_dir=cache_dir if cache_enabled else None,
                )
                response = result.payload
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                stats["llm_errors"] += 1
                if fallback_on_error:
                    stats["llm_fallbacks"] += 1
                    log.warning(
                        "Ollama reranker failed for sample %s; keeping base prediction: %s",
                        i,
                        exc,
                    )
                    continue
                raise

            if result.cache_hit:
                stats["llm_cache_hits"] += 1
            else:
                stats["llm_cache_misses"] += 1
                stats["llm_api_calls"] += 1
            call_seconds = time.monotonic() - call_start
            stats["llm_successes"] += 1
            stats["llm_call_seconds_total"] += call_seconds
            stats["llm_avg_call_seconds"] = (
                stats["llm_call_seconds_total"] / stats["llm_successes"]
            )

            now = time.monotonic()
            processed_samples = i + 1
            remaining_calls, eta_seconds = _estimate_remaining_llm_seconds(
                attempts=stats["llm_attempts"],
                processed_samples=processed_samples,
                total_samples=sample_count,
                max_calls=max_calls,
                avg_call_seconds=stats["llm_avg_call_seconds"],
            )
            stats["llm_elapsed_seconds"] = now - llm_stage_start
            stats["llm_eta_seconds"] = eta_seconds
            stats["llm_estimated_remaining_calls"] = remaining_calls

            should_log_progress = (
                stats["llm_attempts"] == 1
                or stats["llm_attempts"] % 10 == 0
                or now - last_progress_log >= 60
            )
            if should_log_progress:
                log.info(
                    "Ollama reranker progress: sample=%s/%s attempts=%s "
                    "avg_call=%s elapsed=%s eta=%s est_remaining_calls=%.1f",
                    processed_samples,
                    sample_count,
                    stats["llm_attempts"],
                    _format_duration(stats["llm_avg_call_seconds"]),
                    _format_duration(stats["llm_elapsed_seconds"]),
                    _format_duration(stats["llm_eta_seconds"]),
                    stats["llm_estimated_remaining_calls"],
                )
                last_progress_log = now

            valid_labels = {cand["label"] for cand in candidate_labels}
            raw_selected = set(parse_selected_labels(response)).intersection(valid_labels)

            if not raw_selected:
                stats["empty_selections"] += 1
                for cand in candidate_labels:
                    global_idx = args.hmc_dataset.nodes_idx[cand["label"]]
                    adjusted[i, global_idx] = 0.0
                continue

            selected, closure_added = _expand_selected_with_ancestors(
                selected=raw_selected,
                hierarchy=hierarchy,
                node_to_idx=args.hmc_dataset.nodes_idx,
                to_eval=to_eval,
            )
            stats["hierarchy_closure_added"] += closure_added

            for cand in candidate_labels:
                global_idx = args.hmc_dataset.nodes_idx[cand["label"]]
                adjusted[i, global_idx] = (
                    float(cand["score"]) if preserve_scores and cand["label"] in selected
                    else 1.0 if cand["label"] in selected
                    else 0.0
                )
            for label in selected.difference(valid_labels):
                global_idx = args.hmc_dataset.nodes_idx[label]
                if not preserve_scores:
                    adjusted[i, global_idx] = 1.0
            stats["selected_labels"] += len(selected)
            stats["rejected_labels"] += len(candidate_labels) - len(raw_selected)

        if stats["candidate_true_total"]:
            stats["candidate_recall"] = stats["candidate_true_hits"] / stats["candidate_true_total"]
        stats["llm_elapsed_seconds"] = time.monotonic() - llm_stage_start
        stats["llm_eta_seconds"] = 0.0
        stats["llm_estimated_remaining_calls"] = 0.0

        log.info(
            "Ollama reranker (%s): attempts=%s api_calls=%s cache_hits=%s "
            "errors=%s fallbacks=%s samples=%s elapsed=%s avg_call=%s "
            "top_k=%s k_max=%s threshold=%.2f margin=%.2f candidate_recall=%.4f",
            model,
            stats["llm_attempts"],
            stats["llm_api_calls"],
            stats["llm_cache_hits"],
            stats["llm_errors"],
            stats["llm_fallbacks"],
            stats["samples"],
            _format_duration(stats["llm_elapsed_seconds"]),
            _format_duration(stats["llm_avg_call_seconds"]),
            top_k,
            k_max,
            threshold,
            margin,
            stats["candidate_recall"],
        )
        inner_args.llm_rerank_stats = stats
        return adjusted

    return _postprocess


def train_global_llm(dataset_name, args):
    """Train an E2E base model and rerank uncertain predictions with an LLM."""
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    args.device = torch.device(args.device)
    model_name = args.dataset.arxiv_model_name

    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_model_name=model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_load_features=False,
        model_cache_dir=args.dataset.model_cache_dir,
    )

    args.data = dataset_name
    args.ontology = None
    args.to_eval = torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()

    _is_gofun_llm = any(suffix in (dataset_name or "")
                         for suffix in ("_FUN", "_GO", "_others"))
    defaults = (
        args.registry.wos_defaults if dataset_name == "wos"
        else args.registry.gofun_defaults if _is_gofun_llm
        else args.registry.arxiv_defaults
    )
    args.hidden_dim = defaults["hidden_dim"]
    args.lr = defaults["lr"]
    if args.epochs == defaults["epochs"] or args.epochs <= 0:
        args.epochs = 10
    args.weight_decay = defaults["weight_decay"]
    args.batch_size = 4
    args.output_dim = args.hmc_dataset.output_dim
    args.num_to_skip = 1

    args.r_matrix = _build_r_matrix(args.hmc_dataset.hierarchy_manager).to(args.device)
    args.results_path = os.path.join(
        args.output_path,
        "train",
        f"{args.method}-{dataset_name}",
        args.job_id,
    )

    local_model_path = ensure_transformer_model_cached(
        model_name,
        args.dataset.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    text_dataset, _ = _get_transformer_dataset(dataset_name, args, tokenizer, local_model_path)
    train_set, _val_set, test_set = text_dataset.get_datasets()

    args.text_dataset = text_dataset
    args.test_subset = test_set

    args.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.model = E2EConstrainedModel(
        model_name=local_model_path,
        output_dim=args.output_dim,
        r_matrix=args.r_matrix,
        hidden_dim=args.hidden_dim,
        num_layers=defaults["num_layers"],
        dropout=defaults["dropout"],
        model_cache_dir=args.dataset.model_cache_dir,
    )

    postprocess_fn = _make_postprocessor(args)
    return train_e2e_step(args, postprocess_fn=postprocess_fn)


def train_global_llm_lite(dataset_name, args):
    """Train an E2E base model with a cheaper LLM gate."""
    for key, value in _LITE_DEFAULTS.items():
        cli_flag = f"--{key}"
        if cli_flag not in sys.argv and getattr(args, key, None) == _GLOBAL_LLM_DEFAULTS.get(key):
            setattr(args, key, value)
    args.llm_only_uncertain = True
    return train_global_llm(dataset_name, args)

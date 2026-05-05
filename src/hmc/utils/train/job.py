"""
Utility functions for job ID generation, timing, and command-line argument parsing.

This module provides:
- Functions to generate unique job IDs with timestamps.
- Timer utilities for measuring elapsed time.
- Parsing helpers to convert string flags to boolean values in argument objects.
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import psutil
import torch
from tqdm import tqdm

from hmc.utils.dataset.labels import local_to_global_predictions
from hmc.utils.metrics.calculate_metrics import calculate_metrics


def create_job_id_name(prefix="job"):
    """
    Create a unique job ID using the current date and time.

    Args:
        prefix (str): Optional prefix for the job ID (default is "job").

    Returns:
        str: A unique job ID string.
    """
    now = datetime.now()
    job_id = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    return job_id


def start_timer():
    """Start a timer and return the start time."""
    return time.perf_counter()


def end_timer(start):
    """End the timer and print the elapsed time since start."""
    end = time.perf_counter()
    elapsed = end - start
    print(f"Tempo de treino: {elapsed:.2f} segundos")
    return elapsed


def log_gpu_memory(device):
    """Log GPU memory information."""
    result = {}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        total_mib = prop.total_memory / (1024**2)  # Total em MiB

        allocated_mib = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mib = torch.cuda.memory_reserved(device) / (
            1024**2
        )  # MiB como nvidia-smi
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

        # SOMA alocado + reservado (em MiB)
        soma_mib = (
            torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)
        ) / (1024**2)

        result.update(
            {
                "total_mib": total_mib,
                "allocated_mib": allocated_mib,
                "reserved_mib": reserved_mib,
                "peak_mib": peak_mib,
                "soma_allocated_reserved_mib": soma_mib,
            }
        )

    torch.cuda.empty_cache()
    return result


def log_cpu_ram(result):
    """Log CPU RAM information."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_used = process.memory_info().rss / 1024**3  # GB
    result["cpu-percent"] = cpu_percent
    result["cpu-ram"] = ram_used
    return result


def log_system_info(device):
    """Log system information."""
    result = {}
    result = log_gpu_memory(device)
    result = log_cpu_ram(result)
    return result


def find_local_best_threshold(
    local_outputs,
    local_inputs,
    args,
):
    """
    Find the best threshold for local predictions.
    Args:
        local_outputs: Array of local predictions.
        local_inputs: Array of local targets.
        args: Object containing dataset information.
    Returns:
        Tuple of best threshold and best scores.
    """
    if args.best_threshold:
        logging.info("find best theshold")
        best_thresholds = {level: 0 for _, level in enumerate(args.active_levels)}
        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            level: {
                "precision": 0,
                "recall": 0,
                "f1score": 0,
                "average_precision_score": 0,
            }
            for _, level in enumerate(args.active_levels)
        }
        logging.info("Evaluating %d active levels...", len(args.active_levels))
        for level in args.active_levels:
            y_pred = local_outputs[level].to("cpu").numpy()
            y_true = local_inputs[level].to("cpu").int().numpy()
            for actual_threshold in thresholds:
                y_pred_binary = y_pred > actual_threshold
                metrics = calculate_metrics(y_true, y_pred, y_pred_binary)

                if metrics["f1score"] > best_scores[level]["f1score"]:
                    best_thresholds[level] = actual_threshold
                    best_scores[level] = metrics

        logging.info("Best thresholds per level:")
        for idx in args.active_levels:
            logging.info(
                "Level %d: threshold=%.2f, precision=%.4f,"
                + "recall=%.4f, f1-score=%.4f avg score=%.4f",
                idx,
                best_thresholds[idx],
                best_scores[idx]["precision"],
                best_scores[idx]["recall"],
                best_scores[idx]["f1score"],
                best_scores[idx]["average_precision_score"],
            )
    else:
        best_thresholds = {level: 0.5 for _, level in enumerate(args.active_levels)}
    return best_thresholds, best_scores


def find_global_best_threshold(
    all_y_pred,
    y_true_global_original,
    args,
):
    """
    Find the best threshold for global predictions.
    Args:
        all_y_pred: Array of global predictions.
        y_true_global_original: Array of global targets.
        args: Object containing dataset information.
    Returns:
        Tuple of best threshold and best scores.
    """
    # Concat global targets
    best_threshold = 0.5
    best_scores = {
        "precision": 0,
        "recall": 0,
        "f1score": 0,
        "average_precision_score": 0,
    }
    if args.best_threshold:
        logging.info("finding best threshold")

        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            if args.method in ["global_baseline", "global"]:
                y_pred_global = all_y_pred
                y_pred_global_binary = all_y_pred > actual_threshold
            else:
                y_pred_global, y_pred_global_binary = local_to_global_predictions(
                    all_y_pred,
                    args.hmc_dataset.local_nodes_idx,
                    args.hmc_dataset.nodes_idx,
                    threshold=actual_threshold,
                )
            metrics = calculate_metrics(
                y_true_global_original,
                y_pred_global,
                y_pred_global_binary,
            )
            if metrics["f1score"] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1score": metrics["f1score"],
                }

        thresholds = np.linspace(best_threshold - 0.01, best_threshold, 10)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
            "average_precision_score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            if args.method in ["global_baseline", "global"]:
                y_pred_global = all_y_pred
                y_pred_global_binary = all_y_pred > actual_threshold
            else:
                y_pred_global, y_pred_global_binary = local_to_global_predictions(
                    all_y_pred,
                    args.hmc_dataset.local_nodes_idx,
                    args.hmc_dataset.nodes_idx,
                    threshold=actual_threshold,
                )
            metrics = calculate_metrics(
                y_true_global_original,
                y_pred_global,
                y_pred_global_binary,
            )
            if metrics["f1score"] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1score": metrics["f1score"],
                    "average_precision_score": metrics["average_precision_score"],
                }

        logging.info("Best threshold: %.2f", best_threshold)
        logging.info("Best scores: %s", best_scores)

    return best_threshold, best_scores

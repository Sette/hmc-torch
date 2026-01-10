"""
Utility functions for job ID generation, timing, and command-line argument parsing.

This module provides:
- Functions to generate unique job IDs with timestamps.
- Timer utilities for measuring elapsed time.
- Parsing helpers to convert string flags to boolean values in argument objects.
"""
import os
import psutil
import torch
import time
from datetime import datetime


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


def parse_str_flags(args):
    """Convert certain string command-line arguments to boolean."""
    args.best_threshold = args.best_threshold == "true"
    args.use_sample = args.use_sample == "true"
    args.hpo_by_level = args.hpo_by_level == "true"
    args.save_torch_dataset = args.save_torch_dataset == "true"
    args.warmup = args.warmup == "true"
    args.focal_loss = args.focal_loss == "true"
    args.hpo = args.hpo == "true"
    return args


def log_gpu_memory(device):
    result = {}
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        result['gpu-allocated'] = allocated
        result['gpu-reserved'] = reserved
        print(f"GPU - Alocada: {allocated:.2f}GB, Reservada: {reserved:.2f}GB")
    torch.cuda.empty_cache()  # Opcional para reset
    return result


def log_cpu_ram(result):
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_used = process.memory_info().rss / 1024**3  # GB
    result['cpu-percent'] = cpu_percent
    result['cpu-ram'] = ram_used
    print(f"CPU: {cpu_percent:.1f}%, RAM processo: {ram_used:.2f}GB")
    return result


def log_system_info(device):
    result = {}
    result = log_gpu_memory(device)
    result = log_cpu_ram(result)
    return result
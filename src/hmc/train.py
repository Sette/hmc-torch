"""High-level training API for HMC-Torch.

Provides a single :func:`train` entry point that covers all supported
methods and datasets — both built-in and user-registered.

.. code-block:: python

    import hmc

    # Quick training
    results = hmc.train("wos", method="globalE2E", device="cuda", epochs=5)

    # With a custom dataset
    results = hmc.train(
        "my_data", method="tabular_mlp", device="cpu",
        dataset_path="./my_data", output_path="./runs",
    )
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_configs(
    dataset_name: str,
    method: str,
    device: str,
    epochs: int,
    batch_size: int,
    dataset_path: str,
    output_path: str,
    seed: int,
    **kwargs: Any,
) -> tuple:
    """Build DatasetConfig, TrainingConfig, and Args from kwargs."""
    from dataclasses import (
        fields as dc_fields,  # pylint: disable=import-outside-toplevel
    )

    from hmc.arguments import (  # pylint: disable=import-outside-toplevel
        Args,
        DatasetConfig,
        TrainingConfig,
    )

    dataset = DatasetConfig(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        arxiv_model_name=kwargs.pop("arxiv_model_name", "allenai/specter2_base"),
        arxiv_max_records=kwargs.pop("arxiv_max_records", 50_000),
        model_cache_dir=kwargs.pop("model_cache_dir", "./models"),
    )

    _training_fields = {f.name for f in dc_fields(TrainingConfig)}
    training_kwargs = {
        "batch_size": batch_size,
        "device": device,
        "epochs": epochs,
        "seed": seed,
    }
    for k, v in kwargs.items():
        if k in _training_fields:
            training_kwargs[k] = v

    training = TrainingConfig(**training_kwargs)

    return Args(
        dataset=dataset,
        output_path=output_path,
        method=method,
        training=training,
    )


def train(
    dataset_name: str,
    method: str = "global",
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 32,
    dataset_path: str = "./data",
    output_path: str = "./output",
    seed: int = 0,
    **kwargs: Any,
) -> dict:
    """Train an HMC model on *dataset_name* using *method*.

    This is a convenience wrapper around :func:`hmc.main.main` that
    builds the :class:`Args` object from keyword arguments.

    Args:
        dataset_name: Dataset identifier (e.g. ``"wos"``, ``"seq_FUN"``).
        method: One of ``"global"``, ``"globalE2E"``, ``"globalSOTA"``,
            ``"local"``, ``"localE2E"``, ``"tabular_gbdt"``, ``"tabular_mlp"``.
        device: ``"cpu"`` or ``"cuda"``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        dataset_path: Root directory for dataset files.
        output_path: Directory for outputs (models, metrics, manifests).
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments forwarded to :class:`TrainingConfig`
            (e.g. ``hidden_dim``, ``lr``, ``dropout``,
            ``arxiv_model_name``, ``non_lin``, etc.).

    Returns:
        Dictionary with training metrics (``micro_f1``, ``auprc``, …).
    """
    # Import here to avoid circular dependency on module load
    from hmc.main import main as _main_impl  # pylint: disable=import-outside-toplevel

    # Seed everything early
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    args = _build_configs(
        dataset_name=dataset_name,
        method=method,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        dataset_path=dataset_path,
        output_path=output_path,
        seed=seed,
        **kwargs,
    )

    return _main_impl(args)


# Backward-compatible alias
run = train

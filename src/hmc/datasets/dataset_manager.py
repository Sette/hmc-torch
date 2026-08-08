"""Dataset manager — entry point for loading all supported HMC datasets.

Uses :class:`hmc.data.DatasetRegistry` internally so that both built-in
and user-registered datasets are discoverable through a single API.

Supported families (built-in):
  - ``arxiv``, ``wos`` — text transformer-based (SPECTER2)
  - ``aapd``, ``rcv1``, ``eurlex`` — multi-label text
  - ``seq_FUN``, ``cellcycle_FUN``, ..., ``spo_GO`` — ARFF tabular (GoFun)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from hmc.data.registry import DatasetRegistry, pick_defaults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-register built-in datasets on first import
# ---------------------------------------------------------------------------
_registered_builtins: bool = False


def _ensure_builtins() -> None:
    """Register all built-in dataset factories.  Idempotent."""
    global _registered_builtins  # pylint: disable=global-statement
    if _registered_builtins:
        return
    _registered_builtins = True

    # -- ArXiv -----------------------------------------------------------
    def _make_arxiv(**kw: Any) -> Any:
        from hmc.datasets.arxiv.manager import (  # pylint: disable=import-outside-toplevel
            ArXivManager,
        )

        jsonl_path = kw.pop(
            "jsonl_path",
            os.path.join(
                kw.pop("dataset_path", "data"),
                "arxiv",
                "arxiv-metadata-oai-snapshot.json",
            ),
        )
        max_records = kw.pop("max_records", None)
        model_name = kw.pop("model_name", "allenai/specter2_base")
        cache_dir = kw.pop("cache_dir", None)
        load_features = kw.pop("load_features", True)
        model_cache_dir = kw.pop("model_cache_dir", "./models")
        return ArXivManager(
            jsonl_path=jsonl_path,
            model_name=model_name,
            max_records=max_records if max_records and max_records > 0 else None,
            cache_dir=cache_dir,
            load_features=load_features,
            model_cache_dir=model_cache_dir,
        )

    DatasetRegistry.register("arxiv", _make_arxiv)

    # -- WOS -------------------------------------------------------------
    def _make_wos(**kw: Any) -> Any:
        from hmc.datasets.wos.manager import (  # pylint: disable=import-outside-toplevel
            WOSManager,
        )

        data_dir = kw.pop(
            "data_dir",
            os.path.join(kw.pop("dataset_path", "data"), "wos"),
        )
        model_name = kw.pop("model_name", "allenai/specter2_base")
        cache_dir = kw.pop("cache_dir", None)
        load_features = kw.pop("load_features", True)
        model_cache_dir = kw.pop("model_cache_dir", "./models")
        return WOSManager(
            data_dir=data_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            load_features=load_features,
            model_cache_dir=model_cache_dir,
        )

    DatasetRegistry.register("wos", _make_wos)

    # -- AAPD ------------------------------------------------------------
    def _make_aapd(**kw: Any) -> Any:
        from hmc.datasets.aapd.manager import (  # pylint: disable=import-outside-toplevel
            AAPDManager,
        )

        data_dir = kw.pop(
            "data_dir",
            os.path.join(kw.pop("dataset_path", "data"), "aapd"),
        )
        model_name = kw.pop("model_name", "allenai/specter2_base")
        max_records = kw.pop("max_records", None)
        cache_dir = kw.pop("cache_dir", None)
        load_features = kw.pop("load_features", True)
        model_cache_dir = kw.pop("model_cache_dir", "./models")
        return AAPDManager(
            data_dir=data_dir,
            model_name=model_name,
            max_records=max_records if max_records and max_records > 0 else None,
            cache_dir=cache_dir,
            load_features=load_features,
            model_cache_dir=model_cache_dir,
        )

    DatasetRegistry.register("aapd", _make_aapd)

    # -- RCV1 ------------------------------------------------------------
    def _make_rcv1(**kw: Any) -> Any:
        from hmc.datasets.rcv1.manager import (  # pylint: disable=import-outside-toplevel
            RCV1Manager,
        )

        data_dir = kw.pop(
            "data_dir",
            os.path.join(kw.pop("dataset_path", "data"), "rcv1"),
        )
        model_name = kw.pop("model_name", "allenai/specter2_base")
        max_records = kw.pop("max_records", None)
        cache_dir = kw.pop("cache_dir", None)
        load_features = kw.pop("load_features", True)
        model_cache_dir = kw.pop("model_cache_dir", "./models")
        return RCV1Manager(
            data_dir=data_dir,
            model_name=model_name,
            max_records=max_records if max_records and max_records > 0 else None,
            cache_dir=cache_dir,
            load_features=load_features,
            model_cache_dir=model_cache_dir,
        )

    DatasetRegistry.register("rcv1", _make_rcv1)

    # -- EURLex ----------------------------------------------------------
    def _make_eurlex(**kw: Any) -> Any:
        from hmc.datasets.eurlex.manager import (  # pylint: disable=import-outside-toplevel
            EURLexManager,
        )

        data_dir = kw.pop(
            "data_dir",
            os.path.join(kw.pop("dataset_path", "data"), "eurlex"),
        )
        model_name = kw.pop("model_name", "allenai/specter2_base")
        max_records = kw.pop("max_records", None)
        cache_dir = kw.pop("cache_dir", None)
        load_features = kw.pop("load_features", True)
        model_cache_dir = kw.pop("model_cache_dir", "./models")
        return EURLexManager(
            data_dir=data_dir,
            model_name=model_name,
            max_records=max_records if max_records and max_records > 0 else None,
            cache_dir=cache_dir,
            load_features=load_features,
            model_cache_dir=model_cache_dir,
        )

    DatasetRegistry.register("eurlex", _make_eurlex)

    logger.info(
        "Registered %d built-in dataset(s): %s",
        len(DatasetRegistry.list_available()),
        DatasetRegistry.list_available(),
    )


# ---------------------------------------------------------------------------
# GoFun factory (shared across all GoFun ARFF datasets)
# ---------------------------------------------------------------------------


def _load_gofun_dataset(
    name: str,
    device: str,
    dataset_path: str,
    is_global: bool,
    **_: Any,
) -> Any:
    """Load a GoFun ARFF dataset using HMCDatasetManager."""
    from hmc.datasets.gofun.manager import HMCDatasetManager
    from hmc.utils.datasets.paths import get_dataset_paths

    datasets = get_dataset_paths(dataset_path=dataset_path)

    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found in GoFun paths. "
            f"Available: {[k for k in datasets if '_FUN' in k or '_GO' in k or '_others' in k]}"
        )

    return HMCDatasetManager(
        dataset=datasets[name],
        dataset_type="arff",
        device=device,
        is_global=is_global,
    )


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------


def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_path: str = "data/",
    is_global: bool = False,
    arxiv_model_name: str = "allenai/specter2_base",
    arxiv_max_records: int = 50_000,
    arxiv_cache_dir: str | None = None,
    arxiv_load_features: bool = True,
    model_cache_dir: str = "./models",
    **kwargs: Any,
) -> Any:
    """Initialize and return a dataset manager for the specified dataset.

    Supports all built-in datasets (arxiv, wos, aapd, rcv1, eurlex, and
    GoFun ARFF datasets) plus any custom datasets registered via
    :meth:`DatasetRegistry.register` or entry points.

    Args:
        name: Dataset identifier (e.g. ``"wos"``, ``"seq_FUN"``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        dataset_path: Root directory for dataset files.
        is_global: Whether to load in global-classifier mode.
        arxiv_model_name: HuggingFace model for text embeddings.
        arxiv_max_records: Max records to load (0 = all).  For arxiv.
        arxiv_cache_dir: Directory for feature cache.
        arxiv_load_features: Whether to compute/load transformer features.
        model_cache_dir: Directory for downloaded transformer models.
        **kwargs: Additional keyword arguments forwarded to the manager.

    Returns:
        A manager instance satisfying :class:`DatasetManagerProtocol`.
    """
    _ensure_builtins()

    # GoFun ARFF datasets use a different loading path
    if any(suffix in (name or "") for suffix in ("_FUN", "_GO", "_others")):
        return _load_gofun_dataset(
            name=name,
            device=device,
            dataset_path=dataset_path,
            is_global=is_global,
        )

    # Build kwargs for the registry factory
    factory_kwargs: dict[str, Any] = {
        "dataset_path": dataset_path,
        "model_name": arxiv_model_name,
        "cache_dir": arxiv_cache_dir,
        "load_features": arxiv_load_features,
        "model_cache_dir": model_cache_dir,
    }

    if name == "arxiv" or name in ("aapd", "rcv1", "eurlex"):
        factory_kwargs["max_records"] = arxiv_max_records

    # Pass through any extra kwargs
    factory_kwargs.update(kwargs)

    return DatasetRegistry.get(name, **factory_kwargs)


# Re-export for convenience
__all__ = ["DatasetRegistry", "initialize_dataset_experiments", "pick_defaults"]

# Auto-register built-in datasets on module import so that
# DatasetRegistry.list_available() works without calling any function.
_ensure_builtins()

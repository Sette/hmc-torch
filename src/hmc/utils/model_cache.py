"""Utilities for materializing HuggingFace models into a local project cache."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_model_dir_name(model_name: str) -> str:
    return model_name.strip().replace("/", "__").replace("\\", "__")


def _has_model_weights(path: Path) -> bool:
    weight_patterns = (
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    if any((path / name).exists() for name in weight_patterns):
        return True
    return any(path.glob("pytorch_model-*.bin")) or any(path.glob("model-*.safetensors"))


def _has_tokenizer_files(path: Path) -> bool:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "spiece.model",
    )
    return any((path / name).exists() for name in tokenizer_files)


def is_transformer_model_cached(path: str | Path) -> bool:
    """Return True when a local directory can satisfy AutoModel and AutoTokenizer."""
    cache_path = Path(path)
    return (
        cache_path.is_dir()
        and (cache_path / "config.json").exists()
        and _has_model_weights(cache_path)
        and _has_tokenizer_files(cache_path)
    )


def local_transformer_model_path(model_name: str, model_cache_dir: str = "./models") -> Path:
    """Return the deterministic local path used for a HuggingFace model id."""
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return model_path
    return Path(model_cache_dir).expanduser() / _safe_model_dir_name(model_name)


def ensure_transformer_model_cached(model_name: str, model_cache_dir: str = "./models") -> str:
    """Download model/tokenizer once and return a local path for future loads.

    If ``model_name`` is already a local path, it is returned unchanged.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    local_path = local_transformer_model_path(model_name, model_cache_dir)
    if is_transformer_model_cached(local_path):
        logger.info("Loading transformer model from local cache: %s", local_path)
        return str(local_path)

    local_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading transformer model %s to %s", model_name, local_path)

    from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    metadata = {"source_model": model_name}
    (local_path / "hmc_model_cache.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logger.info("Transformer model cached at %s", local_path)
    return str(local_path)

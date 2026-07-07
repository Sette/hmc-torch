"""Argparse + dataclass configuration for hmc-torch (ArXiv and WOS)."""

import argparse
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from hmc.datasets.registry import DatasetRegistry


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    dataset_path: str
    dataset_name: Optional[str] = None
    arxiv_model_name: str = "allenai/specter2_base"
    arxiv_max_records: int = 50_000
    model_cache_dir: str = "./models"


@dataclass
class TrainingConfig:
    """Training loop and optimization settings."""
    batch_size: int = 32
    non_lin: str = "relu"
    device: str = "cuda"
    epochs: int = 50
    seed: int = 0
    best_threshold: bool = False
    use_contrastive_loss: bool = False
    lambda_contrastive: float = 0.1
    lr_transformer: float = 2e-5
    llm_model: str = "qwen3:14b"
    llm_base_url: str = "http://localhost:11434"
    llm_num_ctx: int = 4096
    llm_top_k: int = 8
    llm_margin: float = 0.12
    llm_confidence_threshold: float = 0.60
    llm_max_tokens: int = 256
    llm_temperature: float = 0.0
    llm_only_uncertain: bool = True
    llm_k_max: int = 0
    llm_expand_hierarchy: bool = False
    llm_max_calls: int = 0
    llm_timeout: int = 120
    llm_cache: bool = True
    llm_cache_dir: str = ""
    llm_fallback_on_error: bool = True
    llm_preserve_scores: bool = True
    llm_max_document_chars: int = 6000


@dataclass
class Args:
    """Configuration for HMC model training (ArXiv and WOS datasets)."""

    dataset: DatasetConfig
    output_path: str
    registry: DatasetRegistry = field(default_factory=DatasetRegistry)
    job_id: str = "none"
    method: str = "global"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    results_path: str = "./results/"

    _DIRECT_FIELDS: ClassVar[frozenset] = frozenset({
        "dataset", "output_path", "registry", "job_id", "method",
        "training", "results_path",
    })

    def __getattr__(self, name: str):
        for sub in ("dataset", "training"):
            try:
                cfg = object.__getattribute__(self, sub)
            except AttributeError:
                continue
            if hasattr(cfg, name):
                return getattr(cfg, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in Args._DIRECT_FIELDS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        for sub in ("dataset", "training"):
            try:
                cfg = object.__getattribute__(self, sub)
            except AttributeError:
                continue
            if hasattr(cfg, name):
                setattr(cfg, name, value)
                return
        object.__setattr__(self, name, value)


def _str_to_bool(v: str) -> bool:
    return v.lower() == "true"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an HMC model.")

    parser.add_argument("--job_id", type=str, default="none")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--arxiv_model_name", type=str, default="allenai/specter2_base")
    parser.add_argument("--arxiv_max_records", type=int, default=50_000)
    parser.add_argument("--model_cache_dir", type=str, default="./models")
    parser.add_argument("--use_contrastive_loss", type=str, default="false",
                        choices=["true", "false"])
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    parser.add_argument("--lr_transformer", type=float, default=2e-5)
    parser.add_argument("--llm_model", type=str, default="qwen3:14b")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm_num_ctx", type=int, default=4096)
    parser.add_argument("--llm_top_k", type=int, default=8)
    parser.add_argument("--llm_margin", type=float, default=0.12)
    parser.add_argument("--llm_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--llm_max_tokens", type=int, default=256)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_only_uncertain", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--llm_k_max", type=int, default=0)
    parser.add_argument("--llm_expand_hierarchy", type=str, default="false",
                        choices=["true", "false"])
    parser.add_argument("--llm_max_calls", type=int, default=0)
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_cache", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--llm_cache_dir", type=str, default="")
    parser.add_argument("--llm_fallback_on_error", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--llm_preserve_scores", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--llm_max_document_chars", type=int, default=6000)
    parser.add_argument("--non_lin", type=str, default="relu",
                        choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="global",
                        choices=["global", "globalGNN", "globalE2E", "globalSOTA",
                                 "globalLLM", "globalLLMLite", "local", "localE2E"])
    parser.add_argument("--best_threshold", type=str, default="false",
                        choices=["true", "false"])
    parser.add_argument("--results_path", type=str, default="./results/")
    return parser


def parse_args() -> Args:
    ns = get_parser().parse_args()
    dataset = DatasetConfig(
        dataset_path=ns.dataset_path,
        dataset_name=ns.dataset_name,
        arxiv_model_name=ns.arxiv_model_name,
        arxiv_max_records=ns.arxiv_max_records,
        model_cache_dir=ns.model_cache_dir,
    )
    training = TrainingConfig(
        batch_size=ns.batch_size,
        non_lin=ns.non_lin,
        device=ns.device,
        epochs=ns.epochs,
        seed=ns.seed,
        best_threshold=_str_to_bool(ns.best_threshold),
        use_contrastive_loss=_str_to_bool(ns.use_contrastive_loss),
        lambda_contrastive=ns.lambda_contrastive,
        lr_transformer=ns.lr_transformer,
        llm_model=ns.llm_model,
        llm_base_url=ns.llm_base_url,
        llm_num_ctx=ns.llm_num_ctx,
        llm_top_k=ns.llm_top_k,
        llm_margin=ns.llm_margin,
        llm_confidence_threshold=ns.llm_confidence_threshold,
        llm_max_tokens=ns.llm_max_tokens,
        llm_temperature=ns.llm_temperature,
        llm_only_uncertain=_str_to_bool(ns.llm_only_uncertain),
        llm_k_max=ns.llm_k_max,
        llm_expand_hierarchy=_str_to_bool(ns.llm_expand_hierarchy),
        llm_max_calls=ns.llm_max_calls,
        llm_timeout=ns.llm_timeout,
        llm_cache=_str_to_bool(ns.llm_cache),
        llm_cache_dir=ns.llm_cache_dir,
        llm_fallback_on_error=_str_to_bool(ns.llm_fallback_on_error),
        llm_preserve_scores=_str_to_bool(ns.llm_preserve_scores),
        llm_max_document_chars=ns.llm_max_document_chars,
    )
    return Args(
        dataset=dataset,
        output_path=ns.output_path,
        job_id=ns.job_id,
        method=ns.method,
        training=training,
    )

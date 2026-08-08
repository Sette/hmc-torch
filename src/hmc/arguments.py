"""Argparse + dataclass configuration for hmc-torch (ArXiv and WOS)."""

import argparse
from dataclasses import dataclass, field
from typing import ClassVar

from hmc.datasets.registry import DatasetRegistry


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""

    dataset_path: str
    dataset_name: str | None = None
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

    _DIRECT_FIELDS: ClassVar[frozenset] = frozenset(
        {
            "dataset",
            "output_path",
            "registry",
            "job_id",
            "method",
            "training",
            "results_path",
        }
    )

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
    parser.add_argument(
        "--use_contrastive_loss", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    parser.add_argument(
        "--non_lin", type=str, default="relu", choices=["relu", "tanh", "sigmoid"]
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        type=str,
        default="global",
        choices=[
            "global",
            "globalGNN",
            "globalE2E",
            "globalSOTA",
            "local",
            "localE2E",
            "tabular_gbdt",
            "tabular_mlp",
        ],
    )
    parser.add_argument(
        "--best_threshold", type=str, default="false", choices=["true", "false"]
    )
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
    )
    return Args(
        dataset=dataset,
        output_path=ns.output_path,
        job_id=ns.job_id,
        method=ns.method,
        training=training,
    )

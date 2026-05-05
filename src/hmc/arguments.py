"""
This module defines the Args dataclass for configuring and launching
the training and hyperparameter optimization of a Hierarchical Multi-label
Classification (HMC) model.
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from hmc.datasets.registry import DatasetRegistry


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""

    dataset_path: str
    dataset_name: Optional[str] = None
    use_sample: bool = False
    save_torch_dataset: bool = True
    dataset_type: str = "arff"


@dataclass
class Args:
    """Configuration for HMC model training and hyperparameter optimization."""

    # Required
    dataset: DatasetConfig
    output_path: str

    # Dataset registry (static lookup tables — dimensions, default lrs, epochs)
    registry: DatasetRegistry = field(default_factory=DatasetRegistry)

    # Job identification
    job_id: str = "none"

    # Training
    batch_size: int = 64
    non_lin: str = "relu"
    device: str = "cpu"
    epochs: int = 2000
    epochs_attention: int = 100
    epochs_level: int = 2000
    seed: int = 0
    method: str = "global"
    focal_loss: bool = False
    warmup: bool = False
    n_warmup_epochs: int = 50
    n_warmup_epochs_increment: int = 50
    parent_conditioning: str = "false"
    early_metric: str = "avg-score"
    predict_test: bool = True
    level_model_type: str = "mlp"
    active_levels: Optional[list[int]] = None
    encoder_block: bool = False
    patience: int = 5
    patience_score: int = 20
    epochs_to_evaluate: int = 20
    epochs_to_test: int = 20

    # Threshold
    best_threshold: bool = True

    # HPO
    hpo: bool = False
    hpo_by_level: bool = True
    n_trials: Optional[int] = None

    # Hyperparameters (used when HPO is disabled)
    lr_values: Optional[list[float]] = None
    dropout_values: Optional[list[float]] = None
    hidden_dims: Optional[Any] = None
    num_layers_values: Optional[list[int]] = None
    weight_decay_values: Optional[list[float]] = None

    # Paths
    results_path: str = "./results/"


def _str_to_bool(v: str) -> bool:
    return v.lower() == "true"


def get_parser() -> argparse.ArgumentParser:
    """
    Create and return an argument parser for the HMC model.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Multi-label Classification model."
    )

    parser.add_argument(
        "--job_id",
        type=str,
        default="none",
        required=False,
        help="Job id for trainer job.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default=None,
        help="Dataset name to be used.",
    )

    parser.add_argument(
        "--use_sample",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="USE_SAMPLE",
        required=False,
        help="Enable or disable to use a sample of data (for tests). \
                Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--save_torch_dataset",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="USE_SAMPLE",
        required=False,
        help="Enable or disable to use save torch dataset. \
                    Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to data and metadata files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save models.",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        required=False,
        help="n_trials for hpo.",
    )

    parser.add_argument(
        "--best_threshold",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="best_threshold",
        required=False,
        help="Enable or disable to use find the best thesholds. \
                        Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        required=False,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["csv", "torch", "arff"],
        default="arff",
        metavar="DATASET_TYPE",
        required=False,
        help="Type of dataset to load.",
    )

    parser.add_argument(
        "--non_lin",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
        metavar="NON_LIN",
        required=False,
        help="Non-linearity function.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        metavar="DEVICE",
        required=False,
        help='Device to use (e.g., "cpu" or "cuda").',
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        metavar="EPOCHS",
        required=False,
        help="Total number of training epochs.",
    )

    parser.add_argument(
        "--epochs_attention",
        type=int,
        default=100,
        metavar="EPOCHS_ATTENTION",
        required=False,
        help="Total number of training epochs for attention.",
    )

    parser.add_argument(
        "--epochs_level",
        type=int,
        default=2000,
        metavar="EPOCHS_LEVEL",
        required=False,
        help="Total number of training epochs for level.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="global",
        choices=[
            "global",
            "local",
            "globalLM",
            "global_baseline",
            "local_constraint",
            "local_hat",
            "local_tabat",
            "local_test",
        ],
        metavar="METHOD",
        required=False,
        help="Method type to use.",
    )

    parser.add_argument(
        "--focal_loss",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="FOCAL_LOSS",
        required=False,
        help="Enable or disable Focal Loss. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--warmup",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="WARMUP",
        required=False,
        help="Enable or disable learning rate warmup. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--n_warmup_epochs",
        type=int,
        default=50,
        required=False,
        metavar="N_WARMUP_EPOCHS",
    )

    parser.add_argument(
        "--n_warmup_epochs_increment",
        type=int,
        default=50,
        required=False,
        metavar="N_WARMUP_EPOCHS_INCREMENT",
    )

    parser.add_argument(
        "--hpo",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="HPO",
        required=False,
        help="Enable or disable Hyperparameter Optimization (HPO). \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--hpo_by_level",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="HPO_BY_LEVEL",
        required=False,
        help="Enable or disable HPO by level. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--parent_conditioning",
        type=str,
        default="false",
        choices=["residual", "soft", "teacher_forcing", "none"],
        metavar="PARENT_CONDITIONING",
        required=False,
        help="Select or disable parent conditioning. \
            Use 'residual' or 'soft' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/",
        metavar="RESULTS_PATH",
        required=False,
        help="Path to save results.",
    )

    parser.add_argument(
        "--early_metric",
        type=str,
        default="avg-score",
        choices=["f1-score", "avg-score"],
        metavar="EARLY_METRIC",
        required=False,
        help="Metric to use for early stopping.",
    )

    parser.add_argument(
        "--predict_test",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="PREDICT_TEST",
        required=False,
        help="Enable or disable prediction on test set after training. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--level_model_type",
        type=str,
        default="mlp",
        choices=["mlp", "attention", "gcn", "gat"],
        metavar="LEVEL_MODEL_TYPE",
        required=False,
        help="Specific model type to use at each level.",
    )

    parser.add_argument(
        "--active_levels",
        type=int,
        nargs="+",
        default=None,
        required=False,
        metavar="ACTIVE_LEVELS",
    )

    parser.add_argument(
        "--lr_values",
        type=float,
        nargs="+",
        required=False,
        help="List of values for the learning rate (used when HPO is disabled).",
    )

    parser.add_argument(
        "--dropout_values",
        type=float,
        nargs="+",
        required=False,
        metavar="DROPOUT",
        help="List of values for dropout (used when HPO is disabled).",
    )

    parser.add_argument(
        "--hidden_dims",
        type=json.loads,
        required=False,
        metavar="HIDDEN_DIMS",
        help="List (or list of lists) of hidden neurons. "
        "Can be passed as JSON when HPO is enabled (e.g. '[[128,64],[256]]').",
    )

    parser.add_argument(
        "--num_layers_values",
        type=int,
        nargs="+",
        required=False,
        metavar="NUM_LAYERS",
        help="List of values for the number of layers (used when HPO is disabled).",
    )

    parser.add_argument(
        "--weight_decay_values",
        type=float,
        nargs="+",
        required=False,
        metavar="WEIGHT_DECAY",
        help="List of values for weight decay (used when HPO is disabled).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        metavar="PATIENCE",
        required=False,
        help="Number of epochs with no improvement after which training will be stopped.",
    )

    parser.add_argument(
        "--encoder_block",
        type=bool,
        default=False,
        metavar="ENCODER_BLOCK",
        required=False,
        help="Active Encoder Block in the model.",
    )

    parser.add_argument(
        "--patience_score",
        type=int,
        default=20,
        metavar="PATIENCE_SCORE",
        required=False,
        help="Number of epochs with no improvement after which training will be stopped.",
    )

    parser.add_argument(
        "--epochs_to_evaluate",
        type=int,
        default=20,
        metavar="EPOCHS_TO_EVALUATE",
        required=False,
        help="Number of epochs to evaluate the model during training.",
    )

    parser.add_argument(
        "--epochs_to_test",
        type=int,
        default=20,
        metavar="EPOCHS_TO_TEST",
        required=False,
        help="Number of epochs to test the model during training.",
    )

    return parser


def parse_args() -> Args:
    """Parse CLI arguments and return a typed Args dataclass."""
    ns = get_parser().parse_args()
    dataset = DatasetConfig(
        dataset_path=ns.dataset_path,
        dataset_name=ns.dataset_name,
        use_sample=_str_to_bool(ns.use_sample),
        save_torch_dataset=_str_to_bool(ns.save_torch_dataset),
        dataset_type=ns.dataset_type,
    )
    return Args(
        dataset=dataset,
        output_path=ns.output_path,
        job_id=ns.job_id,
        n_trials=ns.n_trials,
        best_threshold=_str_to_bool(ns.best_threshold),
        batch_size=ns.batch_size,
        non_lin=ns.non_lin,
        device=ns.device,
        epochs=ns.epochs,
        epochs_attention=ns.epochs_attention,
        epochs_level=ns.epochs_level,
        seed=ns.seed,
        method=ns.method,
        focal_loss=_str_to_bool(ns.focal_loss),
        warmup=_str_to_bool(ns.warmup),
        n_warmup_epochs=ns.n_warmup_epochs,
        n_warmup_epochs_increment=ns.n_warmup_epochs_increment,
        hpo=_str_to_bool(ns.hpo),
        hpo_by_level=_str_to_bool(ns.hpo_by_level),
        parent_conditioning=ns.parent_conditioning,
        results_path=ns.results_path,
        early_metric=ns.early_metric,
        predict_test=_str_to_bool(ns.predict_test),
        level_model_type=ns.level_model_type,
        active_levels=ns.active_levels,
        lr_values=ns.lr_values,
        dropout_values=ns.dropout_values,
        hidden_dims=ns.hidden_dims,
        num_layers_values=ns.num_layers_values,
        weight_decay_values=ns.weight_decay_values,
        patience=ns.patience,
        encoder_block=ns.encoder_block,
        patience_score=ns.patience_score,
        epochs_to_evaluate=ns.epochs_to_evaluate,
        epochs_to_test=ns.epochs_to_test,
    )

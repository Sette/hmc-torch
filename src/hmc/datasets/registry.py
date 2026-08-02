"""
Static registry of dataset dimensions and training hyperparameters for all
supported HMC benchmarks.
"""

from dataclasses import dataclass, field


@dataclass
class DatasetRegistry:
    """Lookup tables with per-dataset input/output dimensions and default hyperparameters."""

    input_dims: dict[str, int] = field(
        default_factory=lambda: {
            "diatoms": 371,
            "enron": 1001,
            "imclef07a": 80,
            "imclef07d": 80,
            "cellcycle": 77,
            "derisi": 63,
            "eisen": 79,
            "expr": 561,
            "gasch1": 173,
            "gasch2": 52,
            "seq": 529,
            "spo": 86,
            "arxiv": 256,
        }
    )

    arxiv_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 50,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )

    output_dims: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 499,
                "derisi": 499,
                "eisen": 461,
                "expr": 499,
                "gasch1": 499,
                "gasch2": 499,
                "seq": 499,
                "spo": 499,
            },
            "GO": {
                "cellcycle": 4122,
                "derisi": 4116,
                "eisen": 3570,
                "expr": 4128,
                "gasch1": 4122,
                "gasch2": 4128,
                "seq": 4130,
                "spo": 4116,
            },
            "others": {
                "diatoms": 398,
                "enron": 56,
                "imclef07a": 96,
                "imclef07d": 46,
                "reuters": 102,
            },
        }
    )

    hidden_dims: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 500,
                "derisi": 500,
                "eisen": 500,
                "expr": 1250,
                "gasch1": 1000,
                "gasch2": 500,
                "seq": 2000,
                "spo": 250,
            },
            "GO": {
                "cellcycle": 1000,
                "derisi": 500,
                "eisen": 500,
                "expr": 4000,
                "gasch1": 500,
                "gasch2": 500,
                "seq": 9000,
                "spo": 500,
            },
            "others": {
                "diatoms": 2000,
                "enron": 1000,
                "imclef07a": 1000,
                "imclef07d": 1000,
            },
        }
    )

    lrs: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 1e-4,
                "derisi": 1e-4,
                "eisen": 1e-4,
                "expr": 1e-4,
                "gasch1": 1e-4,
                "gasch2": 1e-4,
                "seq": 1e-4,
                "spo": 1e-4,
            },
            "GO": {
                "cellcycle": 1e-4,
                "derisi": 1e-4,
                "eisen": 1e-4,
                "expr": 1e-4,
                "gasch1": 1e-4,
                "gasch2": 1e-4,
                "seq": 1e-4,
                "spo": 1e-4,
            },
            "others": {
                "diatoms": 1e-5,
                "enron": 1e-5,
                "imclef07a": 1e-5,
                "imclef07d": 1e-5,
            },
        }
    )

    all_epochs: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 106,
                "derisi": 67,
                "eisen": 110,
                "expr": 20,
                "gasch1": 42,
                "gasch2": 123,
                "seq": 13,
                "spo": 115,
            },
            "GO": {
                "cellcycle": 62,
                "derisi": 91,
                "eisen": 123,
                "expr": 70,
                "gasch1": 122,
                "gasch2": 177,
                "seq": 45,
                "spo": 103,
            },
            "others": {
                "diatoms": 474,
                "enron": 133,
                "imclef07a": 592,
                "imclef07d": 588,
            },
        }
    )

"""
Static registry of default hyperparameters for supported HMC datasets.
"""

from dataclasses import dataclass, field


@dataclass
class DatasetRegistry:
    """Lookup tables with per-dataset default hyperparameters."""

    input_dims: dict[str, int] = field(
        default_factory=lambda: {
            "arxiv": 768,
            "wos": 768,
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

    wos_defaults: dict = field(
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

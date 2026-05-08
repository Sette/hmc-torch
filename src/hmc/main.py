"""
Main module for training and hyperparameter optimization of the HMC model.

This module orchestrates the entire training pipeline, handling argument parsing,
configuration, dataset loading, model training, and evaluation. It serves as the
entry point for running experiments with different methods and configurations.
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from hmc.arguments import parse_args
from hmc.pipeline.global_classifier.main import train_global
from hmc.pipeline.local_classifier.main import main_local
from hmc.utils.train.job import create_job_id_name

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> dict:
    """
    Main training function (entrypoint).

    Returns:
        dict: Score dictionary with keys such as ``"f1score"``, ``"precision"``,
        ``"recall"``, and ``"avg_precision"``.
    """
    # Training settings
    args = parse_args()
    print(f"Learning rates: {args.lr_values}")
    args.score = 0.0

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verifica quantas GPUs estão disponíveis
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")

    if args.job_id == "none":
        args.job_id = create_job_id_name()
        print(f"Job ID: {args.job_id}")

    args.results_path = os.path.join(
        args.output_path,
        "train",
        "local",
        args.dataset.dataset_name,
        args.job_id,
    )

    match args.method:
        case "local" | "local_tabat" | "local_hat" | "local_test":
            logging.info("Local method selected")
            main_local(args)
        case "global" | "global_baseline":
            logging.info("Global method selected")
            train_global(args.dataset.dataset_name, args)
        case _:
            print("Invalid option for method. Please select a valid method.")

    score: dict = args.score if isinstance(args.score, dict) else {}
    return score


if __name__ == "__main__":
    main()

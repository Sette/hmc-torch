"""
Main module for training HMC models on ArXiv and WOS datasets.

Supports methods: global (frozen), globalE2E (fine-tuned), globalSOTA (E2E + GCN),
and globalLLM/globalLLMLite (E2E + LLM reranking).
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from hmc.arguments import parse_args
from hmc.pipeline.global_classifier.main import (
    train_global,
    train_global_e2e,
    train_global_llm,
    train_global_llm_lite,
    train_global_sota,
)
from hmc.pipeline.local_classifier.main import train_local, train_local_e2e
from hmc.utils.train.job import create_job_id_name

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> dict:
    """Main training function (entrypoint)."""
    args = parse_args()
    args.score = 0.0

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")

    if args.job_id == "none":
        args.job_id = create_job_id_name()
        print(f"Job ID: {args.job_id}")

    args.results_path = os.path.join(
        args.output_path,
        "train",
        args.method,
        args.dataset.dataset_name,
        args.job_id,
    )

    # GoFun ARFF datasets only support local classifier for now
    _is_gofun = any(suffix in (args.dataset.dataset_name or "")
                    for suffix in ("_FUN", "_GO", "_others"))

    match args.method:
        case "global" | "global_baseline" | "globalGNN" | "globalLM":
            logging.info("Global classifier (frozen embeddings)")
            train_global(args.dataset.dataset_name, args)
        case "globalE2E":
            logging.info("Global E2E (fine-tuned transformer)")
            train_global_e2e(args.dataset.dataset_name, args)
        case "globalSOTA":
            logging.info("Global SOTA (transformer + label GCN)")
            train_global_sota(args.dataset.dataset_name, args)
        case "globalLLM":
            logging.info("Global LLM reranker (E2E + LLM)")
            train_global_llm(args.dataset.dataset_name, args)
        case "globalLLMLite":
            logging.info("Global LLM lite reranker (cheaper LLM gate)")
            train_global_llm_lite(args.dataset.dataset_name, args)
        case "local":
            logging.info("Local classifier (frozen, one MLP per level)")
            train_local(args.dataset.dataset_name, args)
        case "localE2E":
            logging.info("Local E2E (fine-tuned transformer + per-level MLPs)")
            train_local_e2e(args.dataset.dataset_name, args)
        case "tabular_gbdt":
            logging.info("Tabular GBDT One-vs-Rest baseline")
            from hmc.pipeline.tabular.main import train_gbdt
            train_gbdt(args.dataset.dataset_name, args)
        case "tabular_mlp":
            logging.info("Tabular Residual MLP baseline")
            from hmc.pipeline.tabular.main import train_tabular_mlp
            train_tabular_mlp(args.dataset.dataset_name, args)
        case _:
            print(
                f"Unknown method '{args.method}'. "
                "Valid: global, globalE2E, globalSOTA, globalLLM, globalLLMLite, local, localE2E"
            )

    score: dict = args.score if isinstance(args.score, dict) else {}
    return score


if __name__ == "__main__":
    main()

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from hmc.model.arguments import get_parser
from hmc.train.train_global import train_global
from hmc.train.train_globalLM import train_globalLM
from hmc.train.train_global_baseline import train_global_baseline
from hmc.train.train_local import train_local
from hmc.utils.dir import create_job_id

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    # Training settings
    parser = get_parser()
    args = parser.parse_args()

    # Insert her a logic to use all datasets with arguments

    if "all" in args.datasets:
        datasets = [
            "cellcycle_GO",
            "derisi_GO",
            "eisen_GO",
            "expr_GO",
            "gasch1_GO",
            "gasch2_GO",
            "seq_GO",
            "spo_GO",
            "cellcycle_FUN",
            "derisi_FUN",
            "eisen_FUN",
            "expr_FUN",
            "gasch1_FUN",
            "gasch2_FUN",
            "seq_FUN",
            "spo_FUN",
        ]
    else:
        if len(args.datasets) > 1:
            datasets = [str(dataset) for dataset in args.datasets]
        else:
            datasets = args.datasets

    # Dictionaries with number of features and number of labels for each dataset
    args.input_dims = {
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
    }
    output_dims_FUN = {
        "cellcycle": 499,
        "derisi": 499,
        "eisen": 461,
        "expr": 499,
        "gasch1": 499,
        "gasch2": 499,
        "seq": 499,
        "spo": 499,
    }
    output_dims_GO = {
        "cellcycle": 4122,
        "derisi": 4116,
        "eisen": 3570,
        "expr": 4128,
        "gasch1": 4122,
        "gasch2": 4128,
        "seq": 4130,
        "spo": 4116,
    }
    output_dims_others = {
        "diatoms": 398,
        "enron": 56,
        "imclef07a": 96,
        "imclef07d": 46,
        "reuters": 102,
    }
    args.output_dims = {
        "FUN": output_dims_FUN,
        "GO": output_dims_GO,
        "others": output_dims_others,
    }

    # Dictionaries with number of features and number of labels for each dataset
    hidden_dims_FUN = {
        "cellcycle": 500,
        "derisi": 500,
        "eisen": 500,
        "expr": 1250,
        "gasch1": 1000,
        "gasch2": 500,
        "seq": 2000,
        "spo": 250,
    }
    hidden_dims_GO = {
        "cellcycle": 1000,
        "derisi": 500,
        "eisen": 500,
        "expr": 4000,
        "gasch1": 500,
        "gasch2": 500,
        "seq": 9000,
        "spo": 500,
    }
    hidden_dims_others = {
        "diatoms": 2000,
        "enron": 1000,
        "imclef07a": 1000,
        "imclef07d": 1000,
    }
    args.hidden_dims = {
        "FUN": hidden_dims_FUN,
        "GO": hidden_dims_GO,
        "others": hidden_dims_others,
    }
    lrs_FUN = {
        "cellcycle": 1e-4,
        "derisi": 1e-4,
        "eisen": 1e-4,
        "expr": 1e-4,
        "gasch1": 1e-4,
        "gasch2": 1e-4,
        "seq": 1e-4,
        "spo": 1e-4,
    }
    lrs_GO = {
        "cellcycle": 1e-4,
        "derisi": 1e-4,
        "eisen": 1e-4,
        "expr": 1e-4,
        "gasch1": 1e-4,
        "gasch2": 1e-4,
        "seq": 1e-4,
        "spo": 1e-4,
    }
    lrs_others = {"diatoms": 1e-5, "enron": 1e-5, "imclef07a": 1e-5, "imclef07d": 1e-5}
    args.lrs = {"FUN": lrs_FUN, "GO": lrs_GO, "others": lrs_others}
    epochss_FUN = {
        "cellcycle": 106,
        "derisi": 67,
        "eisen": 110,
        "expr": 20,
        "gasch1": 42,
        "gasch2": 123,
        "seq": 13,
        "spo": 115,
    }
    epochss_GO = {
        "cellcycle": 62,
        "derisi": 91,
        "eisen": 123,
        "expr": 70,
        "gasch1": 122,
        "gasch2": 177,
        "seq": 45,
        "spo": 103,
    }
    epochss_others = {"diatoms": 474, "enron": 133, "imclef07a": 592, "imclef07d": 588}
    args.epochss = {"FUN": epochss_FUN, "GO": epochss_GO, "others": epochss_others}

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

    args.job_id = create_job_id()
    print(f"Job ID: {args.job_id}")

    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dataset_name in datasets:
        args.dataset_name = dataset_name
        if args.method == "global":
            train_global(dataset_name, args)

        if args.method == "local":
            train_local(args)

        if args.method == "globalLM":
            train_globalLM(dataset_name, args)

        if args.method == "global_baseline":
            train_global_baseline(dataset_name, args)


if __name__ == "__main__":
    main()

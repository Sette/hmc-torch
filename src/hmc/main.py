import os
import random
import sys
from pathlib import Path
import logging
import numpy as np
import torch

from hmc.arguments import get_parser
from hmc.trainers.global_classifier.constrained.train_global import train_global

from hmc.utils.dir import create_job_id
from hmc.utils.dir import create_dir

import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.trainers.local_classifier.core.hpo.hpo_local_level import (
    optimize_hyperparameters,
)

from hmc.trainers.local_classifier.core.test_local import (
    test_step as test_step_core,
)

from hmc.trainers.local_classifier.core.train_local import (
    train_step as train_step_core,
)

# Import necessary modules for constrained training local classifiers
from hmc.models.local_classifier.constrained.model import ConstrainedHMCLocalModel


# Import necessary modules for training baseline local classifiers
from hmc.models.local_classifier.baseline.model import HMCLocalModel


# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments


def get_train_methods(x):
    match x:
        case "local_constrained":
            return {
                "model": ConstrainedHMCLocalModel,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step_core,
                "train_step": train_step_core,
            }
        case "local":
            return {
                "model": HMCLocalModel,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step_core,
                "train_step": train_step_core,
            }
        case "local_mask":
            return {
                "model": HMCLocalModel,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step_core,
                "train_step": train_step_core,
            }
        case "local_test":
            return {
                "model": HMCLocalModel,
                "test_step": test_step_core,
            }
        case "global":
            return {
                "model": ConstrainedHMCLocalModel,
            }
        case _:
            raise ValueError(f"Método '{x}' não reconhecido.")


def assert_hyperparameter_lengths(
    args, lr_values, dropout_values, hidden_dims, num_layers_values, weight_decay_values
):
    checks = {
        "lr_values": lr_values,
        "dropout_values": dropout_values,
        "hidden_dims": hidden_dims,
        "num_layers_values": num_layers_values,
        "weight_decay_values": weight_decay_values,
    }
    all_passed = True
    for name, lst in checks.items():
        try:
            assert (
                len(lst) == args.max_depth
            ), f"{name} length {len(lst)} != max_depth {args.max_depth}"
        except AssertionError as e:
            print(f"Assert failed: {e}")
            all_passed = False

    if all_passed:
        print("All hyperparameter lists have the correct length.")
    else:
        print("One or more hyperparameter lists have the wrong length.")


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
    if not args.hidden_dims:
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

    if args.job_id == "false":
        args.job_id = create_job_id()
        logging.info(f"Job ID created: {args.job_id}")
    else:
        logging.info(f"Using Job ID: {args.job_id}")

    if args.use_sample == "true":
        args.use_sample = True
    else:
        args.use_sample = False

    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dataset_name in datasets:
        args.dataset_name = dataset_name

        args.results_path = (
            f"{args.output_path}/train/local/{args.dataset_name}/{args.job_id}"
        )

        logging.info(".......................................")
        logging.info("Experiment with %s dataset", args.dataset_name)

        args.train_methods = get_train_methods(args.method)

        if args.method == "local_constrained":
            logging.info("Using constrained local model")

        args.model_regularization = None

        # Load train, val and test set

        if not torch.cuda.is_available():
            print("CUDA não está disponível. Usando CPU.")
            args.device = torch.device("cpu")
        else:
            args.device = torch.device(args.device)

        args.data, args.ontology = args.dataset_name.split("_")

        create_dir(args.results_path)

        # Caminhos
        train_path = os.path.join(args.results_path, "train_dataset.pt")
        val_path = os.path.join(args.results_path, "val_dataset.pt")
        test_path = os.path.join(args.results_path, "test_dataset.pt")

        read_data = True

        # Se já existir, carrega. Se não, cria e salva
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            print("Loading existing datasets...")
            read_data = False
            args.hmc_dataset = initialize_dataset_experiments(
                args.dataset_name,
                device=args.device,
                dataset_path=args.dataset_path,
                dataset_type=args.dataset_type,
                is_global=False,
                read_data=read_data,
                use_sample=args.use_sample,
            )
            train_dataset = torch.load(train_path, weights_only=False)
            val_dataset = torch.load(val_path, weights_only=False)
            test_dataset = torch.load(test_path, weights_only=False)
        else:
            args.hmc_dataset = initialize_dataset_experiments(
                args.dataset_name,
                device=args.device,
                dataset_path=args.dataset_path,
                dataset_type=args.dataset_type,
                is_global=False,
                read_data=read_data,
                use_sample=args.use_sample,
            )
            data_train, data_valid, data_test = args.hmc_dataset.get_datasets()

            scaler = preprocessing.StandardScaler().fit(
                np.concatenate((data_train.X, data_valid.X))
            )
            imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
                np.concatenate((data_train.X, data_valid.X))
            )
            data_valid.X = (
                torch.tensor(scaler.transform(imp_mean.transform(data_valid.X)))
                .clone()
                .detach()
                .to(args.device)
            )
            data_train.X = (
                torch.tensor(scaler.transform(imp_mean.transform(data_train.X)))
                .clone()
                .detach()
                .to(args.device)
            )

            data_test.X = (
                torch.as_tensor(scaler.transform(imp_mean.transform(data_test.X)))
                .clone()
                .detach()
                .to(args.device)
            )

            data_test.Y = torch.as_tensor(data_test.Y).clone().detach().to(args.device)
            data_valid.Y = torch.tensor(data_valid.Y).clone().detach().to(args.device)
            data_train.Y = torch.tensor(data_train.Y).clone().detach().to(args.device)

            # Create loaders using local (per-level) y labels
            train_dataset = [
                (x, y_levels, y)
                for (x, y_levels, y) in zip(
                    data_train.X, data_train.Y_local, data_train.Y
                )
            ]

            val_dataset = [
                (x, y_levels, y)
                for (x, y_levels, y) in zip(
                    data_valid.X, data_valid.Y_local, data_valid.Y
                )
            ]

            test_dataset = [
                (x, y_levels, y)
                for (x, y_levels, y) in zip(data_test.X, data_test.Y_local, data_test.Y)
            ]

            if args.save_torch_dataset:
                # Save datasets in torch format
                torch.save(train_dataset, train_path)
                torch.save(val_dataset, val_path)
                torch.save(test_dataset, test_path)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False
        )
        args.train_loader = train_loader
        args.val_loader = val_loader
        args.test_loader = test_loader
        args.levels_size = args.hmc_dataset.levels_size
        args.input_dim = args.input_dims[args.data]
        args.max_depth = args.hmc_dataset.max_depth
        args.to_eval = args.hmc_dataset.to_eval
        args.constrained = True
        logging.info("Active levels before processing: %s", args.active_levels)
        if args.active_levels is None:
            args.active_levels = list(range(args.max_depth))
        else:
            args.active_levels = [int(x) for x in args.active_levels]

        criterions = [nn.BCELoss() for _ in args.hmc_dataset.levels_size]
        args.criterions = criterions

        if args.hpo == "true":
            logging.info("Hyperparameter optimization")
            # args.n_trials = 30
            best_params = args.train_methods["optimize_hyperparameters"](args=args)

            logging.info(best_params)
        else:
            if args.lr_values:
                args.lr_values = [float(x) for x in args.lr_values]
                args.dropout_values = [float(x) for x in args.dropout_values]
                # args.hidden_dims = [int(x) for x in args.hidden_dims]
                args.num_layers_values = [int(x) for x in args.num_layers_values]
                args.weight_decay_values = [float(x) for x in args.weight_decay_values]

                # Ensure all hyperparameter lists have the same length as 'max_depth'
                assert_hyperparameter_lengths(
                    args,
                    args.lr_values,
                    args.dropout_values,
                    args.hidden_dims,
                    args.num_layers_values,
                    args.weight_decay_values,
                )
                params = {
                    "levels_size": args.hmc_dataset.levels_size,
                    "input_size": args.input_dims[args.data],
                    "hidden_dims": args.hidden_dims,
                    "num_layers": args.num_layers_values,
                    "dropout": args.dropout_values,
                    "active_levels": args.active_levels,
                }

                if args.method == "local_constrained":
                    params["all_matrix_r"] = args.hmc_dataset.all_matrix_r

                args.model = args.train_methods["model"](**params)
                logging.info(args.model)

        match args.method:
            case "local" | "local_constrained" | "local_mask":
                logging.info("Local method selected")
                args.train_methods["train_step"](args)
                args.train_methods["test_step"](args)
                # train(args)
            case "global" | "global_baseline":
                logging.info("Global method selected")
                train_global(dataset_name, args)
            case "local_test":

                logging.info("Test local method selected")

                args.results_path = (
                    f"{args.output_path}/train/local/{args.dataset_name}/{args.job_id}"
                )

                args.train_methods["test_step"](args)
            case _:  # Default case (like 'default' in other languages
                print("Invalid option")


if __name__ == "__main__":
    main()

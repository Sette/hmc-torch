"""
Module for training local hierarchical multi-label classifiers.

This module provides utility functions and methods to train, evaluate, and perform
hyperparameter optimization on local-level neural network classifiers for hierarchical
multi-label classification (HMC) tasks. It supports handling data loading, preprocessing,
and experiment setup for local and constrained local classifiers. Hyperparameter lists
are validated for correct configuration according to the levels of the hierarchy.

Main functionality:
    - Selects and initializes the appropriate classifier and associated train/test methods.
    - Loads, normalizes, and imputes missing values in train, validation, and test datasets.
    - Constructs per-level data loaders for efficient training and evaluation.
    - Supports hyperparameter optimization (HPO) and manual hyperparameter configuration.
    - Provides robust reproducibility through controlled random seed settings.
    - Validates configuration consistency per hierarchical level.

Classes and functions:
    - get_train_methods: Returns method mappings based on the classifier type.
    - assert_hyperparameter_lengths: Validates hyperparameter list lengths per hierarchy level.
    - train_local: Main training routine for local (and constrained local) HMC classifiers.

Dependencies:
    - torch, numpy, sklearn, hmc.utils, hmc.datasets, hmc.models, hmc.pipeline

Intended for research and experimentation with HMC classifier benchmarks.

Authors: Bruno Sette
"""

import logging
import os
import time
from collections.abc import Sequence

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader


from hmc.models.local_classifier.baseline.model import HMCLocalModel
from hmc.pipeline.local_classifier.core.train import train_step
from hmc.pipeline.local_classifier.core.validate import validate_step
from hmc.pipeline.local_classifier.hpo.hpo_local import optimize_hyperparameters
from hmc.utils.path.files import create_dir
from hmc.utils.train.job import log_system_info
from hmc.pipeline.local_classifier.core.predict import test_step


def get_train_methods(method: str) -> dict[str, object]:
    """
    Given a local classifier method string, returns a mapping of train/test/model/HPO methods.

    Args:
        method (str): Type of local classifier. Options:
            - "local_constrained": Constrained local classifier (per level, constraints enforced)
            - "local": Standard per-level classifier
            - "local_mask": Standard per-level classifier with mask variant

    Returns:
        dict: Dictionary with keys "model", "optimize_hyperparameters",
            "test_step", and "train_step" mapping to the appropriate
            functions or classes.

    Raises:
        ValueError: If an unknown method string is provided.
    """
    model_functions: dict[str, object] = {}
    match method:
        case "local" | "local_mask" | "local_test":
            model_functions = {
                "model": HMCLocalModel,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step,
                "valid_step": validate_step,
                "train_step": train_step,
            }
        case _:
            raise ValueError(f"Método {method} não reconhecido.")

    return model_functions


def assert_hyperparameter_lengths(
    args: object,
) -> None:
    """
    Validates that all hyperparameter lists have a length equal to the
        maximum depth of the hierarchy.

    Args:
        args: Arguments object containing max_depth and relevant
            experiment settings.
        lr_values (list): List of learning rates per level.
        dropout_values (list): List of dropout rates per level.
        hidden_dims (list): List of hidden layer sizes per level.
        num_layers_values (list): List of number of layers per level.
        weight_decay_values (list): List of weight decay values per level.

    Side Effects:
        - Prints assertion results to stdout.

    Raises:
        AssertionError: If any list does not have a length equal to args.max_depth.
    """
    checks: dict[str, Sequence[int | float]] = {
        "lr_values": args.hyperparameters["lr_values"],
        "dropout_values": args.hyperparameters["dropout_values"],
        "hidden_dims": args.hyperparameters["hidden_dims"],
        "num_layers_values": args.hyperparameters["num_layers_values"],
        "weight_decay_values": args.hyperparameters["weight_decay_values"],
    }
    all_passed = True
    for name, lst in checks.items():
        try:
            assert len(lst) == args.max_depth, (
                f"{name} length {len(lst)} != max_depth {args.max_depth}"
            )
        except AssertionError as e:
            print(f"Assert failed: {e}")
            all_passed = False

    if all_passed:
        print("All hyperparameter lists have the correct length.")
    else:
        print("One or more hyperparameter lists have the wrong length.")


def create_dataloader(
    data,
    scaler,
    imp_mean,
    args,
    is_test=False,
):
    """
    Creates a dataset by preprocessing features and labels.

    Args:
        data: Dataset object containing X (features), Y (labels), and Y_local (per-level labels).
        scaler: Fitted StandardScaler for feature normalization.
        imp_mean: Fitted SimpleImputer for handling missing values.
        args: Arguments object containing device information.

    Returns:
        list: List of tuples (x, y_levels, y) where:
            - x: Preprocessed feature tensor
            - y_levels: Per-level label tensor
            - y: Full label tensor

    Side Effects:
        - Modifies data.samples.x and data.samples.y in-place by converting
          to tensors and moving to device.
    """
    if is_test:
        shuffle = False
    else:
        shuffle = True

    data.samples.x = (
        torch.tensor(scaler.transform(imp_mean.transform(data.x)))
        .clone()
        .detach()
        .to(args.device)
    )
    data.samples.y = torch.tensor(data.y).clone().detach().to(args.device)
    # Create loaders using local (per-level) y labels
    dataset = list(zip(data.x, data.y_local, data.y))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    return data_loader


def main_local(args):
    """
    Main function to train and test a local hierarchical multi-label classifier.
    """
    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset.dataset_name)

    args.train_methods = get_train_methods(args.method)

    # Load train, val and test set

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    args.data, args.ontology = args.dataset.dataset_name.split("_")

    create_dir(args.results_path)

    # path
    train_path = os.path.join(args.results_path, "train_dataset.pt")
    val_path = os.path.join(args.results_path, "val_dataset.pt")
    test_path = os.path.join(args.results_path, "test_dataset.pt")

    if dataset_name == 'arxiv':
        dataset_type = 'jsonl'
    else:
        dataset_type = 'arff'

    args.hmc_dataset = initialize_dataset_experiments(
        args.dataset.dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        dataset_type=dataset_type,
        is_global=False,
    )

    args.levels_size = args.hmc_dataset.levels_size
    args.input_dim = args.registry.input_dims[args.data]
    args.max_depth = args.hmc_dataset.max_depth
    

    # 1. Initialize scaler and imputer as None (Default for NLP/Text datasets)
    scaler = None
    imp_mean = None
    data_train, data_valid, data_test = args.hmc_dataset.get_datasets()
    # 2. Apply transformations ONLY if the dataset is tabular/numerical
    if args.dataset_type == "arff":
        args.to_eval = args.hmc_dataset.to_eval
        data_concat = np.concatenate((data_train.x, data_valid.x, data_test.x))
        scaler = preprocessing.StandardScaler().fit(data_concat)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(data_concat)

    # 3. Create the test dataloader
    # Ensure your create_dataloader function internally ignores scaler/imp_mean if they are None
    args.test_loader = create_dataloader(
        data_test,
        scaler=scaler,
        imp_mean=imp_mean,
        args=args,
        is_test=True,
    )
    if args.dataset.save_torch_dataset:
        torch.save(args.test_loader, test_path)

    if args.method != "local_test":
        args.val_dataloader = create_dataloader(
            data_valid,
            scaler=scaler,
            imp_mean=imp_mean,
            args=args,
        )
        args.train_dataloader = create_dataloader(
            data_train,
            scaler=scaler,
            imp_mean=imp_mean,
            args=args,
        )
        if args.dataset.save_torch_dataset:
            # Save datasets in torch format
            torch.save(args.train_dataloader, train_path)
            torch.save(args.val_dataloader, val_path)

        train_local(args)
    test_local(args)
    return args.score


def train_local(args):
    """
    Trains a local hierarchical multi-label classifier using the specified \
        arguments.
    This function sets up the experiment environment, loads and preprocesses \
        the dataset,
    creates data loaders for training, validation, and testing, \
        initializes loss functions,
    and either performs hyperparameter optimization or trains the model with\
        provided hyperparameters.
    Args:
        args: An argparse.Namespace or similar object containing the \
            following attributes:
            - dataset_name (str): Name of the dataset in the format "data_ontology".
            - device (str): Device to use ("cpu" or "cuda").
            - batch_size (int): Batch size for data loaders.
            - input_dims (dict): Dictionary mapping dataset names to input dimensions.
            - hpo (str): Whether to perform hyperparameter \
                optimization ("true" or "false").
            - lr_values (list): List of learning rates per level.
            - dropout_values (list): List of dropout rates per level.
            - hidden_dims (list): List of hidden layer sizes per level.
            - num_layers_values (list): List of number of layers per level.
            - weight_decay_values (list): List of weight decay values per level.
            - active_levels (list or None): List of active levels to train, \
                or None for all.
            - Other attributes required by downstream functions.
    Side Effects:
        - Updates the `args` object with data loaders, dataset information, \
            loss functions, and model.
        - Logs experiment information and progress.
    Raises:
        AssertionError: If the lengths of hyperparameter lists \
            do not match the number of levels.
    """

    if args.active_levels is None:
        args.active_levels = list(range(args.max_depth))
    else:
        args.active_levels = [int(x) for x in args.active_levels]
    logging.info("Active levels: %s", args.active_levels)

    args.criterion_list = [nn.BCELoss() for _ in args.hmc_dataset.levels_size]

    if args.hpo:
        logging.info("Hyperparameter optimization")
        args.n_trials = 30
        best_params = args.train_methods["optimize_hyperparameters"](args=args)
        logging.info(best_params)
    else:
        args.hyperparameters = {
            "lr_values": [float(x) for x in args.lr_values],
            "dropout_values": [float(x) for x in args.dropout_values],
            "hidden_dims": [int(x) for x in args.hidden_dims],
            "num_layers_values": [int(x) for x in args.num_layers_values],
            "weight_decay_values": [float(x) for x in args.weight_decay_values],
        }

        # Ensure all hyperparameter lists have the same length as 'max_depth'
        assert_hyperparameter_lengths(
            args,
        )

        params = {
            "levels_size": args.hmc_dataset.levels_size,
            "input_size": args.registry.input_dims[args.data],
            "hidden_dims": args.hidden_dims,
            "num_layers": args.num_layers_values,
            "dropouts": args.dropout_values,
            "active_levels": args.active_levels,
            "results_path": args.results_path,
        }

        model = args.train_methods["model"](**params)
        args.model = model
        logging.info(model)

        start_train = time.perf_counter()
        args.train_methods["train_step"](args)
        end_train = time.perf_counter()
        args.usage = log_system_info(args.device)
        args.training_time_seconds = end_train - start_train
        print("Tempo de treino: %f segundos", args.training_time_seconds)


def test_local(args):
    """
    Tests a local hierarchical multi-label classifier using the specified \
        arguments.
    """

    params = {
        "levels_size": args.hmc_dataset.levels_size,
        "input_size": args.registry.input_dims[args.data],
        "hidden_dims": args.hidden_dims,
        "num_layers": args.num_layers_values,
        "dropouts": args.dropout_values,
        "active_levels": args.active_levels,
        "results_path": args.results_path,
    }

    model = args.train_methods["model"](**params)
    args.model = model
    logging.info(model)
    args.train_methods["test_step"](args)

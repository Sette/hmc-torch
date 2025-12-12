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
    - torch, numpy, sklearn, hmc.utils, hmc.datasets, hmc.models, hmc.trainers

Intended for research and experimentation with HMC classifier benchmarks.

Authors: Bruno Sette
"""

import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.trainers.local_classifier.hpo.hpo_local_level import (
    optimize_hyperparameters,
)

from hmc.trainers.local_classifier.core.test import (
    test_step as test_step_core,
)

from hmc.trainers.local_classifier.core.train import (
    train_step as train_step_core,
)

from hmc.models.local_classifier.baseline import HMCLocalModel
from hmc.models.local_classifier.constraint import HMCLocalModelConstraint

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
from hmc.utils.train.job import parse_str_flags
from hmc.utils.path.dir import create_dir


def get_train_methods(x):
    """
    Given a local classifier method string, returns a mapping of train/test/model/HPO methods.

    Args:
        x (str): Type of local classifier. Options:
            - "local_constrained": Constrained local classifier (per level, constraints enforced)
            - "local": Standard per-level classifier
            - "local_mask": Standard per-level classifier with mask variant

    Returns:
        dict: Dictionary with keys "model", "optimize_hyperparameters", "test_step", and "train_step"
            mapping to the appropriate functions or classes.

    Raises:
        ValueError: If an unknown method string is provided.
    """
    match x:
        case "local_constrained":
            return {
                "model": HMCLocalModelConstraint,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step_core,
                "train_step": train_step_core,
            }
        case "local" | "local_mask" | "local_test":
            return {
                "model": HMCLocalModel,
                "optimize_hyperparameters": optimize_hyperparameters,
                "test_step": test_step_core,
                "train_step": train_step_core,
            }

        case _:
            raise ValueError(f"Método '{x}' não reconhecido.")


def assert_hyperparameter_lengths(
    args,
    lr_values,
    dropout_values,
    hidden_dims,
    num_layers_values,
    weight_decay_values,
):
    """
    Validates that all hyperparameter lists have a length equal to the maximum depth of the hierarchy.

    Args:
        args: Arguments object containing max_depth and relevant experiment settings.
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

    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset_name)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check how many GPUs are available
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")

    args = parse_str_flags(args)

    args.results_path = os.path.join(
        args.output_path,
        "train",
        "local",
        args.dataset_name,
        args.job_id,
    )

    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset_name)

    args.train_methods = get_train_methods(args.method)

    if args.method == "local_constrained":
        logging.info("Using constrained local model")

    # Load train, val and test set

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    args.data, args.ontology = args.dataset_name.split("_")

    create_dir(args.results_path)

    # path
    train_path = os.path.join(args.results_path, "train_dataset.pt")
    val_path = os.path.join(args.results_path, "val_dataset.pt")
    test_path = os.path.join(args.results_path, "test_dataset.pt")
    # is_global = args.method == "global" or args.method == "global_baseline"

    hmc_dataset = initialize_dataset_experiments(
        args.dataset_name,
        device=args.device,
        dataset_path=args.dataset_path,
        dataset_type="arff",
        is_global=False,
    )
    data_train, data_valid, data_test = hmc_dataset.get_datasets()

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
        for (x, y_levels, y) in zip(data_train.X, data_train.Y_local, data_train.Y)
    ]

    val_dataset = [
        (x, y_levels, y)
        for (x, y_levels, y) in zip(data_valid.X, data_valid.Y_local, data_valid.Y)
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
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    args.train_loader = train_loader
    args.val_loader = val_loader
    args.test_loader = test_loader
    args.hmc_dataset = hmc_dataset
    args.levels_size = hmc_dataset.levels_size
    args.input_dim = args.input_dims[args.data]
    args.max_depth = hmc_dataset.max_depth
    args.to_eval = hmc_dataset.to_eval
    args.constrained = True
    if args.active_levels is None:
        args.active_levels = list(range(args.max_depth))
    else:
        args.active_levels = [int(x) for x in args.active_levels]
    logging.info("Active levels: %s", args.active_levels)

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.hpo == "true":
        logging.info("Hyperparameter optimization")
        args.n_trials = 30
        best_params = args.train_methods["optimize_hyperparameters"](args=args)

        logging.info(best_params)
    else:
        if args.lr_values:
            args.lr_values = [float(x) for x in args.lr_values]
            args.dropout_values = [float(x) for x in args.dropout_values]
            # hidden_dims = [int(x) for x in args.hidden_dims]
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
            "dropouts": args.dropout_values,
            "active_levels": args.active_levels,
            "results_path": args.results_path,
            "residual": args.model_regularization == "residual",
        }

        if args.method == "local_constrained":
            params["all_matrix_r"] = hmc_dataset.all_matrix_r

        model = args.train_methods["model"](**params)
        args.model = model
        logging.info(model)
        # Create the model
        # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
        #                                     input_size=args.input_dims[data],
        #                                     hidden_size=args.hidden_dim)
        args.train_methods["train_step"](args)
        args.train_methods["test_step"](args)


def test_local(args):
    """
    Tests a local hierarchical multi-label classifier using the specified arguments.

    This function loads the trained model, prepares the test dataset, and evaluates
    the model's performance on the test set.

    Args:
        args: An argparse.Namespace or similar object containing the following attributes:
            - dataset_name (str): Name of the dataset in the format "data_ontology".
            - device (str): Device to use ("cpu" or "cuda").
            - batch_size (int): Batch size for data loaders.
            - input_dims (dict): Dictionary mapping dataset names to input dimensions.
            - method (str): Type of local classifier.
            - model_path (str): Path to the trained model file.
            - Other attributes required by downstream functions.
    Side Effects:
        - Logs test results and performance metrics.
    """
    logging.info(".......................................")
    logging.info("Predict with %s dataset", args.dataset_name)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check how many GPUs are available
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")

    args = parse_str_flags(args)

    args.results_path = os.path.join(
        args.output_path,
        "train",
        "local",
        args.dataset_name,
        args.job_id,
    )

    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset_name)

    args.train_methods = get_train_methods(args.method)

    if args.method == "local_constrained":
        logging.info("Using constrained local model")

    # Load train, val and test set

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    args.data, args.ontology = args.dataset_name.split("_")

    create_dir(args.results_path)

    # path
    # train_path = os.path.join(args.results_path, "train_dataset.pt")
    # val_path = os.path.join(args.results_path, "val_dataset.pt")
    # test_path = os.path.join(args.results_path, "test_dataset.pt")
    # is_global = args.method == "global" or args.method == "global_baseline"

    hmc_dataset = initialize_dataset_experiments(
        args.dataset_name,
        device=args.device,
        dataset_path=args.dataset_path,
        dataset_type="arff",
        is_global=False,
    )
    data_train, data_valid, data_test = hmc_dataset.get_datasets()

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
    # data_valid.Y = torch.tensor(data_valid.Y).clone().detach().to(args.device)
    # data_train.Y = torch.tensor(data_train.Y).clone().detach().to(args.device)

    # # Create loaders using local (per-level) y labels
    # train_dataset = [
    #     (x, y_levels, y)
    #     for (x, y_levels, y) in zip(data_train.X, data_train.Y_local, data_train.Y)
    # ]

    # val_dataset = [
    #     (x, y_levels, y)
    #     for (x, y_levels, y) in zip(data_valid.X, data_valid.Y_local, data_valid.Y)
    # ]

    test_dataset = [
        (x, y_levels, y)
        for (x, y_levels, y) in zip(data_test.X, data_test.Y_local, data_test.Y)
    ]

    # train_loader = DataLoader(
    #     dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    # )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    # val_loader = DataLoader(
    #     dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    # )
    args.test_loader = test_loader
    args.hmc_dataset = hmc_dataset
    args.levels_size = hmc_dataset.levels_size
    args.input_dim = args.input_dims[args.data]
    args.max_depth = hmc_dataset.max_depth
    args.to_eval = hmc_dataset.to_eval
    args.constrained = True
    if args.active_levels is None:
        args.active_levels = list(range(args.max_depth))
    else:
        args.active_levels = [int(x) for x in args.active_levels]
    logging.info("Active levels: %s", args.active_levels)

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.lr_values:
        args.lr_values = [float(x) for x in args.lr_values]
        args.dropout_values = [float(x) for x in args.dropout_values]
        # hidden_dims = [int(x) for x in args.hidden_dims]
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
        "dropouts": args.dropout_values,
        "active_levels": args.active_levels,
        "results_path": args.results_path,
        "residual": args.model_regularization == "residual",
    }

    if args.method == "local_constrained":
        params["all_matrix_r"] = hmc_dataset.all_matrix_r

    model = args.train_methods["model"](**params)
    args.model = model
    logging.info(model)
    # Create the model
    # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
    #                                     input_size=args.input_dims[data],
    #                                     hidden_size=args.hidden_dim)
    # args.train_methods["train_step"](args)

    args.train_methods["test_step"](args)

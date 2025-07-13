import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.local_classifier.constrained.model import ConstrainedHMCLocalModel
from hmc.train.local_classifier.constrained.hpo.hpo_local import (
    optimize_hyperparameters_per_level,
)
from hmc.train.local_classifier.constrained.test import test_step
from hmc.train.local_classifier.constrained.train import train_step


def train_local(args):
    """
    Trains a local hierarchical multi-label classifier using the specified arguments.
    This function sets up the experiment environment, loads and preprocesses the dataset,
    creates data loaders for training, validation, and testing, initializes loss functions,
    and either performs hyperparameter optimization or trains the model with provided hyperparameters.
    Args:
        args: An argparse.Namespace or similar object containing the following attributes:
            - dataset_name (str): Name of the dataset in the format "data_ontology".
            - device (str): Device to use ("cpu" or "cuda").
            - batch_size (int): Batch size for data loaders.
            - input_dims (dict): Dictionary mapping dataset names to input dimensions.
            - hpo (str): Whether to perform hyperparameter optimization ("true" or "false").
            - lr_values (list): List of learning rates per level.
            - dropout_values (list): List of dropout rates per level.
            - hidden_dims (list): List of hidden layer sizes per level.
            - num_layers_values (list): List of number of layers per level.
            - weight_decay_values (list): List of weight decay values per level.
            - active_levels (list or None): List of active levels to train, or None for all.
            - Other attributes required by downstream functions.
    Side Effects:
        - Updates the `args` object with data loaders, dataset information, loss functions, and model.
        - Logs experiment information and progress.
    Raises:
        AssertionError: If the lengths of hyperparameter lists do not match the number of levels.
    """

    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset_name)
    # Load train, val and test set

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA não está disponível. Usando CPU.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    args.data, args.ontology = args.dataset_name.split("_")
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
    args.hmc_dataset = hmc_dataset
    args.levels_size = hmc_dataset.levels_size
    args.input_dim = args.input_dims[args.data]
    args.max_depth = hmc_dataset.max_depth
    args.to_eval = hmc_dataset.to_eval
    args.constrained = True
    if args.active_levels is None:
        args.active_levels = [i for i in range(args.max_depth)]
    else:
        args.active_levels = [int(x) for x in args.active_levels]
    logging.info("Active levels: %s", args.active_levels)

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.hpo == "true":
        logging.info("Hyperparameter optimization")
        args.n_trials = 30
        best_params = optimize_hyperparameters_per_level(args=args)

        logging.info(best_params)
    else:
        lr_values = [float(x) for x in args.lr_values]
        dropout_values = [float(x) for x in args.dropout_values]
        hidden_dims = [int(x) for x in args.hidden_dims]
        num_layers_values = [int(x) for x in args.num_layers_values]
        weight_decay_values = [float(x) for x in args.weight_decay_values]

        assert (
            len(lr_values)
            == len(dropout_values)
            == len(hidden_dims)
            == len(num_layers_values)
            == len(weight_decay_values)
            == args.max_depth
        ), "All hyperparameter lists must have the same length."

        params = {
            "levels_size": hmc_dataset.levels_size,
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dims,
            "num_layers": num_layers_values,
            "dropout": dropout_values,
            "active_levels": args.active_levels,
            "all_matrix_r": hmc_dataset.all_matrix_r,
        }

        model = ConstrainedHMCLocalModel(**params)
        args.model = model
        logging.info(model)
        # Create the model
        # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
        #                                     input_size=args.input_dims[data],
        #                                     hidden_size=args.hidden_dim)
        train_step(args)
        test_step(args)

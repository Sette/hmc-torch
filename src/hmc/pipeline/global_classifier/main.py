"""
Train a global classifier
"""

import logging

import networkx as nx
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import (
    ConstrainedLightningModel,
    ConstrainedModel,
)
from hmc.pipeline.global_classifier.core.train import train_step
from hmc.utils.train.job import (
    create_job_id_name,
)


def train_global(dataset_name, args):
    """
    Train a global classifier
    """
    logging.info(".......................................")
    logging.info("Experiment with %s dataset ", dataset_name)
    # Load train, val and test set
    args.device = torch.device(args.device)
    args.data, args.ontology = dataset_name.split("_")

    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        dataset_type="arff",
        is_global=True,
    )
    args.train, args.valid, args.test = args.hmc_dataset.get_datasets()

    args.job_id = create_job_id_name(prefix="test")

    args.to_eval = (
        torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    )

    args.results_path = f"output/train/{args.method}-{args.dataset.dataset_name}/{args.job_id}"

    experiment = True
    epochs_by_args = False

    if experiment:
        args.hidden_dim = args.registry.hidden_dims[args.ontology][args.data]
        args.lr = args.registry.lrs[args.ontology][args.data]
        if not epochs_by_args:
            args.epochs = args.registry.all_epochs[args.ontology][args.data]
        args.weight_decay = 1e-5
        args.batch_size = 4
        args.num_layers = 3
        args.dropout = 0.7
        args.non_lin = "relu"

    args.hyperparams = {
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "non_lin": args.non_lin,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    args.r_matrix = np.zeros(args.hmc_dataset.a.shape)
    np.fill_diagonal(args.r_matrix, 1)
    g = nx.DiGraph(args.hmc_dataset.a)
    for i in range(len(args.hmc_dataset.a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            args.r_matrix[i, ancestors] = 1
    args.r_matrix = torch.tensor(args.r_matrix)
    args.r_matrix = args.r_matrix.transpose(1, 0)
    args.r_matrix = args.r_matrix.unsqueeze(0).to(args.device)

    scaler = preprocessing.StandardScaler().fit(
        np.concatenate((args.valid.x, args.valid.x))
    )

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
        np.concatenate((args.valid.x, args.valid.x, args.valid.x))
    )
    args.valid.x = (
        torch.tensor(scaler.transform(imp_mean.transform(args.valid.x)))
        .clone()
        .detach()
        .to(args.device)
    )
    args.valid.y = torch.tensor(args.valid.y).clone().detach().to(args.device)

    args.train.x = (
        torch.tensor(scaler.transform(imp_mean.transform(args.train.x)))
        .clone()
        .detach()
        .to(args.device)
    )
    args.train.y = torch.tensor(args.train.y).clone().detach().to(args.device)

    args.test.x = (
        torch.as_tensor(scaler.transform(imp_mean.transform(args.test.x)))
        .clone()
        .detach()
        .to(args.device)
    )
    args.test.y = torch.as_tensor(args.test.y).clone().detach().to(args.device)

    # Create loaders
    args.train_dataset = list(zip(args.train.x, args.train.y))
    if "others" not in args.dataset.dataset_name:
        # val_dataset = [(x, y) for (x, y) in zip(valid.x, valid.y)]
        for x, y in zip(args.valid.x, args.valid.y):
            args.train_dataset.append((x, y))
    args.test_dataset = list(zip(args.test.x, args.test.y))

    args.train_loader = DataLoader(
        dataset=args.train_dataset, batch_size=args.batch_size, shuffle=True
    )
    args.test_loader = DataLoader(
        dataset=args.test_dataset, batch_size=args.batch_size, shuffle=False
    )

    if "GO" in args.dataset.dataset_name:
        args.num_to_skip = 4
    else:
        args.num_to_skip = 1

    return fit_trainer(args)


def fit_trainer(args):
    """
    Fit the trainer
    """
    if args.method == "globalLM":
        configs = {
            "input_dim": args.registry.input_dims[args.data],
            "hidden_dim": args.hidden_dim,
            "output_dim": args.registry.output_dims[args.ontology][args.data] + args.num_to_skip,
            "hyperparams": args.hyperparams,
            "r_matrix": args.r_matrix,
            "to_eval": args.to_eval,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        args.model = ConstrainedLightningModel(**configs)

        trainer = Trainer(
            max_epochs=args.num_epochs,
            accelerator=args.device,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=20, mode="max")],
        )

        trainer.fit(args.model, args.train_loader, args.val_loader)
        trainer.test(args.model, args.test_loader)
    else:
        baseline = args.method == "global_baseline"
        configs = {
            "input_dim": args.registry.input_dims[args.data],
            "hidden_dim": args.hidden_dim,
            "output_dim": args.registry.output_dims[args.ontology][args.data] + args.num_to_skip,
            "hyperparams": args.hyperparams,
            "r_matrix": args.r_matrix,
            "baseline_model": baseline,
        }
        # Create the model
        args.model = ConstrainedModel(**configs)

        train_step(args)
    return args

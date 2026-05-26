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

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import (
    ConstrainedGNNModel,
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

    args.device = torch.device(args.device)
    is_arxiv = dataset_name == "arxiv"

    if is_arxiv:
        args.data = "arxiv"
        args.ontology = None
        dataset_type = "arxiv"
    else:
        args.data, args.ontology = dataset_name.split("_")
        dataset_type = "arff"

    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        dataset_type=dataset_type,
        is_global=True,
        arxiv_feature_type=args.dataset.arxiv_feature_type,
        arxiv_model_name=args.dataset.arxiv_model_name,
    )
    args.train, args.valid, args.test = args.hmc_dataset.get_datasets()

    args.job_id = create_job_id_name(prefix="test")

    args.to_eval = (
        torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    )

    args.results_path = (
        f"output/train/{args.method}-{args.dataset.dataset_name}/{args.job_id}"
    )

    if is_arxiv:
        defaults = args.registry.arxiv_defaults
        args.hidden_dim = defaults["hidden_dim"]
        args.lr = defaults["lr"]
        args.epochs = defaults["epochs"]
        args.weight_decay = defaults["weight_decay"]
        args.batch_size = defaults["batch_size"]
        args.num_layers = defaults["num_layers"]
        args.dropout = defaults["dropout"]
        args.non_lin = "relu"
        args.input_dim = args.hmc_dataset.input_dim
        args.output_dim = args.hmc_dataset.output_dim
        args.num_to_skip = 1
    else:
        args.hidden_dim = args.registry.hidden_dims[args.ontology][args.data]
        args.lr = args.registry.lrs[args.ontology][args.data]
        args.epochs = args.registry.all_epochs[args.ontology][args.data]
        args.weight_decay = 1e-5
        args.batch_size = 4
        args.num_layers = 3
        args.dropout = 0.7
        args.non_lin = "relu"
        args.input_dim = args.registry.input_dims[args.data]
        args.num_to_skip = 4 if "GO" in dataset_name else 1
        args.output_dim = (
            args.registry.output_dims[args.ontology][args.data] + args.num_to_skip
        )

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

    # Undirected edge_index for ConstrainedGNNModel label-hierarchy GCN.
    # Both directions are included so GCN can propagate parent→child and child→parent.
    rows, cols = np.where(args.hmc_dataset.a > 0)
    fwd = torch.tensor([rows, cols], dtype=torch.long)
    rev = torch.tensor([cols, rows], dtype=torch.long)
    args.label_edge_index = torch.cat([fwd, rev], dim=1).to(args.device)

    if is_arxiv:
        # Text features: convert directly to tensors without sklearn scaling.
        for split in (args.train, args.valid, args.test):
            split.samples.x = (
                torch.tensor(split.x).clone().detach().float().to(args.device)
            )
            split.samples.y = (
                torch.tensor(split.y).clone().detach().float().to(args.device)
            )
    else:
        scaler = preprocessing.StandardScaler().fit(
            np.concatenate((args.train.x, args.valid.x))
        )
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
            np.concatenate((args.train.x, args.valid.x, args.test.x))
        )
        for split in (args.train, args.valid, args.test):
            split.samples.x = (
                torch.tensor(scaler.transform(imp_mean.transform(split.x)))
                .clone()
                .detach()
                .float()
                .to(args.device)
            )
            split.samples.y = (
                torch.tensor(split.y).clone().detach().float().to(args.device)
            )

    args.train_dataset = list(zip(args.train.x, args.train.y))
    if "others" not in dataset_name:
        for x, y in zip(args.valid.x, args.valid.y):
            args.train_dataset.append((x, y))
    args.test_dataset = list(zip(args.test.x, args.test.y))

    args.train_loader = DataLoader(
        dataset=args.train_dataset, batch_size=args.batch_size, shuffle=True
    )
    args.test_loader = DataLoader(
        dataset=args.test_dataset, batch_size=args.batch_size, shuffle=False
    )

    return fit_trainer(args)


def fit_trainer(args):
    """
    Fit the trainer
    """
    if args.method == "globalLM":
        configs = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
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
    elif args.method == "globalGNN":
        configs = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "r_matrix": args.r_matrix,
            "edge_index": args.label_edge_index,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
        }
        args.model = ConstrainedGNNModel(**configs)
        train_step(args)
    else:
        baseline = args.method == "global_baseline"
        configs = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "hyperparams": args.hyperparams,
            "r_matrix": args.r_matrix,
            "baseline_model": baseline,
        }
        args.model = ConstrainedModel(**configs)

        train_step(args)
    return args

"""
Train a global classifier
"""

import logging
import os

import networkx as nx
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import (
    ConstrainedGNNModel,
    ConstrainedLightningModel,
    ConstrainedModel,
)
from hmc.pipeline.global_classifier.core.train import train_step
from hmc.utils.model_cache import ensure_transformer_model_cached
from hmc.utils.train.job import (
    create_job_id_name,
)


def _get_defaults(registry, dataset_name: str) -> dict:
    """Pick default hyperparams based on dataset family."""
    if dataset_name == "wos":
        return registry.wos_defaults
    if any(suffix in (dataset_name or "")
           for suffix in ("_FUN", "_GO", "_others")):
        return registry.gofun_defaults
    return registry.arxiv_defaults


def train_global(dataset_name, args):
    """
    Train a global classifier
    """
    logging.info(".......................................")
    logging.info("Experiment with %s dataset ", dataset_name)

    args.device = torch.device(args.device)
    args.data = dataset_name
    args.ontology = None

    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_model_name=args.dataset.arxiv_model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_cache_dir=args.output_path,
        model_cache_dir=args.dataset.model_cache_dir,
    )
    args.train, args.valid, args.test = args.hmc_dataset.get_datasets()

    args.job_id = create_job_id_name(prefix="test")

    args.to_eval = (
        torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    )

    args.results_path = (
        f"output/train/{args.method}-{args.dataset.dataset_name}/{args.job_id}"
    )

    defaults = _get_defaults(args.registry, dataset_name)
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
    rows, cols = np.where(args.hmc_dataset.a > 0)
    fwd = torch.tensor([rows, cols], dtype=torch.long)
    rev = torch.tensor([cols, rows], dtype=torch.long)
    args.label_edge_index = torch.cat([fwd, rev], dim=1).to(args.device)

    # Text features: convert directly to tensors without sklearn scaling.
    for split in (args.train, args.valid, args.test):
        split.samples.x = (
            torch.tensor(split.x).clone().detach().float().to(args.device)
        )
        split.samples.y = (
            torch.tensor(split.y).clone().detach().float().to(args.device)
        )

    args.train_dataset = list(zip(args.train.x, args.train.y))
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


def _get_transformer_dataset(dataset_name, args, tokenizer, model_name):
    """Return (PyTorchDataset, jsonl_path_or_data_dir) for transformer datasets."""
    if dataset_name == "arxiv":
        from hmc.datasets.arxiv.dataset_arxiv import (  # pylint: disable=import-outside-toplevel
            ArXivPyTorchDataset,
        )
        jsonl_path = os.path.join(
            args.dataset.dataset_path, "arxiv", "arxiv-metadata-oai-snapshot.json"
        )
        max_records = args.dataset.arxiv_max_records or None
        ds = ArXivPyTorchDataset(
            jsonl_path=jsonl_path,
            hierarchy_manager=args.hmc_dataset.hierarchy_manager,
            tokenizer=tokenizer,
            max_records=max_records,
        )
        return ds, max_records
    # wos
    from hmc.datasets.wos.dataset_wos import (  # pylint: disable=import-outside-toplevel
        WOSPyTorchDataset,
    )
    data_dir = os.path.join(args.dataset.dataset_path, "wos")
    ds = WOSPyTorchDataset(
        data_dir=data_dir,
        hierarchy_manager=args.hmc_dataset.hierarchy_manager,
        tokenizer=tokenizer,
    )
    return ds, None


def train_global_e2e(dataset_name, args):
    """End-to-end fine-tuning of a HuggingFace transformer for HMC (globalE2E)."""
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    from hmc.models.global_classifier.e2e.model import (  # pylint: disable=import-outside-toplevel
        E2EConstrainedModel,
    )
    from hmc.pipeline.global_classifier.e2e_train import (  # pylint: disable=import-outside-toplevel
        train_e2e_step,
    )

    args.device = torch.device(args.device)
    model_name = args.dataset.arxiv_model_name

    # 1. Load manager for hierarchy metadata (no pre-computed features)
    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_model_name=model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_load_features=False,
        model_cache_dir=args.dataset.model_cache_dir,
    )

    # 2. Hierarchy-derived tensors
    args.data = dataset_name
    args.ontology = None
    args.to_eval = (
        torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    )
    defaults = _get_defaults(args.registry, dataset_name)
    args.hidden_dim = defaults["hidden_dim"]
    args.lr = defaults["lr"]
    # Respect --epochs from CLI if explicitly set, otherwise use defaults
    if args.epochs == defaults["epochs"] or args.epochs <= 0:
        args.epochs = 10  # E2E converges faster
    args.weight_decay = defaults["weight_decay"]
    args.batch_size = 4  # smaller batch for E2E (GPU memory)
    args.output_dim = args.hmc_dataset.output_dim
    args.num_to_skip = 1

    # R-matrix for hierarchical constraint
    args.r_matrix = np.zeros(args.hmc_dataset.a.shape)
    np.fill_diagonal(args.r_matrix, 1)
    g = nx.DiGraph(args.hmc_dataset.a)
    for i in range(len(args.hmc_dataset.a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            args.r_matrix[i, ancestors] = 1
    args.r_matrix = (
        torch.tensor(args.r_matrix).transpose(1, 0).unsqueeze(0).to(args.device)
    )

    args.results_path = f"output/train/{args.method}-{dataset_name}/{args.job_id}"

    # 4. Build text dataset
    local_model_path = ensure_transformer_model_cached(
        model_name,
        args.dataset.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    text_dataset, _ = _get_transformer_dataset(
        dataset_name, args, tokenizer, local_model_path
    )
    train_set, _val_set, test_set = text_dataset.get_datasets()

    args.train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    args.test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 5. E2E model
    args.model = E2EConstrainedModel(
        model_name=local_model_path,
        output_dim=args.output_dim,
        r_matrix=args.r_matrix,
        hidden_dim=args.hidden_dim,
        num_layers=defaults["num_layers"],
        dropout=defaults["dropout"],
        model_cache_dir=args.dataset.model_cache_dir,
    )

    return train_e2e_step(args)


def train_global_sota(dataset_name, args):
    """globalSOTA: fine-tuned transformer + label-hierarchy GCN for HMC.

    Combines globalE2E (end-to-end transformer fine-tuning) with the GCN label
    encoder.  The model architecture follows HiAGM (Zhou et al., ACL 2020):
    a text encoder and a graph-aware label encoder whose embeddings
    are combined via dot-product scoring.
    """
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    from hmc.models.global_classifier.e2e.model import (  # pylint: disable=import-outside-toplevel
        E2EGNNModel,
    )
    from hmc.pipeline.global_classifier.e2e_train import (  # pylint: disable=import-outside-toplevel
        train_e2e_step,
    )

    args.device = torch.device(args.device)
    model_name = args.dataset.arxiv_model_name

    # 1. Load manager for hierarchy metadata
    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=True,
        arxiv_model_name=model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_load_features=False,
        model_cache_dir=args.dataset.model_cache_dir,
    )

    args.data = dataset_name
    args.ontology = None
    args.to_eval = (
        torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    )
    defaults = _get_defaults(args.registry, dataset_name)
    args.hidden_dim = defaults["hidden_dim"]
    args.lr = defaults["lr"]
    # Respect --epochs from CLI if explicitly set
    if args.epochs == defaults["epochs"] or args.epochs <= 0:
        args.epochs = 10  # SOTA converges faster
    args.weight_decay = defaults["weight_decay"]
    args.batch_size = 4  # smaller batch for fine-tuning
    args.output_dim = args.hmc_dataset.output_dim
    args.num_to_skip = 1

    # 2. R-matrix for hierarchical constraint
    args.r_matrix = np.zeros(args.hmc_dataset.a.shape)
    np.fill_diagonal(args.r_matrix, 1)
    g = nx.DiGraph(args.hmc_dataset.a)
    for i in range(len(args.hmc_dataset.a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            args.r_matrix[i, ancestors] = 1
    args.r_matrix = (
        torch.tensor(args.r_matrix).transpose(1, 0).unsqueeze(0).to(args.device)
    )

    # 3. Undirected edge_index for label-hierarchy GCN
    rows, cols = np.where(args.hmc_dataset.a > 0)
    fwd = torch.tensor([rows, cols], dtype=torch.long)
    rev = torch.tensor([cols, rows], dtype=torch.long)
    label_edge_index = torch.cat([fwd, rev], dim=1).to(args.device)

    # 4. Job ID and output path
    args.job_id = create_job_id_name(prefix="sota")
    args.results_path = f"output/train/{args.method}-{dataset_name}/{args.job_id}"

    # 5. Text DataLoader (tokenized)
    local_model_path = ensure_transformer_model_cached(
        model_name,
        args.dataset.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    text_dataset, _ = _get_transformer_dataset(
        dataset_name, args, tokenizer, local_model_path
    )
    train_set, _val_set, test_set = text_dataset.get_datasets()

    args.train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    args.test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 6. E2E + GNN model
    args.model = E2EGNNModel(
        model_name=local_model_path,
        output_dim=args.output_dim,
        r_matrix=args.r_matrix,
        edge_index=label_edge_index,
        hidden_dim=args.hidden_dim,
        num_layers=defaults["num_layers"],
        dropout=defaults["dropout"],
        model_cache_dir=args.dataset.model_cache_dir,
    )

    return train_e2e_step(args)


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

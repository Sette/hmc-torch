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
from hmc.models.global_classifier.constraint.model import ConstrainedLightningModel

def train_global_lm(dataset_name, args):
    """
    Train a global classifier
    """
    logging.info("Experiment with %s dataset ", dataset_name)
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = dataset_name.split("_")

    # Load dataset paths
    hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset_path,
        dataset_type="arff",
        is_global=True,
    )
    train, valid, test = hmc_dataset.get_datasets()
    to_eval = torch.as_tensor(hmc_dataset.to_eval, dtype=torch.bool).clone().detach()

    experiment = True

    if experiment:
        args.hidden_dim = args.hidden_dims[ontology][data]
        args.lr = args.lrs[ontology][data]
        args.num_epochs = args.epochss[ontology][data]
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

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 \
    # if class i is descendant of class j
    r_matrix = np.zeros(hmc_dataset.dataset_values['a'].shape)
    np.fill_diagonal(r_matrix, 1)
    g = nx.DiGraph(
        hmc_dataset.dataset_values['a']
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(hmc_dataset.dataset_values['a'])):
        ancestors = list(
            nx.descendants(
                g,
                # here we need to use the function nx.descendants(), \
                # because in the directed graph the edges \
                # have source from the descendant \
                # and point towards the ancestor
                i,
            )
        )
        if ancestors:
            r_matrix[i, ancestors] = 1
    r_matrix = torch.tensor(r_matrix)
    # Transpose to get the descendants for each node
    r_matrix = r_matrix.transpose(1, 0)
    r_matrix = r_matrix.unsqueeze(0).to(device)

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.x, valid.x)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
        np.concatenate((train.x, valid.x))
    )
    valid.x = (
        torch.tensor(scaler.transform(imp_mean.transform(valid.x)))
        .clone()
        .detach()
        .to(device),
    )
    valid.y = torch.tensor(valid.y).clone().detach().to(device)

    train.x = (
        torch.tensor(scaler.transform(imp_mean.transform(train.x)))
        .clone()
        .detach()
        .to(device),
    )
    train.y = torch.tensor(train.y).clone().detach().to(device)
    test.x = (
        torch.as_tensor(scaler.transform(imp_mean.transform(test.x)))
        .clone()
        .detach()
        .to(device)
    )
    test.y = torch.as_tensor(test.y).clone().detach().to(device)

    # Create loaders
    train_dataset = list(zip(train.x, train.y))

    val_dataset = list(zip(valid.x, valid.y))

    test_dataset = list(zip(test.x, test.y))

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )

    if "GO" in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    model = ConstrainedLightningModel(
        input_dim=args.input_dims[data],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dims[ontology][data] + num_to_skip,
        hyperparams=args.hyperparams,
        r_matrix=r_matrix,
        to_eval=to_eval,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.device,
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor="train_loss", patience=20, mode="max")],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

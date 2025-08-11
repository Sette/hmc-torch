import networkx as nx
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.global_classifier.constrained.model import ConstrainedLightningModel


def train_globalLM(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
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
    R = np.zeros(hmc_dataset.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        hmc_dataset.A
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(hmc_dataset.A)):
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
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, valid.X)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
        np.concatenate((train.X, valid.X))
    )
    valid.X = (
        torch.tensor(scaler.transform(imp_mean.transform(valid.X)))
        .clone()
        .detach()
        .to(device),
    )
    valid.Y = torch.tensor(valid.Y).clone().detach().to(device)

    train.X = (
        torch.tensor(scaler.transform(imp_mean.transform(train.X)))
        .clone()
        .detach()
        .to(device),
    )
    train.Y = torch.tensor(train.Y).clone().detach().to(device)
    test.X = (
        torch.as_tensor(scaler.transform(imp_mean.transform(test.X)))
        .clone()
        .detach()
        .to(device)
    )
    test.Y = torch.as_tensor(test.Y).clone().detach().to(device)

    # Create loaders
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]

    val_dataset = [(x, y) for (x, y) in zip(valid.X, valid.Y)]

    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

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
        R=R,
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

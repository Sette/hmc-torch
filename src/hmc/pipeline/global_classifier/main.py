import networkx as nx
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import (
    ConstrainedModel,
)
from hmc.utils.train.job import (
    create_job_id_name,
)
from hmc.pipeline.global_classifier.core.train import train_step

def train_global(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = dataset_name.split("_")

    hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset_path,
        dataset_type="arff",
        is_global=True,
    )
    train, valid, test = hmc_dataset.get_datasets()

    job_id = create_job_id_name(prefix="test")

    args.to_eval = torch.as_tensor(hmc_dataset.to_eval, dtype=torch.bool).clone().detach()

    args.results_path = f"output/train/{args.method}-{args.dataset_name}/{job_id}"

    experiment = True
    epochs_by_args = False

    if experiment:
        args.hidden_dim = args.hidden_dims[ontology][data]
        args.lr = args.lrs[ontology][data]
        if not epochs_by_args:
            args.epochs = args.all_epochs[ontology][data]
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

    r_matrix = np.zeros(hmc_dataset.a.shape)
    np.fill_diagonal(r_matrix, 1)
    g = nx.DiGraph(hmc_dataset.a)
    for i in range(len(hmc_dataset.a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r_matrix[i, ancestors] = 1
    r_matrix = torch.tensor(r_matrix)
    r_matrix = r_matrix.transpose(1, 0)
    args.r_matrix = r_matrix.unsqueeze(0).to(device)

    scaler = preprocessing.StandardScaler().fit(np.concatenate((valid.x, valid.x)))

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
        np.concatenate((valid.x, valid.x, valid.x))
    )
    valid.x = (
        torch.tensor(scaler.transform(imp_mean.transform(valid.x)))
        .clone()
        .detach()
        .to(device)
    )
    valid.y = torch.tensor(valid.y).clone().detach().to(device)

    train.x = (
        torch.tensor(scaler.transform(imp_mean.transform(train.x)))
        .clone()
        .detach()
        .to(device)
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
    train_dataset = [(x, y) for (x, y) in zip(train.x, train.y)]
    if "others" not in args.dataset_name:
        # val_dataset = [(x, y) for (x, y) in zip(valid.x, valid.y)]
        for x, y in zip(valid.x, valid.y):
            train_dataset.append((x, y))
    test_dataset = [(x, y) for (x, y) in zip(test.x, test.y)]

    args.train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    args.test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    if "GO" in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Create the model
    args.model = ConstrainedModel(
        args.input_dims[data],
        args.hidden_dim,
        args.output_dims[ontology][data] + num_to_skip,
        args.hyperparams,
        r_matrix,
    )
    

    train_step(args)
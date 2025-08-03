import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.global_classifier.baseline.model import BaselineFFNNModel
from hmc.utils.dir import create_dir


def train_global_baseline(dataset_name, args):
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

    # R = hmc_dataset.compute_matrix_R().to(device)
    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    R = np.zeros(hmc_dataset.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        hmc_dataset.A
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(hmc_dataset.A)):
        # here we need to use the function nx.descendants() \
        # because in the directed graph the edges have source \
        # from the descendant and point towards the ancestor
        ancestors = list(nx.descendants(g, i))
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
        .to(device)
    )
    valid.Y = torch.tensor(valid.Y).clone().detach().to(device)

    train.X = (
        torch.tensor(scaler.transform(imp_mean.transform(train.X)))
        .clone()
        .detach()
        .to(device)
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
    if "others" not in args.datasets:
        # val_dataset = [(x, y) for (x, y) in zip(valid.X, valid.Y)]
        for x, y in zip(valid.X, valid.Y):
            train_dataset.append((x, y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    num_epochs = args.num_epochs
    if "GO" in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Create the model
    model = BaselineFFNNModel(
        args.input_dims[data],
        args.hidden_dim,
        args.output_dims[ontology][data] + num_to_skip,
        args.hyperparams,
        R,
    )
    model = model.to(device)
    to_eval = to_eval.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    # Set patience
    # patience, max_patience = 20, 20
    # max_score = 0.0

    for epoch in range(num_epochs):
        model.train()

        for i, (x, labels) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            # MCLoss
            # constr_output = get_constr_out(output, R)
            # train_output = labels * output.double()
            # train_output = get_constr_out(train_output, R)
            # train_output = (1 - labels) * constr_output.double() + labels * train_output

            loss = criterion(output[:, to_eval].double(), labels[:, to_eval])

            predicted = output.data > 0.5

            # Total number of labels
            # total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            # correct_train = (predicted == labels.byte()).sum()

            loss.backward()
            optimizer.step()

    for i, (x, y) in enumerate(test_loader):

        model.eval()

        x = x.to(device)
        y = y.to(device)

        constrained_output = model(x.float())
        predicted = constrained_output.data > 0.5
        # Total number of labels
        # total = y.size(0) * y.size(1)
        # Total correct predictions
        # correct = (predicted == y.byte()).sum()

        # Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to("cpu")
        cpu_constrained_output = constrained_output.to("cpu")
        y = y.to("cpu")
        to_eval = to_eval.to("cpu")

        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim=0)

    score = average_precision_score(
        y_test[:, to_eval].double(), constr_test.data[:, to_eval], average="micro"
    )
    # create_dir("results")
    create_dir("results/results_baseline")
    f = open("results/results_baseline/" + dataset_name + ".csv", "a")
    f.write(str(args.seed) + "," + str(epoch) + "," + str(score) + "\n")
    f.close()

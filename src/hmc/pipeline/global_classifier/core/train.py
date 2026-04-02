import logging
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import (
    ConstrainedModel,
    get_constr_out,
)
from hmc.utils.dataset.labels import global_to_local_predictions
from hmc.utils.path.files import create_dir
from hmc.utils.path.output import (
    save_dict_to_json,
)
from hmc.utils.train.job import (
    create_job_id_name,
    log_system_info,
)


def train_global(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = dataset_name.split("_")
    best_threshold = 0.19

    hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset_path,
        dataset_type="arff",
        is_global=True,
    )
    train, valid, test = hmc_dataset.get_datasets()

    job_id = create_job_id_name(prefix="test")

    to_eval = torch.as_tensor(hmc_dataset.to_eval, dtype=torch.bool).clone().detach()

    results_path = f"results/train/{args.method}-{args.dataset_name}/{job_id}"

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
    r_matrix = r_matrix.unsqueeze(0).to(device)

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

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    if "GO" in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Create the model
    model = ConstrainedModel(
        args.input_dims[data],
        args.hidden_dim,
        args.output_dims[ontology][data] + num_to_skip,
        args.hyperparams,
        r_matrix,
    )
    model = model.to(device)
    to_eval = to_eval.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    start_train = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for _, (x, labels) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(x.float())

            # MCLoss
            constr_output = get_constr_out(output, r_matrix)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, r_matrix)
            train_output = (1 - labels) * constr_output.double() + labels * train_output

            loss = criterion(train_output[:, to_eval], labels[:, to_eval])
            loss.backward()
            optimizer.step()
    usage = log_system_info(device)
    end_train = time.perf_counter()
    total_time = end_train - start_train
    print("Tempo de treino: %f segundos", total_time)
    for i, (x, y) in enumerate(test_loader):
        model.eval()

        x = x.to(device)
        y = y.to(device)

        constrained_output = model(x.float())
        predicted = constrained_output.data > 0.5
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

    if args.best_threshold:
        logging.info("finding best threshold")

        thresholds = np.linspace(0.1, 0.9, 17)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
            "average_precision_score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            score = precision_recall_fscore_support(
                y_test[:, to_eval],
                constr_test.data[:, to_eval] > actual_threshold,
                average="micro",
                zero_division=0,
            )

            if score[2] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": score[0],
                    "recall": score[1],
                    "f1score": score[2],
                }

        thresholds = np.linspace(best_threshold - 0.01, best_threshold, 10)
        best_scores = {
            "precision": 0,
            "recall": 0,
            "f1score": 0,
            "average_precision_score": 0,
        }

        for actual_threshold in tqdm(thresholds):
            score = precision_recall_fscore_support(
                y_test[:, to_eval],
                constr_test.data[:, to_eval] > actual_threshold,
                average="micro",
                zero_division=0,
            )

            if score[2] > best_scores["f1score"]:
                best_threshold = actual_threshold
                best_scores = {
                    "precision": score[0],
                    "recall": score[1],
                    "f1score": score[2],
                }

    y_pred_local_binary = global_to_local_predictions(
        constr_test.data > best_threshold,
        hmc_dataset.train.local_nodes_idx,
        hmc_dataset.train.nodes_idx,
    )

    y_test_local_binary = global_to_local_predictions(
        y_test,
        hmc_dataset.train.local_nodes_idx,
        hmc_dataset.train.nodes_idx,
    )

    # Get local scores
    local_test_score = {
        level: {"f1score": None, "precision": None, "recall": None}
        for level in range(len(y_test_local_binary))
    }
    for level, (y_test_local, y_pred_local) in enumerate(
        zip(y_test_local_binary, y_pred_local_binary)
    ):
        score = precision_recall_fscore_support(
            y_test_local,
            y_pred_local,
            average="micro",
            zero_division=0,
        )
        local_test_score[level]["precision"] = score[0]  # Precision
        local_test_score[level]["recall"] = score[1]  # Recall
        local_test_score[level]["f1score"] = score[2]  # F1-score
        print("Local evaluation score:")
        print(
            "Level %d Precision: %.4f, Recall: %.4f, F1-score: %.4f"
            % (level, score[0], score[1], score[2])
        )

    score = precision_recall_fscore_support(
        y_test[:, to_eval],
        constr_test.data[:, to_eval] > best_threshold,
        average="micro",
        zero_division=0,
    )

    local_test_score["global"] = {"f1score": None, "precision": None, "recall": None}
    local_test_score["global"]["precision"] = score[0]  # Precision
    local_test_score["global"]["recall"] = score[1]  # Recall
    local_test_score["global"]["f1score"] = score[2]  # F1-score
    local_test_score["global"]["best_threshold"] = best_threshold
    local_test_score["global"]["avg_precision"] = average_precision_score(
        y_test[:, to_eval], constr_test.data[:, to_eval], average="micro"
    )

    local_test_score["global"]["usage"] = usage
    local_test_score["global"]["training_time_seconds"] = total_time

    logging.info("Global evaluation score with best threshold %.3f", best_threshold)
    logging.info(
        "Precision: %.4f, Recall: %.4f, F1-score: %.4f", score[0], score[1], score[2]
    )

    create_dir(results_path)

    save_dict_to_json(
        local_test_score,
        f"{results_path}/test-scores.json",
    )

    print("Average precision score: %.4f" % local_test_score["global"]["avg_precision"])

    args.score = local_test_score["global"]

    return args

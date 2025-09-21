"""
This code was adapted from https://github.com/lucamasera/AWX
"""

import json
import logging
import math
import os
from itertools import chain

import keras
import numpy as np
import pandas as pd
import torch

from hmc.datasets.datasets.gofun import HMCDatasetArff, HMCDatasetCsv
from hmc.utils import create_dir

# Criar um logger
logger = logging.getLogger(__name__)


def create_example(data):
    # Convert inputs explicitly to tensors if they're not yet tensors
    features, labels = data

    # Return as a single dictionary of tensors
    example_tensor = {"features": features, "labels": labels}

    return example_tensor


# Parser convert arff to csv #####


class arff_data_to_csv:
    def __init__(self, arff_file, is_go, output_path, dataset_name):
        self.arff_file = arff_file
        self.output_path = output_path
        if is_go:
            self.dataset_path = os.path.join(output_path, "datasets_GO", dataset_name)
        else:
            self.dataset_path = os.path.join(output_path, "datasets_FUN", dataset_name)
        create_dir(self.dataset_path)
        self.X, self.Y = self.parse_arff_to_csv(arff_file=arff_file, is_go=is_go)
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i, j] = m[j]

    def to_csv(self, dataset="train"):
        """Salva X e Y como um arquivo CSV."""
        # Criando DataFrame para X
        data_path = os.path.join(str(self.dataset_path), dataset, "csv")
        create_dir(data_path)
        batch_size = 1024 * 50  # 50k records from each file batch
        count = 0
        total = math.ceil(len(self.X) / batch_size)
        for i in range(0, len(self.X), batch_size):
            last_index = i + batch_size
            batch_X = self.X[i:last_index]
            batch_Y = self.Y[i:last_index]
            # num_features = len(batch_X[0])
            # columns = [f"feat_{i}" for i in range(num_features)]
            # Create DataFrame and save to CSV
            # df_x = pd.DataFrame(batch_X, columns=columns)
            df_x = pd.DataFrame({"features": batch_X.tolist()})
            # Criando DataFrame para Y, convertendo para int se necessário
            df_y = pd.DataFrame({"labels": batch_Y})
            # Concatenando X e Y
            df = pd.concat([df_x, df_y], axis=1)

            path = f"{data_path}/{str(count).zfill(10)}.csv"

            # Salvando como CSV
            df.to_csv(path, sep="|", index=False)
            print(f"CSV salvo em: {path}")
            count += 1
            print(f"{count}/{total} batches / {count * batch_size} processed")

        print(f"{count}/{total} batches / {len(self.X)} processed")

    def to_pt(self, dataset="train"):
        # logger.info(f'X shape: {self.X.shape} e Y shape: {self.Y.shape}')
        """Salva X e Y como um arquivo pt."""
        # Criando DataFrame para X
        data_path = os.path.join(str(self.dataset_path), dataset, "torch")
        create_dir(data_path)
        batch_size = 1024 * 50  # 50k records from each file batch
        count = 0
        total = math.ceil(len(self.X) / batch_size)
        for i in range(0, len(self.X), batch_size):
            last_index = i + batch_size
            batch_X = self.X[i:last_index]
            batch_Y = self.Y[i:last_index]
            pt_records = [create_example(data) for data in zip(batch_X, batch_Y)]
            path = f"{data_path}/{str(count).zfill(10)}.pt"

            torch.save(pt_records, path)

            print(f"{count} {len(pt_records)} {path}")
            count += 1
            print(f"{count}/{total} batches / {count * batch_size} processed")

        print(f"{count}/{total} batches / {len(self.X)} processed")

    def parse_arff_to_csv(self, arff_file, is_go=False):
        with open(arff_file) as f:
            read_data = False
            X = []
            Y = []

            feature_types = []
            d = []
            cats_lens = []
            all_terms = []
            for line in f:
                if line.startswith("@ATTRIBUTE"):
                    if line.startswith("@ATTRIBUTE class"):
                        h = line.split("hierarchical")[1].strip()
                        for branch in h.split(","):
                            branch = branch.replace("/", ".")
                            all_terms.append(branch)

                    else:
                        _, _, f_type = line.split()

                        if f_type == "numeric" or f_type == "NUMERIC":
                            d.append([])
                            cats_lens.append(1)
                            feature_types.append(
                                lambda x, i: [float(x)] if x != "?" else [np.nan]
                            )

                        else:
                            cats = f_type[1:-1].split(",")
                            cats_lens.append(len(cats))
                            cat_dict = {}
                            for i, key in enumerate(cats):
                                cat_dict[key] = keras.utils.to_categorical(
                                    i, len(cats)
                                ).tolist()
                            d.append(cat_dict)
                            feature_types.append(
                                lambda x, i: d[i].get(x, [0.0] * cats_lens[i])
                            )
                elif line.startswith("@DATA"):
                    read_data = True
                elif read_data:
                    d_line = line.split("%")[0].strip().split(",")
                    lab = d_line[len(feature_types)].replace("/", ".").strip()

                    X.append(
                        list(
                            chain(
                                *[
                                    feature_types[i](x, i)
                                    for i, x in enumerate(d_line[: len(feature_types)])
                                ]
                            )
                        )
                    )

                    Y.append(lab)
            X = np.array(X)
            Y = np.stack(Y)
            categories = {"labels": all_terms}
            if "train" in arff_file:
                labels_file = os.path.join(str(self.dataset_path), "labels.json")
                with open(labels_file, "w+") as f:
                    f.write(json.dumps(categories))
            # np.save('all_terms.npy', np.array(all_terms))
        return X, Y


def initialize_dataset_arff(name, datasets):
    is_go, train, val, test = datasets[name]
    return (
        HMCDatasetArff(train, is_go),
        HMCDatasetArff(val, is_go),
        HMCDatasetArff(test, is_go),
    )


def initialize_dataset_csv(name, datasets):
    train, val, test, labels_json = datasets[name]
    is_go = any(keyword in train for keyword in ["GO", "go"])
    return (
        HMCDatasetCsv(train, labels_json, is_go),
        HMCDatasetCsv(val, labels_json, is_go),
        HMCDatasetCsv(test, labels_json, is_go),
    )


def initialize_other_dataset(name, datasets):
    is_go, train, test = datasets[name]
    return HMCDatasetArff(train, is_go), HMCDatasetArff(test, is_go, True)


def initialize_dataset_arff_tocsv(name, datasets, output_path):
    is_go, train, val, test = datasets[name]
    return (
        arff_data_to_csv(train, is_go, output_path, dataset_name=name),
        arff_data_to_csv(val, is_go, output_path, dataset_name=name),
        arff_data_to_csv(test, is_go, output_path, dataset_name=name),
    )
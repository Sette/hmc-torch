#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import json

from tqdm import tqdm

# In[2]:


import numpy as np
import pandas as pd
from itertools import chain

# from hmc.utils import create_dir, __load_json__


# In[3]:


def create_dir(path):
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True


def __load_json__(path):
    with open(path, "r") as f:
        tmp = json.loads(f.read())

    return tmp


# In[5]:


import torch


# In[6]:


import keras


# In[8]:


data_path = "/home/bruno/git/hmc-torch/data"


# In[9]:


dataset_arff_path = os.path.join(data_path, "HMC_data_arff")


# In[10]:


dataset_fun_path = os.path.join(dataset_arff_path, "datasets_FUN")


# In[11]:


dataset_go_path = os.path.join(dataset_arff_path, "datasets_GO")


# In[12]:


dataset_others_path = os.path.join(dataset_arff_path, "others")


# In[31]:


datasets = {
    "enron_others": (
        False,
        os.path.join(dataset_others_path, "Enron_corr_trainvalid.arff"),
        os.path.join(dataset_others_path, "Enron_corr_test.arff"),
    ),
    "diatoms_others": (
        False,
        os.path.join(dataset_others_path, "Diatoms_train.arff"),
        os.path.join(dataset_others_path, "Diatoms_test.arff"),
    ),
    "imclef07a_others": (
        False,
        os.path.join(dataset_others_path, "ImCLEF07A_Train.arff"),
        os.path.join(dataset_others_path, "ImCLEF07A_Test.arff"),
    ),
    "imclef07d_others": (
        False,
        os.path.join(dataset_others_path, "ImCLEF07D_Train.arff"),
        os.path.join(dataset_others_path, "ImCLEF07D_Test.arff"),
    ),
    "cellcycle_FUN": (
        False,
        os.path.join(dataset_fun_path, "cellcycle_FUN/cellcycle_FUN.train.arff"),
        os.path.join(dataset_fun_path, "cellcycle_FUN/cellcycle_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "cellcycle_FUN/cellcycle_FUN.test.arff"),
    ),
    "derisi_FUN": (
        False,
        os.path.join(dataset_fun_path, "derisi_FUN/derisi_FUN.train.arff"),
        os.path.join(dataset_fun_path, "derisi_FUN/derisi_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "derisi_FUN/derisi_FUN.test.arff"),
    ),
    "eisen_FUN": (
        False,
        os.path.join(dataset_fun_path, "eisen_FUN/eisen_FUN.train.arff"),
        os.path.join(dataset_fun_path, "eisen_FUN/eisen_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "eisen_FUN/eisen_FUN.test.arff"),
    ),
    "expr_FUN": (
        False,
        os.path.join(dataset_fun_path, "expr_FUN/expr_FUN.train.arff"),
        os.path.join(dataset_fun_path, "expr_FUN/expr_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "expr_FUN/expr_FUN.test.arff"),
    ),
    "gasch1_FUN": (
        False,
        os.path.join(dataset_fun_path, "gasch1_FUN/gasch1_FUN.train.arff"),
        os.path.join(dataset_fun_path, "gasch1_FUN/gasch1_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "gasch1_FUN/gasch1_FUN.test.arff"),
    ),
    "gasch2_FUN": (
        False,
        os.path.join(dataset_fun_path, "gasch2_FUN/gasch2_FUN.train.arff"),
        os.path.join(dataset_fun_path, "gasch2_FUN/gasch2_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "gasch2_FUN/gasch2_FUN.test.arff"),
    ),
    "seq_FUN": (
        False,
        os.path.join(dataset_fun_path, "seq_FUN/seq_FUN.train.arff"),
        os.path.join(dataset_fun_path, "seq_FUN/seq_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "seq_FUN/seq_FUN.test.arff"),
    ),
    "spo_FUN": (
        False,
        os.path.join(dataset_fun_path, "spo_FUN/spo_FUN.train.arff"),
        os.path.join(dataset_fun_path, "spo_FUN/spo_FUN.valid.arff"),
        os.path.join(dataset_fun_path, "spo_FUN/spo_FUN.test.arff"),
    ),
    "cellcycle_GO": (
        True,
        os.path.join(dataset_go_path, "cellcycle_GO/cellcycle_GO.train.arff"),
        os.path.join(dataset_go_path, "cellcycle_GO/cellcycle_GO.valid.arff"),
        os.path.join(dataset_go_path, "cellcycle_GO/cellcycle_GO.test.arff"),
    ),
    "derisi_GO": (
        True,
        os.path.join(dataset_go_path, "derisi_GO/derisi_GO.train.arff"),
        os.path.join(dataset_go_path, "derisi_GO/derisi_GO.valid.arff"),
        os.path.join(dataset_go_path, "derisi_GO/derisi_GO.test.arff"),
    ),
    "eisen_GO": (
        True,
        os.path.join(dataset_go_path, "eisen_GO/eisen_GO.train.arff"),
        os.path.join(dataset_go_path, "eisen_GO/eisen_GO.valid.arff"),
        os.path.join(dataset_go_path, "eisen_GO/eisen_GO.test.arff"),
    ),
    "expr_GO": (
        True,
        os.path.join(dataset_go_path, "expr_GO/expr_GO.train.arff"),
        os.path.join(dataset_go_path, "expr_GO/expr_GO.valid.arff"),
        os.path.join(dataset_go_path, "expr_GO/expr_GO.test.arff"),
    ),
    "gasch1_GO": (
        True,
        os.path.join(dataset_go_path, "gasch1_GO/gasch1_GO.train.arff"),
        os.path.join(dataset_go_path, "gasch1_GO/gasch1_GO.valid.arff"),
        os.path.join(dataset_go_path, "gasch1_GO/gasch1_GO.test.arff"),
    ),
    "gasch2_GO": (
        True,
        os.path.join(dataset_go_path, "gasch2_GO/gasch2_GO.train.arff"),
        os.path.join(dataset_go_path, "gasch2_GO/gasch2_GO.valid.arff"),
        os.path.join(dataset_go_path, "gasch2_GO/gasch2_GO.test.arff"),
    ),
    "seq_GO": (
        True,
        os.path.join(dataset_go_path, "seq_GO/seq_GO.train.arff"),
        os.path.join(dataset_go_path, "seq_GO/seq_GO.valid.arff"),
        os.path.join(dataset_go_path, "seq_GO/seq_GO.test.arff"),
    ),
    "spo_GO": (
        True,
        os.path.join(dataset_go_path, "spo_GO/spo_GO.train.arff"),
        os.path.join(dataset_go_path, "spo_GO/spo_GO.valid.arff"),
        os.path.join(dataset_go_path, "spo_GO/spo_GO.test.arff"),
    ),
}


# In[32]:


def create_example(data):
    features, labels = data
    example = {"features": features, "labels": labels}

    return example


# In[33]:


class arff_data_to_csv:
    def __init__(self, arff_file, is_GO, output_path):
        self.arrf_file = arff_file
        self.output_path = output_path
        create_dir(self.output_path)
        self.csv_file = "/".join(arff_file.split("/")[-3:]).replace(".arff", ".csv")
        self.X, self.Y = parse_arff_to_csv(
            arff_file=arff_file, output_path=self.output_path, is_GO=is_GO
        )
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i][j] = m[j]

    def to_csv(self, output_file):
        """Salva X e Y como um arquivo CSV."""
        # Criando DataFrame para X
        output_path = "/".join(output_file.split("/")[:-1])
        create_dir(output_path)

        df_X = pd.DataFrame({"features": [json.dumps(x) for x in self.X]})
        # df_X = pd.DataFrame({'features': json.dumps(self.X)})
        # df_X = pd.DataFrame({'features': json.dumps(self.X)})
        # Criando DataFrame para Y, convertendo para int se necessário
        df_Y = pd.DataFrame({"labels": self.Y})

        # Concatenando X e Y
        df = pd.concat([df_X, df_Y], axis=1)

        # Salvando como CSV
        df.to_csv(output_file, sep="|", index=False)
        print(f"CSV salvo em: {output_file}")

    def to_pt(self, output_path):
        """Salva X e Y como um arquivo pt."""
        # Criando DataFrame para X
        create_dir(output_path)
        batch_size = 1024 * 50  # 50k records from each file batch
        count = 0
        total = math.ceil(len(self.X) / batch_size)
        for i in range(0, len(self.X), batch_size):
            batch_X = self.X[i : i + batch_size]
            batch_Y = self.Y[i : i + batch_size]
            pt_records = [create_example(data) for data in zip(batch_X, batch_Y)]
            path = f"{output_path}/{str(count).zfill(10)}.pt"

            torch.save(pt_records, path)

            print(f"{count} {len(pt_records)} {path}")
            count += 1
            print(f"{count}/{total} batches / {count * batch_size} processed")

        print(f"{count}/{total} batches / {len(self.X)} processed")


def parse_arff_to_csv(arff_file, output_path, is_GO=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []

        feature_types = []
        d = []
        cats_lens = []
        all_terms = []
        for num_line, l in enumerate(f):
            if l.startswith("@ATTRIBUTE"):
                if l.startswith("@ATTRIBUTE class"):
                    h = l.split("hierarchical")[1].strip()
                    for branch in h.split(","):
                        branch = branch.replace("/", ".")
                        all_terms.append(branch)

                else:
                    _, f_name, f_type = l.split()

                    if f_type == "numeric" or f_type == "NUMERIC":
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(
                            lambda x, i: [float(x)] if x != "?" else [np.nan]
                        )

                    else:
                        cats = f_type[1:-1].split(",")
                        cats_lens.append(len(cats))
                        d.append(
                            {
                                key: keras.utils.to_categorical(i, len(cats)).tolist()
                                for i, key in enumerate(cats)
                            }
                        )
                        feature_types.append(
                            lambda x, i: d[i].get(x, [0.0] * cats_lens[i])
                        )
            elif l.startswith("@DATA"):
                read_data = True
            elif read_data:
                d_line = l.split("%")[0].strip().split(",")
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

                # for t in lab.split('@'):
                #    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] = 1
                #    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(lab)
        # X = np.array(X)
        # Y = np.stack(Y)
        categories = {"labels": all_terms}
        if "train" in arff_file or "Train" in arff_file:

            create_dir(output_path)
            labels_file = os.path.join(output_path, "labels.json")
            with open(labels_file, "w+") as f:
                f.write(json.dumps(categories))
        # np.save('all_terms.npy', np.array(all_terms))
    return X, Y


# In[34]:
def initialize_dataset_arff_tocsv(name, datasets, output_path):
    dataset = datasets[name]

    is_GO = dataset[0]
    train = dataset[1]
    test = dataset[2]
    if len(dataset) == 4:
        val = dataset[3]
        return (
            arff_data_to_csv(train, is_GO, output_path),
            arff_data_to_csv(test, is_GO, output_path),
            arff_data_to_csv(val, is_GO, output_path),
        )
    else:
        return (
            arff_data_to_csv(train, is_GO, output_path),
            arff_data_to_csv(test, is_GO, output_path),
        )


def main():

    # In[35]:
    for dataset_name in tqdm(datasets.keys()):
        print(f"Processing dataset: {dataset_name}")

        # dataset_name = 'seq_FUN'

        # In[36]:

        output_path = os.path.join(data_path, "HMC_data_csv", dataset_name)

        # In[38]:

        dataset = initialize_dataset_arff_tocsv(dataset_name, datasets, output_path)

        train = dataset[0]
        test = dataset[1]

        train.to_csv(os.path.join(output_path, "train.csv"))
        test.to_csv(os.path.join(output_path, "test.csv"))
        if len(dataset) == 3:
            val = dataset[2]
            val.to_csv(os.path.join(output_path, "val.csv"))


if __name__ == "__main__":
    main()

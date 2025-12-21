#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import json
import yaml


def __load_json__(path):
    with open(path, "r") as f:
        tmp = json.loads(f.read())
    return tmp


# In[24]:


def _get_hpo_values(dataset_name):
    dataset_path = os.path.join(hpo_path, dataset_name)
    hpo_train_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
    json_file_path = os.path.join(hpo_train_path, f"best_params_{dataset_name}.json")
    return json_file_path


# In[25]:


def get_hpo_values(datasets):
    datasets_params = {}
    for dataset_name in datasets:
        json_file = _get_hpo_values(dataset_name)
        print(json_file)
        if not os.path.exists(json_file):
            print(f"Arquivo não encontrado para dataset: {dataset_name}")
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        # Cria listas para cada hiperparâmetro
        hidden_dims = []
        lr_values = []
        dropout_values = []
        num_layers_values = []
        weight_decay_values = []
        # Coleta valores de cada trial
        for k in data:
            trial = data[k]
            hidden_dims.append(trial["hidden_dims"])
            lr_values.append(trial["lr"])
            dropout_values.append(trial["dropout"])
            num_layers_values.append(trial["num_layers"])
            weight_decay_values.append(trial["weight_decay"])
        # Adiciona no dicionário final
        datasets_params[dataset_name] = {
            "hidden_dims": hidden_dims,
            "lr_values": lr_values,
            "dropout_values": dropout_values,
            "num_layers_values": num_layers_values,
            "weight_decay_values": weight_decay_values,
        }
    # Formata para salvar compatível com YAML de exemplo
    out_dict = {"datasets_params": datasets_params}
    with open("datasets_params.yaml", "w") as f:
        yaml.dump(out_dict, f, allow_unicode=True, sort_keys=False)
    print("Arquivo YAML criado: datasets_params.yaml")


# In[ ]:


if __name__ == "__main__":

    hpo_path = "/home/bruno/Projetos/git/hmc-torch/results/hpo/local"

    # In[12]:

    # hpo_path = r'C:\Users\bruno\git\hmc-torch\results\hpo\local_constraint'

    # In[21]:

    datasets = os.listdir(hpo_path)
    get_hpo_values(datasets)

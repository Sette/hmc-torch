"""
HPO Results Processing Module

This module provides functionality to extract and consolidate hyperparameter optimization
(HPO) results from multiple datasets into a unified YAML configuration file.

The module processes JSON files containing HPO trial results and aggregates the
hyperparameter values across all trials for each dataset, creating a structured
output suitable for further analysis or configuration purposes.

Main Functions:
    get_hpo_values(): Processes HPO results and generates consolidated YAML output
    __load_json__(path): Utility function for loading and parsing JSON files

Dependencies:
    - json: For parsing JSON files
    - os: For file system operations
    - yaml: For generating YAML output

Output:
    Creates 'datasets_params.yaml' file containing aggregated hyperparameter values
    organized by dataset name.
"""

import json
import os

import yaml


def __load_json__(path):
    """Load and parse a JSON file.
    Args:
        path (str): Path to the JSON file to load.
    Returns:
        dict: Parsed JSON data as a Python dictionary.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        tmp = json.loads(f.read())
    return tmp


def _get_hpo_values(dataset_name, hpo_path):
    dataset_path = os.path.join(hpo_path, dataset_name)
    hpo_train_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
    json_file_path = os.path.join(hpo_train_path, f"best_params_{dataset_name}.json")
    return json_file_path


def get_hpo_values():
    """Extract hyperparameter optimization values from multiple datasets and save to YAML.

    This function processes HPO results for multiple datasets, extracting hyperparameter
    values from JSON files and consolidating them into a single YAML output file.

    Args:
        datasets (list): List of dataset names to process.
        hpo_path (str): Base path where HPO results are stored.

    Returns:
        None: Function saves results to 'datasets_params.yaml' file.

    Side Effects:
        - Prints file paths and error messages to stdout
        - Creates 'datasets_params.yaml' file in current directory

    File Structure Expected:
        {hpo_path}/{dataset_name}/{trial_dir}/best_params_{dataset_name}.json

    JSON Format Expected:
        {
            "trial_id": {
                "hidden_dims": int,
                "lr": float,
                "dropout": float,
                "num_layers": int,
                "weight_decay": float
            }
        }
    """
    hpo_path = "/home/bruno/Projetos/git/hmc-torch/results/hpo/local"
    datasets = os.listdir(hpo_path)

    datasets_params = {}
    for dataset_name in datasets:
        json_file = _get_hpo_values(dataset_name, hpo_path)
        print(json_file)
        if not os.path.exists(json_file):
            print(f"Arquivo não encontrado para dataset: {dataset_name}")
            continue
        with open(json_file, "r", encoding="utf-8") as f:
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
    with open("datasets_params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(out_dict, f, allow_unicode=True, sort_keys=False)
    print("Arquivo YAML criado: datasets_params.yaml")


# In[ ]:


if __name__ == "__main__":

    get_hpo_values()

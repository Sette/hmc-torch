"""
This module contains the dataset class for HMC local classifier.
"""

import os

import torch


class HMCDatasetTorch:
    """
    Dataset torch para HMC local classifier.
    """
    def __init__(self, path):
        """
        Inicializa o dataset.
        :param data: Estrutura de dados carregada do .pt
        """
        self.x = []
        self.y = []
        self.examples = []

        pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]

        for file in pt_files:
            data = torch.load(os.path.join(path, file), weights_only=False)
            if isinstance(data, list):
                self.examples.extend(data)
            elif isinstance(data, dict):
                self.examples.append(data)
            else:
                raise ValueError(f"Arquivo {file} possui tipo inesperado: {type(data)}")

        self.parse_to_array()

    def __len__(self):
        """
        Retorna o número de exemplos no dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Retorna uma amostra dos dados.
        """
        item = self.examples[idx]
        features = item["features"]  # tensor
        labels = item["labels"]  # lista de strings
        return features, labels

    def parse_to_array(self):
        """
        Parse os dados do dataset para arrays.
        """
        for example in self.examples:
            self.x.append(example["features"])
            self.y.append(example["labels"])

    def set_y(self, y):
        self.y = y

import os

import torch


class HMCDatasetTorch:
    def __init__(self, path):
        """
        Inicializa o dataset.
        :param data: Estrutura de dados carregada do .pt
        """
        self.X = []
        self.Y = []
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
        for example in self.examples:
            self.X.append(example["features"])
            self.Y.append(example["labels"])

    def set_y(self, y):
        self.Y = y

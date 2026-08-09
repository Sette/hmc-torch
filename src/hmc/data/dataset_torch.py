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
        Initialize the dataset.
        :param path: Path to directory containing .pt files.
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
                raise TypeError(f"File {file} has unexpected type: {type(data)}")

        self.parse_to_array()

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        """
        item = self.examples[idx]
        features = item["features"]  # tensor
        labels = item["labels"]  # list of strings
        return features, labels

    def parse_to_array(self):
        """
        Parses the dataset to arrays.
        """
        for example in self.examples:
            self.x.append(example["features"])
            self.y.append(example["labels"])

    def set_y(self, y):
        """
        Sets the labels of the dataset.
        :param y: The dataset's labels
        """
        self.y = y

# -*- coding: utf-8 -*-
# @Author  : settebr
# @File    : dataset_csv.py
import ast
import os

import numpy as np
import pandas as pd


def load_and_concat_csv_files(directory, sep="|"):
    """
    Loads all CSV files from a given directory and concatenates them \
        into a single DataFrame.
    :param directory: Path to the directory containing CSV files.
    :return: A single concatenated DataFrame.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, sep=sep)
        # Convert 'features' column from string representation of lists to actual lists

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True) if dataframes else None


def safe_parse(val):
    if pd.isna(val) or val == "None":
        return None
    try:
        parsed = ast.literal_eval(val)
        return parsed
    except Exception as e:
        print(f"Erro ao converter: {val} => {e}")
        return None


class HMCDatasetCsv:
    def __init__(self, csv_path, is_go):
        self.df = pd.DataFrame()
        self.X, self.Y = None, None
        self.is_go = is_go
        self.csv_path = csv_path
        self.parse_csv()

    def set_y(self, y):
        self.Y = np.stack(y)

    def transform_features(self):
        # print(self.df.features[0])
        # print(type(self.df.features[0]))
        self.df.features = self.df.features.apply(safe_parse)
        self.X = self.df.features.values.tolist()
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i, j] = m[j]

    def parse_csv(self):
        # self.df = pd.read_csv(self.csv_path, sep='|')
        if self.csv_path.endswith(".csv"):
            self.df = pd.read_csv(self.csv_path, sep="|")
        else:
            # Assuming the path is a directory containing multiple CSV files
            self.df = load_and_concat_csv_files(self.csv_path, sep="|")
        # X = df['features'].tolist()
        # self.df['features'] = self.df['features'].apply(json.loads)
        print(self.df.columns)
        if "features" not in self.df.columns:
            raise ValueError("The CSV file does not contain a 'features' column.")
        if "categories" not in self.df.columns:
            raise ValueError("The CSV file does not contain a 'categories' column.")
        self.Y = self.df["categories"].tolist()
        self.transform_features()

        # X = np.array([json.loads(x) for x in df['features']])

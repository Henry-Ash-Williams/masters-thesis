from enum import Enum

import pandas as pd

DEFAULT_CSV_PATH = "/Users/henrywilliams/Documents/uni/masters-thesis/notebooks/data/malware-dataset.csv"


class Label(Enum):
    BENIGN = 0
    MALWARE = 1

    def __str__(self):
        return str(self.name).title()


class MalwareDataset:
    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, idx: int):
        if idx >= len(self.df):
            raise IndexError("Dataframe index out of range")

        _, path, label = self.df.iloc[idx, :]

        return path, label

    def __len__(self):
        return len(self.df)

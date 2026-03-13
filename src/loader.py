import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self):
        self.data = None

    def load_csv(self, path):
        self.data = pd.read_csv(path)
        return self.data

    def load_sklearn_dataset(self):
        self.data = load_breast_cancer(as_frame=True).frame
        return self.data

    def split(self, data, target, test_size=0.2, random_state=42):
        X = data.drop(target, axis=1)
        y = data[target]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil

def get_data():
    train = pd.read_csv("./data/train.csv")
    y = train["target"]
    X = train.drop(["ID_code", "target"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), ceil(0.2*len(ds))])

    test = pd.read_csv("./data/test.csv")
    test_ids = test["ID_code"]
    X = test.drop(["ID_code"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids
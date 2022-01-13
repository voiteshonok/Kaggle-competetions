import pandas as pd
import numpy as np
import torch

def get_predictions(loader, model, device):
    model.eval()
    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            saved_preds += preds.tolist()
            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels


def get_submission(loader, model, device, test_ids):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)
            all_preds += preds.float().tolist()

    model.train()

    return pd.DataFrame({
        "ID_code": test_ids.values,
        "target": np.array(all_preds)
    })
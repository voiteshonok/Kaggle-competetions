import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

def get_predictions(loader, model, device):
    model.eval()
    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for sample in loader:
            x = sample["image"].to(device)
            y = sample["label"].to(device)
            preds = model(x)
            saved_preds += preds.tolist()
            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels

def save_feature_vectors(model, loader, file, device):
    model.eval()
    images, labels = [], []

    for idx, sample in enumerate(tqdm(loader)):
        x, y = sample["image"], sample["label"]
        x = x.to(device)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"X_{file}.npy", np.concatenate(images, axis=1))
    np.save(f"y_{file}.npy", np.concatenate(labels, axis=1))

    model.train()
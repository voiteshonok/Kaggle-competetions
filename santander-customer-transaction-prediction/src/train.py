import re
from turtle import forward
import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_data
from utils import get_predictions, get_submission


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 512

model = NN(input_size=200)
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()

train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(EPOCHS):
    #data, targets = next(iter(train_loader)) # sanity check that model learns smth

    probabilities, true = get_predictions(val_loader, model, DEVICE)
    print(f'epoch: {epoch}')
    print(f'val roc_auc: {metrics.roc_auc_score(true, probabilities)}')

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(DEVICE)
        model = model.to(DEVICE)

        #forward
        preds = model(data)

        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()

test_df = get_submission(test_loader, model, DEVICE, test_ids)
test_df.to_csv('sub.csv', index=False)
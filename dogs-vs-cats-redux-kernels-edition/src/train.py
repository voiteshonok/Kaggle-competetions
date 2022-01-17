from operator import imod
import re
from turtle import forward
import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models

from dataset import DogsCatsDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32

AA = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 1)
        
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.linear(x)
        return torch.sigmoid(x).view(-1)

model = NN()
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

train_loader = DataLoader(DogsCatsDataset("data/train", AA, split="train"), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DogsCatsDataset("data/train", AA, split="val"), batch_size=BATCH_SIZE, shuffle=True)
#train_dataloader = DataLoader(DogsCatsDataset("data/train", AA), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    sample = next(iter(train_loader)) # sanity check that model learns smth
    

    #probabilities, true = get_predictions(val_loader, model, DEVICE)
    #print(f'epoch: {epoch}')
    #print(f'val roc_auc: {metrics.roc_auc_score(true, probabilities)}')

    #for batch_idx, sample in enumerate(tqdm(train_loader)):
    data, targets = sample["image"], sample["label"]
    targets = targets.to(DEVICE)
    optimizer.zero_grad()

    data = data.to(DEVICE)
    model = model.to(DEVICE)

    #forward
    preds = model(data)

    loss = loss_fn(preds, targets)
    print(loss)

    loss.backward()
    optimizer.step()
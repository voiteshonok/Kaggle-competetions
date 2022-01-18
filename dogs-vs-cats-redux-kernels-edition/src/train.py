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
from sklearn import metrics
from copy import copy
from efficientnet_pytorch import EfficientNet

from dataset import DogsCatsDataset
from utils import get_predictions, save_feature_vectors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32

base_transform = A.Compose(
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

model = EfficientNet.from_pretrained("efficientnet-b4")
#optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
#loss_fn = nn.BCEWithLogitsLoss()

train_loader = DataLoader(DogsCatsDataset("data/train", base_transform, split="train"), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DogsCatsDataset("data/train", base_transform, split="val"), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(DogsCatsDataset("data/train", base_transform), batch_size=BATCH_SIZE, shuffle=False)

model = model.to(DEVICE)
save_feature_vectors(model, train_loader, file = "train_b4", device = DEVICE)
save_feature_vectors(model, val_loader, file = "val_b4", device = DEVICE)
save_feature_vectors(model, test_loader, file = "test_b4", device = DEVICE)
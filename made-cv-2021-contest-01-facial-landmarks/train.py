"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import NUM_PTS, CROP_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

from EarlyStopping import EarlyStopping

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()

def data_provider(split=None, fold=None):
    """ return pytorch Dataloader for train or validation mode"""

    print(f"Reading {split} data for fold={fold}...")
    if split == "train":
        train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                                 train_transforms, split="train", fold=fold)
        return data.DataLoader(train_dataset,
                               batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, shuffle=True, drop_last=True)
    elif split == "val":
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                               train_transforms, split="val", fold=fold)
        return data.DataLoader(val_dataset,
                               batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, shuffle=False, drop_last=False)
    else:
        raise ValueError("Illegal 'split' parameter value")



def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.to(torch.device("cpu"))
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"]#.to(device)

        with torch.no_grad():
            pred_landmarks = model(images)
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader

        self.val_dataloader = val_dataloader

        self.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, amsgrad=True)
        self.loss_fn = fnn.mse_loss

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, steps_per_epoch=len(train_dataloader), max_lr=0.1, epochs=5)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)

        self.early_stopping = EarlyStopping(patience=5, verbose=True)

    def start(self, epochs, start_with=0):
        # 2. train & validate
        print("Ready for training...")
        best_val_loss = np.inf
        for epoch in range(start_with, epochs + start_with):
            train_loss = train(self.model, self.train_dataloader, self.loss_fn, self.optimizer, device=self.device)
            val_loss = validate(self.model, self.val_dataloader, self.loss_fn, device=self.device)


            print("Epoch #{:2}:\ttrain loss: {:7.4}\tval loss: {:7.4}".format(epoch, train_loss, val_loss))

            self.scheduler.step()
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # save weights only if model has improved score on validation
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                with open(os.path.join("./runs", "resnet50_ep{}_loss{:.4}.pth".format(epoch, val_loss)),
                          "wb") as fp:
                    torch.save(self.model.state_dict(), fp)

        return epoch + 1


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",)),
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    model = models.resnet50(pretrained=True)
    model.requires_grad_(False)

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_fn = fnn.mse_loss

    name = './runs/resnet50_ep14_loss17.76.pth'
    with open(f"{name}", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    for param in model.parameters():
        param.requires_grad = False
        #
        # train only head
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        #
        # train layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    trainer = Trainer(model, train_dataloader, val_dataloader)    
    trainer.start(5, 11)
    '''
    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))
'''

if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))

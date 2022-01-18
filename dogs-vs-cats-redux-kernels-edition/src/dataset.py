import os
import cv2
import torch
import tqdm
import numpy as np
from torch.utils import data
from math import ceil

np.random.seed(42)
torch.manual_seed(42)

TRAIN_SIZE = 0.9


class DogsCatsDataset(data.Dataset):
    def __init__(self, root, transforms, split="train"):
        """
        Params
        ________
        split should be "train" or "val" or "test"
        """
        super(DogsCatsDataset, self).__init__()
        self.root = root
        self.transforms = transforms

        self.image_names = []

        names = os.listdir(root)

        if split == "test":
            self.image_names = names
            return
            
        np.random.shuffle(names)
        self.image_names = names[:int(TRAIN_SIZE * len(names))] if split=="train" else names[ceil(TRAIN_SIZE * len(names)):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        sample = {}

        image_path = os.path.join(self.root, self.image_names[index])
        if self.image_names[index].split(".").__len__() > 2:
            sample["label"] = np.float32("dog" in self.image_names[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample["image"] = self.transforms(image=sample["image"])["image"]

        return sample
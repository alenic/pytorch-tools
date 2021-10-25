import sys

import pytorchtools as pt
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

dataset_root = "dataset_for_test"

def cv2_loader(path):
    try:
        sample = cv2.imread(path, cv2.IMREAD_COLOR)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    except:
        print("Error", path)
    return sample


val_transform = lambda x: A.Compose([A.Resize(width=224,height=224), A.Normalize(mean=0, std=1), ToTensorV2()])(image=x)["image"]
val_dataset = ImageFolder(dataset_root, loader=cv2_loader, transform=None)

n_tr = len(pt.valid_transforms)

fig, ax = plt.subplots(6, n_tr)

for i, tr_str in enumerate(pt.valid_transforms):

    ax[0, i].set_title(tr_str)
    for m in range(6):
        r_dataset = pt.RobustnessDataset(val_dataset, distortion_str=tr_str, magnitude=m, image_id=0, transform=val_transform)

        val_loader = DataLoader(
            r_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        for image, label in val_loader:
            ax[m, i].imshow(image[0].numpy().transpose(1,2,0))

plt.show()

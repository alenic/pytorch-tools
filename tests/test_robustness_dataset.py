from torchvision.transforms.transforms import Normalize
from robustenss_dataset import RobustnessDataset
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

dataset_root = "/media/alenic/SecondM2/dataset/enel/isolatori_rigidi_max512_csv1.0_r_splitted"

def cv2_loader(path):
    try:
        sample = cv2.imread(path, cv2.IMREAD_COLOR)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB).astype(np.float32)
    except:
        print("Error", path)
    return sample


val_transform = A.Compose([A.Resize(width=224,height=224), A.Normalize(mean=0, std=1), ToTensorV2()])

val_dataset = ImageFolder(os.path.join(dataset_root, "train"), loader=cv2_loader, transform=None)
r_dataset = RobustnessDataset(val_dataset, distortion_str="low_brightness", magnitude=5, image_id=0, transform=val_transform)

val_loader = DataLoader(
    r_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

for image, label in val_loader:
    plt.imshow(image[0].numpy().transpose(1,2,0))
    plt.show()

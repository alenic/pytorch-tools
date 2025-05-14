"""
Useful Dataset
author: Alessandro Nicolosi - https://github.com/alenic/pytorch-tools
"""

import torchvision.datasets as datasets
import os
import numpy as np


class InferenceDataset(datasets.vision.VisionDataset):
    def __init__(
        self,
        root,
        transform=None,
        walk=False,
        max_images=None,
        shuffle=False,
        loader=datasets.folder.default_loader,
    ):

        super(InferenceDataset, self).__init__(
            root, transform=transform, target_transform=None
        )
        self.loader = loader

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )

        self.samples = []

        if walk:
            for folder, subfolder, files in os.walk(root):
                for file in files:
                    for ext in extensions:
                        if file.lower().endswith(ext):
                            file_path = os.path.join(
                                os.path.join(os.path.abspath(folder), file)
                            )
                            self.samples.append(file_path)
                            break
        else:
            file_list = os.listdir(root)
            for filename in file_list:
                for ext in extensions:
                    if filename.lower().endswith(ext):
                        file_path = os.path.join(root, filename)
                        self.samples.append(file_path)
                        break

        if shuffle:
            np.random.shuffle(self.samples)

        if max_images is not None:
            self.samples = self.samples[:max_images]

    def get_paths(self):
        return self.samples

    def __getitem__(self, index):
        """
        Args:
        index (int): Index.

        Returns:
        tuple: (sample, path) where path is the path file
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.samples)

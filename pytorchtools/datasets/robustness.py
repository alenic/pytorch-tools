"""
Dataset for corruption evaluation
"""

# edited by Alessandro Nicolosi - https://github.com/alenic

from torch.utils.data import Dataset
from functools import partial
import cv2
import numpy as np
from PIL import Image, ImageEnhance

robustness_config = dict(
    rotation_interp=cv2.INTER_LINEAR, rotation_border_mode=cv2.BORDER_REFLECT
)

valid_transforms = [
    "rotate_c",
    "rotate_cc",
    "jpeg_compression",
    "blur",
    "high_brightness",
    "low_brightness",
    "high_contrast",
    "low_contrast",
]


def distortion_rotate(image, magnitude, cc=False):
    if cc:
        angle = [0, 10, 20, 30, 40, 50][magnitude]
    else:
        angle = [0, -10, -20, -30, -40, -50][magnitude]
    h, w, c = image.shape
    image_center = (h // 2, w // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=robustness_config["rotation_interp"],
        borderMode=robustness_config["rotation_border_mode"],
    )
    return image


def distortion_jpeg_compression(image, magnitude):
    quality = [100, 80, 60, 40, 20, 10][magnitude]

    img_encode = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    image = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
    return image


def distortion_blur(image, magnitude):
    h, w, c = image.shape
    max_wh = max(w, h)
    k_size = max(
        int(
            [0, max_wh / 100, max_wh / 90, max_wh / 80, max_wh / 70, max_wh / 60][
                magnitude
            ]
        ),
        2,
    )
    image = cv2.blur(image, ksize=(k_size, k_size))
    return image


def distortion_brightness(image, magnitude, high=True):
    if high:
        base = 255 - image.mean()
    else:
        base = image.mean()

    delta_b = int(
        [0, base * 0.15, base * 0.30, base * 0.45, base * 0.6, base * 0.75][magnitude]
    )

    if high:
        image = image.astype(int) + delta_b
    else:
        image = image.astype(int) - delta_b

    return np.clip(image, 0, 255).astype(np.uint8)


def distortion_contrast(image, magnitude, high=True):
    im = Image.fromarray(image, "RGB")
    enhancer = ImageEnhance.Contrast(im)

    delta_b = [0, 0.15, 0.30, 0.45, 0.6, 0.75][magnitude]

    if high:
        image = np.array(enhancer.enhance(1 + 1.2 * delta_b))
    else:
        image = np.array(enhancer.enhance(1 - delta_b))

    return np.clip(image, 0, 255).astype(np.uint8)


class RobustnessDataset(Dataset):
    def __init__(
        self,
        dataset,
        distortion_str="rotate_c",
        magnitude=1,
        image_id=0,
        transform=None,
        resize=None,
        resize_interp=cv2.INTER_LINEAR,
    ):
        """
        - dataset: torch.utils.data.Dataset
        - distortion_str: type of distortion: rotate_c (clockwise), rotate_cc (counter-clockwise), jpeg_compression, blur, high_brightness, low_birghtness

        """
        self.dataset = dataset
        self.distortion = self.get_distortion(distortion_str)
        self.magnitude = magnitude
        assert self.magnitude >= 0 and self.magnitude <= 5

        self.image_id = image_id
        self.transform = transform
        self.resize = resize
        self.resize_interp = resize_interp

    def __len__(self):
        return len(self.dataset)

    def get_distortion(self, distortion_str):
        assert distortion_str in valid_transforms

        if distortion_str == "rotate_c":
            return distortion_rotate
        elif distortion_str == "rotate_cc":
            return partial(distortion_rotate, cc=True)
        elif distortion_str == "jpeg_compression":
            return distortion_jpeg_compression
        elif distortion_str == "blur":
            return distortion_blur
        elif distortion_str == "high_brightness":
            return partial(distortion_brightness, high=True)
        elif distortion_str == "low_brightness":
            return partial(distortion_brightness, high=False)
        elif distortion_str == "high_contrast":
            return partial(distortion_contrast, high=True)
        elif distortion_str == "low_contrast":
            return partial(distortion_contrast, high=False)

    def __getitem__(self, index):
        item = list(self.dataset[index])
        image = item[self.image_id]
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=self.resize_interp)

        assert isinstance(image, np.ndarray)
        if self.magnitude > 0:
            image = self.distortion(image, self.magnitude)

        if self.transform is not None:
            image = self.transform(image)

        item[self.image_id] = image
        return tuple(item)

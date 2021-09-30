from torch.utils.data import Dataset
from functools import partial
import PIL
import cv2
import numpy as np

rotation_interp = cv2.INTER_LINEAR
rotation_border_mode = cv2.BORDER_REFLECT


def distortion_rotate(image, magnitude):
    angle = [0, 10, 20, 30, 40, 50][magnitude]
    h,w,c = image.shape
    image_center = (h//2, w//2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, (w, h), flags=rotation_interp, borderMode=rotation_border_mode)
    return image

def distortion_jpeg_compression(image, magnitude):
    quality = [100, 80, 60, 40, 20, 10][magnitude]

    img_encode = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    image = cv2.imdecode( img_encode , cv2.IMREAD_COLOR)
    return image

def distortion_blur(image, magnitude):
    h,w,c = image.shape
    max_wh = max(w,h)
    k_size = int([0, max_wh/100, max_wh/90, max_wh/80, max_wh/70, max_wh/60][magnitude])
    image = cv2.blur(image, ksize=(k_size, k_size))
    return image

def distortion_brightness(image, magnitude, high=True):
    if high:
        base = (255-image.mean())
    else:
        base = image.mean()
    delta_b = int([0, base*0.15, base*0.30, base*0.45, base*0.6, base*0.75][magnitude])

    if high:
        image = np.array(image + delta_b)
    else:
        image = np.array(image - delta_b)
    
    return np.clip(image,0,255).astype(np.uint8)



def distortion_high_brightness(image, magnitude):
    return distortion_brightness(image, magnitude, high=True)

def distortion_low_brightness(image, magnitude):
    return distortion_brightness(image, magnitude, high=False)

class RobustnessDataset(Dataset):
    def __init__(self, dataset, distortion_str="rotate_c", magnitude=1, image_id=0, transform=None, resize=None, resize_interp=cv2.INTER_LINEAR):
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
        if distortion_str == "rotate_c":
            return distortion_rotate
        elif distortion_str == "jpeg_compression":
            return distortion_jpeg_compression
        elif distortion_str == "blur":
            return distortion_blur
        elif distortion_str == "high_brightness":
            return distortion_high_brightness
        elif distortion_str == "low_brightness":
            return distortion_low_brightness
        else:
            raise ValueError("Invalid distortion_str")

    def __getitem__(self, index):
        item = list(self.dataset[index])
        image = item[self.image_id]
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=self.resize_interp)

        assert isinstance(image, np.ndarray)
        if self.magnitude > 0:
            image = self.distortion(image, self.magnitude)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        item[self.image_id] = image
        return tuple(item)
    

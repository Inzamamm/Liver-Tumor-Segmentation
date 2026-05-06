import cv2
import numpy as np


def intensity_clip(image: np.ndarray, window_min: float = -200, window_max: float = 250) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image, window_min, window_max)


def zscore_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + eps)


def minmax_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return (image - image.min()) / (image.max() - image.min() + eps)


def resize_image(image: np.ndarray, size: int = 256, is_mask: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def preprocess_ct(image: np.ndarray, size: int = 256, window_min: float = -200, window_max: float = 250) -> np.ndarray:
    image = intensity_clip(image, window_min, window_max)
    image = zscore_normalize(image)
    image = minmax_normalize(image)
    image = resize_image(image, size=size, is_mask=False)
    return image.astype(np.float32)


def preprocess_mask(mask: np.ndarray, size: int = 256) -> np.ndarray:
    mask = resize_image(mask, size=size, is_mask=True)
    mask = (mask > 0).astype(np.float32)
    return mask

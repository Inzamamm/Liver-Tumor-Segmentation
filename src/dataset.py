import os
from glob import glob
from typing import List, Tuple
import cv2
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_ct, preprocess_mask

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy', '.nii', '.nii.gz')


def list_files(folder: str) -> List[str]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob(os.path.join(folder, f'*{ext}')))
    return sorted(files)


def base_name(path: str) -> str:
    name = os.path.basename(path)
    name = name.replace('.nii.gz', '')
    name = os.path.splitext(name)[0]
    return name


def pair_image_masks(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    images = list_files(image_dir)
    masks = list_files(mask_dir)
    mask_map = {base_name(m): m for m in masks}
    pairs = []
    for img in images:
        key = base_name(img)
        if key in mask_map:
            pairs.append((img, mask_map[key]))
    if not pairs:
        raise FileNotFoundError('No image-mask pairs found. Ensure matching filenames in image_dir and mask_dir.')
    return pairs


def load_array(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        arr = np.load(path)
    elif path.endswith('.nii') or path.endswith('.nii.gz'):
        arr = nib.load(path).get_fdata()
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f'Unable to read file: {path}')
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.asarray(arr)


class LiverTumorDataset(Dataset):
    def __init__(self, pairs, image_size=256, window_min=-200, window_max=250, augment=False):
        self.pairs = pairs
        self.image_size = image_size
        self.window_min = window_min
        self.window_max = window_max
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, image, mask):
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = load_array(image_path)
        mask = load_array(mask_path)

        if image.ndim == 3:
            image = image[:, :, image.shape[2] // 2]
        if mask.ndim == 3:
            mask = mask[:, :, mask.shape[2] // 2]

        image = preprocess_ct(image, self.image_size, self.window_min, self.window_max)
        mask = preprocess_mask(mask, self.image_size)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, os.path.basename(image_path)


def make_splits(pairs, train_split=0.70, val_split=0.15, test_split=0.15, seed=42):
    train_pairs, temp_pairs = train_test_split(pairs, train_size=train_split, random_state=seed, shuffle=True)
    relative_val = val_split / (val_split + test_split)
    val_pairs, test_pairs = train_test_split(temp_pairs, train_size=relative_val, random_state=seed, shuffle=True)
    return train_pairs, val_pairs, test_pairs

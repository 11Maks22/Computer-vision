from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .baselines import make_baselines_from_hr
from .config import COLOR_Y_CONFIG, GRAYSCALE_CONFIG, SEED
from .metrics import y_channel


class PairDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class SplitData:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def load_olivetti_faces_dataset() -> np.ndarray:
    data = fetch_olivetti_faces(shuffle=True, random_state=SEED)
    return data.images.astype(np.float32)


def make_grayscale_pairs(faces: np.ndarray, lr_size: int, hr_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for face in faces:
        _, baselines = make_baselines_from_hr(face, lr_size=lr_size, hr_size=hr_size)
        x_list.append(baselines["bicubic"])
        y_list.append(face)
    x = np.asarray(x_list, dtype=np.float32)[:, None, :, :]
    y = np.asarray(y_list, dtype=np.float32)[:, None, :, :]
    return x, y


def split_grayscale_data() -> SplitData:
    cfg = GRAYSCALE_CONFIG
    faces = load_olivetti_faces_dataset()
    x, y = make_grayscale_pairs(faces, lr_size=cfg["lr_size"], hr_size=cfg["hr_size"])
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=cfg["test_size"],
        random_state=SEED,
        shuffle=True,
    )
    return SplitData(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)


def make_color_y_pairs(images_rgb: list[np.ndarray], lr_size: int, hr_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for image_rgb in images_rgb:
        y = y_channel(image_rgb)
        _, baselines = make_baselines_from_hr(y, lr_size=lr_size, hr_size=hr_size)
        x_list.append(baselines["bicubic"])
        y_list.append(y)
    x = np.asarray(x_list, dtype=np.float32)[:, None, :, :]
    y = np.asarray(y_list, dtype=np.float32)[:, None, :, :]
    return x, y


def split_color_data(images_rgb: list[np.ndarray]) -> tuple[SplitData, list[np.ndarray], list[np.ndarray]]:
    cfg = COLOR_Y_CONFIG
    split_idx = max(1, int(len(images_rgb) * cfg["train_fraction"]))
    train_rgb = images_rgb[:split_idx]
    val_rgb = images_rgb[split_idx:] or images_rgb[-2:]
    x_train, y_train = make_color_y_pairs(train_rgb, lr_size=cfg["lr_size"], hr_size=cfg["hr_size"])
    x_val, y_val = make_color_y_pairs(val_rgb, lr_size=cfg["lr_size"], hr_size=cfg["hr_size"])
    return SplitData(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val), train_rgb, val_rgb


def make_dataloader_pair(split: SplitData, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(PairDataset(split.x_train, split.y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(split.x_val, split.y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_gray_loaders() -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    split = split_grayscale_data()
    train_loader, val_loader = make_dataloader_pair(split, batch_size=GRAYSCALE_CONFIG["batch_size"])
    return train_loader, val_loader, split.x_val, split.y_val


def get_color_loaders(train_rgb: list[np.ndarray], val_rgb: list[np.ndarray]) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    x_train, y_train = make_color_y_pairs(train_rgb, lr_size=COLOR_Y_CONFIG["lr_size"], hr_size=COLOR_Y_CONFIG["hr_size"])
    x_val, y_val = make_color_y_pairs(val_rgb, lr_size=COLOR_Y_CONFIG["lr_size"], hr_size=COLOR_Y_CONFIG["hr_size"])
    split = SplitData(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)
    train_loader, val_loader = make_dataloader_pair(split, batch_size=COLOR_Y_CONFIG["batch_size"])
    return train_loader, val_loader, split.x_val, split.y_val

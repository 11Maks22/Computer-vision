from __future__ import annotations

import cv2
import numpy as np


def create_lr(image: np.ndarray, lr_size: int) -> np.ndarray:
    """Create a low-resolution version using area interpolation."""
    return cv2.resize(image, (lr_size, lr_size), interpolation=cv2.INTER_AREA)


def upsample(image: np.ndarray, hr_size: int, method: str) -> np.ndarray:
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    if method not in interpolation_map:
        raise ValueError(f"Unknown interpolation method: {method}")
    out = cv2.resize(image, (hr_size, hr_size), interpolation=interpolation_map[method])
    return np.clip(out, 0.0, 1.0)


def make_baselines_from_hr(image: np.ndarray, lr_size: int, hr_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create LR image plus standard interpolation baselines from HR target."""
    lr = create_lr(image, lr_size)
    baselines = {
        "nearest": upsample(lr, hr_size, "nearest"),
        "bilinear": upsample(lr, hr_size, "bilinear"),
        "bicubic": upsample(lr, hr_size, "bicubic"),
    }
    return lr, baselines

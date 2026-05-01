from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse_score(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((target.astype(np.float32) - pred.astype(np.float32)) ** 2))


def psnr_score(target: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def ssim_score(target: np.ndarray, pred: np.ndarray) -> float:
    if target.ndim == 2:
        return float(structural_similarity(target, pred, data_range=1.0))
    return float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))


def y_channel(image_rgb: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32) / 255.0


def summarize_metrics(df: pd.DataFrame, group_col: str = "method", sort_col: str = "psnr") -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {group_col, "sample_index", "image_name"}]
    summary = df.groupby(group_col, as_index=False)[numeric_cols].mean()
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return summary

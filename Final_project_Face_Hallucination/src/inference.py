from __future__ import annotations

import cv2
import numpy as np
import torch

from .baselines import make_baselines_from_hr
from .config import COLOR_Y_CONFIG, GRAYSCALE_CONFIG
from .metrics import y_channel
from .models import SRCNN
from .preprocessing import preprocess_color_image_file
from .train_utils import get_device


def _load_model(model_path, in_channels: int = 1, out_channels: int = 1) -> SRCNN:
    model = SRCNN(in_channels=in_channels, out_channels=out_channels)
    state = torch.load(model_path, map_location=get_device())
    model.load_state_dict(state)
    model.eval()
    return model.to(get_device())


def load_grayscale_model() -> SRCNN:
    return _load_model(GRAYSCALE_CONFIG['model_path'])


def load_color_model() -> SRCNN:
    return _load_model(COLOR_Y_CONFIG['model_path'])


def predict_grayscale(image_gray: np.ndarray) -> np.ndarray:
    cfg = GRAYSCALE_CONFIG
    model = load_grayscale_model()
    _, baselines = make_baselines_from_hr(image_gray, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
    inp = torch.from_numpy(baselines['bicubic'][None, None, ...]).float().to(get_device())
    with torch.no_grad():
        pred = model(inp).cpu().numpy()[0, 0]
    return np.clip(pred, 0.0, 1.0)


def predict_color_array(image_rgb: np.ndarray) -> np.ndarray:
    cfg = COLOR_Y_CONFIG
    model = load_color_model()
    y = y_channel(image_rgb)
    _, base_y = make_baselines_from_hr(y, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
    inp = torch.from_numpy(base_y['bicubic'][None, None, ...]).float().to(get_device())
    with torch.no_grad():
        pred_y = model(inp).cpu().numpy()[0, 0]
    ycrcb = cv2.cvtColor(np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32) / 255.0
    cb = ycrcb[:, :, 2].astype(np.float32) / 255.0
    _, base_cr = make_baselines_from_hr(cr, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
    _, base_cb = make_baselines_from_hr(cb, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
    merged = cv2.merge([
        np.clip(pred_y * 255.0, 0, 255).astype(np.uint8),
        np.clip(base_cr['bicubic'] * 255.0, 0, 255).astype(np.uint8),
        np.clip(base_cb['bicubic'] * 255.0, 0, 255).astype(np.uint8),
    ])
    pred_rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return np.clip(pred_rgb, 0.0, 1.0)


def predict_color_from_file(image_path: str) -> np.ndarray:
    image_rgb = preprocess_color_image_file(image_path, output_size=COLOR_Y_CONFIG['hr_size'])
    if image_rgb is None:
        raise ValueError(f'Could not read image: {image_path}')
    return predict_color_array(image_rgb)

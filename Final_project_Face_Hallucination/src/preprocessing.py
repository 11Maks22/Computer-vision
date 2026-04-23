from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade for face detection.")
    return detector


def normalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def rgb_to_ycrcb(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(denormalize_image(image_rgb), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0


def ycrcb_to_rgb(image_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(denormalize_image(image_ycrcb), cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0


def make_square_crop(image: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.2) -> np.ndarray:
    ih, iw = image.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2
    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(iw, sx1 + side)
    sy2 = min(ih, sy1 + side)
    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)
    return image[sy1:sy2, sx1:sx2]


def central_face_crop(image_rgb: np.ndarray, output_size: int = 128) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    y1, y2 = int(h * 0.1), int(h * 0.9)
    x1, x2 = int(w * 0.1), int(w * 0.9)
    crop = image_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return normalize_image(crop)


def detect_and_crop_face(image_rgb: np.ndarray, output_size: int = 128) -> np.ndarray:
    detector = get_face_detector()
    gray = cv2.cvtColor(denormalize_image(image_rgb), cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return central_face_crop(image_rgb, output_size=output_size)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = make_square_crop(image_rgb, x, y, w, h, pad_ratio=0.2)
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return normalize_image(crop)


def preprocess_color_image_file(path: str | Path, output_size: int = 128) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return detect_and_crop_face(img, output_size=output_size)

from __future__ import annotations

from pathlib import Path
from .config import (
    COLOR_RESULTS_DIR,
    GRAYSCALE_RESULTS_DIR,
    HISTORY_DIR,
    INPUT_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from .preprocessing import preprocess_color_image_file


def ensure_folders() -> None:
    for path in [INPUT_DIR, GRAYSCALE_RESULTS_DIR, COLOR_RESULTS_DIR, METRICS_DIR, MODELS_DIR, HISTORY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_color_images(max_images: int | None = None, output_size: int = 128):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(sorted(INPUT_DIR.glob(ext)))

    images = []
    names = []
    for path in image_paths[:max_images]:
        img = preprocess_color_image_file(path, output_size=output_size)
        if img is None:
            continue
        images.append(img)
        names.append(path.stem)
    return images, names

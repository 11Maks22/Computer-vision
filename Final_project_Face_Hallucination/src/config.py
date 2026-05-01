from __future__ import annotations

from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
GRAYSCALE_DATA_DIR = DATA_DIR / "grayscale"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
GRAYSCALE_RESULTS_DIR = RESULTS_DIR / "grayscale"
COLOR_RESULTS_DIR = RESULTS_DIR / "color_y"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
HISTORY_DIR = OUTPUTS_DIR / "history"
NOTEBOOK_DIR = PROJECT_DIR / "notebook"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3

GRAYSCALE_CONFIG = {
    "name": "grayscale",
    "hr_size": 64,
    "lr_size": 16,
    "test_size": 0.2,
    "epochs": 20,
    "batch_size": 16,
    "max_eval_images": 20,
    "model_path": MODELS_DIR / "grayscale_srcnn.pt",
    "history_path": HISTORY_DIR / "grayscale_history.csv",
    "all_metrics_path": METRICS_DIR / "grayscale_all_metrics.csv",
    "summary_metrics_path": METRICS_DIR / "grayscale_summary_metrics.csv",
    "comparison_figure_path": GRAYSCALE_RESULTS_DIR / "comparison_example.png",
    "barplot_path": METRICS_DIR / "grayscale_psnr_barplot.png",
}

COLOR_Y_CONFIG = {
    "name": "color_y",
    "hr_size": 128,
    "lr_size": 32,
    "train_fraction": 0.8,
    "epochs": 25,
    "batch_size": 8,
    "max_images": 50,
    "min_images": 3,
    "model_path": MODELS_DIR / "color_y_srcnn.pt",
    "history_path": HISTORY_DIR / "color_y_history.csv",
    "all_metrics_path": METRICS_DIR / "color_all_metrics.csv",
    "summary_metrics_path": METRICS_DIR / "color_summary_metrics.csv",
    "barplot_path": METRICS_DIR / "color_psnr_barplot.png",
}

EXPERIMENTS = {
    "grayscale": GRAYSCALE_CONFIG,
    "color_y": COLOR_Y_CONFIG,
}

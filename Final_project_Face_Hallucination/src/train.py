from __future__ import annotations

from .experiment import run_color_y_experiment, run_grayscale_experiment
from .io_utils import ensure_folders
from .train_utils import set_seed


def run_training_pipeline() -> None:
    set_seed()
    ensure_folders()
    print("Running grayscale CNN experiment...")
    run_grayscale_experiment()
    print("\nRunning color CNN experiment...")
    run_color_y_experiment()
    print("\nDone. Check outputs/results, outputs/metrics, outputs/history and outputs/models.")


if __name__ == "__main__":
    run_training_pipeline()

# Final Project: Learning-Based Face Super-Resolution

## Objective
This project studies learning-based face super-resolution using a shallow SRCNN model and compares it against standard interpolation baselines.

The goal is to learn a mapping from low-resolution facial images to high-resolution targets and evaluate the results both quantitatively and visually.

Two experimental settings are included:
1. Grayscale face hallucination (Olivetti dataset)
2. Color face hallucination (Y-channel super-resolution)

---

## Technical Project Structure

Final_project_Face_Hallucination/
├── notebook/
│   └── Final_Project_Face_Hallucination.ipynb
├── src/
│   ├── baselines.py
│   ├── config.py
│   ├── data_utils.py
│   ├── experiment.py
│   ├── inference.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train.py
│   └── train_utils.py
├── data/
│   └── input/
├── outputs/
│   ├── history/
│   ├── metrics/
│   ├── models/
│   └── results/
│       ├── grayscale/
│       └── color_y/
├── requirements.txt
└── run.py

---

## Methods

### Baselines
- Nearest neighbor interpolation
- Bilinear interpolation
- Bicubic interpolation

### Learned Model
- SRCNN (3 convolutional layers)
- MSE loss
- Adam optimizer
- Best model selected by validation loss

---

## Evaluation

The project computes:
- MSE
- PSNR
- SSIM
- Y-channel PSNR/SSIM (for color experiment)

Also generated:
- training history (loss curves)
- visual comparison results
- summary metrics tables

---

## Installation

Recommended Python version:
Python 3.10–3.11

### Windows

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

### Linux / macOS

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

---

## Running the Project

Run full pipeline:

python run.py

Or:

python -m src.train

---

## Color Experiment Input

Add at least 1–3 face images into:

data/input/

Supported formats:
.jpg, .jpeg, .png, .bmp

---

## Important Notes

- This is not a reproduction of the classical face hallucination method based on steerable pyramids and Bayesian MAP.
- This is a CNN-based technical coursework project in the same domain.
- Color reconstruction is implemented as Y-channel super-resolution (luminance only), not full RGB end-to-end reconstruction.

---

## Conclusion (Short)

The project demonstrates how a convolutional neural network can learn a mapping from low-resolution to high-resolution images and how its performance compares to classical interpolation methods.

Grayscale reconstruction shows stable results, while the color experiment highlights limitations of Y-channel-based approaches.

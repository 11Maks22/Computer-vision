from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .baselines import make_baselines_from_hr
from .config import COLOR_RESULTS_DIR, COLOR_Y_CONFIG, GRAYSCALE_CONFIG, GRAYSCALE_RESULTS_DIR
from .data_utils import get_color_loaders, get_gray_loaders
from .io_utils import load_color_images
from .metrics import mse_score, psnr_score, ssim_score, summarize_metrics, y_channel
from .models import SRCNN
from .train_utils import get_device, train_model


def _save_barplot(summary: pd.DataFrame, path, title: str, column: str = 'psnr') -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(summary['method'], summary[column])
    plt.title(title)
    plt.ylabel(column.upper())
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def run_grayscale_experiment() -> pd.DataFrame:
    cfg = GRAYSCALE_CONFIG
    print('Preparing grayscale data...')
    train_loader, val_loader, x_val, y_val = get_gray_loaders()
    print('Training grayscale SRCNN...')
    model, history = train_model(
        SRCNN(),
        train_loader,
        val_loader,
        epochs=cfg['epochs'],
        history_path=cfg['history_path'],
    )
    torch.save(model.state_dict(), cfg['model_path'])
    device = get_device()
    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in range(min(len(x_val), cfg['max_eval_images'])):
            input_tensor = torch.from_numpy(x_val[idx:idx + 1]).float().to(device)
            target = y_val[idx, 0]
            prediction = model(input_tensor).cpu().numpy()[0, 0]
            _, baselines = make_baselines_from_hr(target, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
            methods = {
                'nearest': baselines['nearest'],
                'bilinear': baselines['bilinear'],
                'bicubic': baselines['bicubic'],
                'proposed_cnn': prediction,
            }
            for method_name, pred in methods.items():
                rows.append({
                    'sample_index': idx,
                    'method': method_name,
                    'mse': mse_score(target, pred),
                    'psnr': psnr_score(target, pred),
                    'ssim': ssim_score(target, pred),
                })
            if idx == 0:
                lr = cv2.resize(target, (cfg['lr_size'], cfg['lr_size']), interpolation=cv2.INTER_AREA)
                images = [
                    ('Ground Truth', target),
                    ('Low Resolution', lr),
                    ('Bicubic', baselines['bicubic']),
                    ('SRCNN', prediction),
                ]
                plt.figure(figsize=(10, 3))
                for i, (title, image) in enumerate(images, start=1):
                    plt.subplot(1, 4, i)
                    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
                    plt.title(title)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(cfg['comparison_figure_path'], dpi=200, bbox_inches='tight')
                plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(cfg['all_metrics_path'], index=False)
    summary = summarize_metrics(df)
    summary.to_csv(cfg['summary_metrics_path'], index=False)
    _save_barplot(summary, cfg['barplot_path'], 'Grayscale PSNR Comparison')
    print('Grayscale evaluation completed.')
    print(summary)
    return summary


def run_color_y_experiment() -> pd.DataFrame | None:
    cfg = COLOR_Y_CONFIG
    images_rgb, names = load_color_images(max_images=cfg['max_images'], output_size=cfg['hr_size'])
    if len(images_rgb) < cfg['min_images']:
        print(f"Color evaluation skipped: add at least {cfg['min_images']} images to data/input/.")
        return None

    split_idx = max(1, int(len(images_rgb) * cfg['train_fraction']))
    train_rgb = images_rgb[:split_idx]
    val_rgb = images_rgb[split_idx:] or images_rgb[-2:]
    val_names = names[split_idx:] or names[-2:]

    print('Preparing color Y-channel data...')
    train_loader, val_loader, x_val, y_val = get_color_loaders(train_rgb, val_rgb)
    print('Training color Y-channel SRCNN...')
    model, history = train_model(
        SRCNN(),
        train_loader,
        val_loader,
        epochs=cfg['epochs'],
        history_path=cfg['history_path'],
    )
    torch.save(model.state_dict(), cfg['model_path'])
    device = get_device()
    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for name, image_rgb, y_input, y_target in zip(val_names, val_rgb, x_val, y_val):
            input_tensor = torch.from_numpy(y_input[None, ...]).float().to(device)
            pred_y = model(input_tensor).cpu().numpy()[0, 0]
            target_y = y_target[0]
            ycrcb = cv2.cvtColor(np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            cr = ycrcb[:, :, 1].astype(np.float32) / 255.0
            cb = ycrcb[:, :, 2].astype(np.float32) / 255.0
            _, base_cr = make_baselines_from_hr(cr, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
            _, base_cb = make_baselines_from_hr(cb, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
            _, base_y = make_baselines_from_hr(target_y, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'])
            baselines_rgb = {}
            for method in ('nearest', 'bilinear', 'bicubic'):
                merged = cv2.merge([
                    np.clip(base_y[method] * 255.0, 0, 255).astype(np.uint8),
                    np.clip(base_cr[method] * 255.0, 0, 255).astype(np.uint8),
                    np.clip(base_cb[method] * 255.0, 0, 255).astype(np.uint8),
                ])
                baselines_rgb[method] = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
            merged_pred = cv2.merge([
                np.clip(pred_y * 255.0, 0, 255).astype(np.uint8),
                np.clip(base_cr['bicubic'] * 255.0, 0, 255).astype(np.uint8),
                np.clip(base_cb['bicubic'] * 255.0, 0, 255).astype(np.uint8),
            ])
            proposed_rgb = cv2.cvtColor(merged_pred, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
            methods = {
                'nearest': baselines_rgb['nearest'],
                'bilinear': baselines_rgb['bilinear'],
                'bicubic': baselines_rgb['bicubic'],
                'proposed_cnn_y': proposed_rgb,
            }
            for method_name, pred in methods.items():
                rows.append({
                    'image_name': name,
                    'method': method_name,
                    'mse': mse_score(image_rgb, pred),
                    'psnr': psnr_score(image_rgb, pred),
                    'ssim': ssim_score(image_rgb, pred),
                    'y_psnr': psnr_score(y_channel(image_rgb), y_channel(pred)),
                    'y_ssim': ssim_score(y_channel(image_rgb), y_channel(pred)),
                })
            lr_rgb = cv2.resize(image_rgb, (cfg['lr_size'], cfg['lr_size']), interpolation=cv2.INTER_AREA)
            lr_up = cv2.resize(lr_rgb, (cfg['hr_size'], cfg['hr_size']), interpolation=cv2.INTER_CUBIC)
            images = [('Bicubic', baselines_rgb['bicubic']), ('SRCNN (Y)', proposed_rgb), ('Ground Truth', image_rgb)]
            plt.figure(figsize=(9, 3))
            for i, (title, image) in enumerate(images, start=1):
                plt.subplot(1, 3, i)
                plt.imshow(np.clip(image, 0.0, 1.0))
                plt.title(title)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(COLOR_RESULTS_DIR / f'{name}_comparison.png', dpi=200, bbox_inches='tight')
            plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(cfg['all_metrics_path'], index=False)
    summary = summarize_metrics(df)
    summary.to_csv(cfg['summary_metrics_path'], index=False)
    _save_barplot(summary, cfg['barplot_path'], 'Color PSNR Comparison')
    print('Color evaluation completed.')
    print(summary)
    return summary

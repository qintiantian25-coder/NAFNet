#!/usr/bin/env python3
"""Evaluate full-frame and blind-pixel metrics for restoration outputs.

Usage example:
  python tools/eval_blind.py \
    --gt_dir /root/Qtt/Restormer/data/test_sharp \
    --input_dir /root/Qtt/Restormer/data/test_blur \
    --output_dir experiments/NAFNet_blind_test/results \
    --mask_csv /root/Qtt/Restormer/data/test_mask/001/blind_pixel_coords.csv
"""
import os
import re
import csv
import argparse
import numpy as np
import cv2
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    if not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def map_images(folder, ext='.png'):
    m = {}
    if not os.path.exists(folder):
        return m
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(ext):
                m[f] = os.path.join(root, f)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--input_dir', required=False, default=None,
                        help='optional: original blind inputs (for input blind metrics)')
    parser.add_argument('--output_dir', required=True, help='restored images')
    parser.add_argument('--mask_csv', required=False, default=None,
                        help='csv of blind coords (x,y). If omitted, blind metrics skipped')
    parser.add_argument('--ext', default='.png')
    parser.add_argument('--save_dir', default=None, help='where to save per-image csv (default: output_dir/blind_eval)')
    args = parser.parse_args()

    out_imgs = sorted([f for f in os.listdir(args.output_dir) if f.lower().endswith(args.ext)], key=natural_sort_key)
    gt_map = map_images(args.gt_dir, ext=args.ext)
    input_map = map_images(args.input_dir, ext=args.ext) if args.input_dir else {}

    blind_coords = load_blind_coords(args.mask_csv) if args.mask_csv else None

    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []

    print(f"===> 开始定量打分，准备比对 {len(out_imgs)} 张图片...")
    for img_name in out_imgs:
        out_path = os.path.join(args.output_dir, img_name)
        gt_path = gt_map.get(img_name)
        if gt_path and os.path.exists(out_path):
            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None and out_img is not None:
                if out_img.shape != gt_img.shape:
                    out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

                # full-frame metrics (use 0 crop_border)
                try:
                    full_psnr = float(calculate_psnr(gt_img, out_img, crop_border=0, input_order='HWC', test_y_channel=False))
                except Exception:
                    full_psnr = None
                try:
                    full_ssim = float(calculate_ssim(gt_img, out_img, crop_border=0, input_order='HWC', test_y_channel=False))
                except Exception:
                    full_ssim = None

                row = {
                    'image': img_name,
                    'psnr': full_psnr,
                    'ssim': full_ssim,
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'blind_mae_input': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                    'blind_count': 0
                }

                # blind-pixel metrics
                if blind_coords is not None:
                    h, w = gt_img.shape[:2]
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]

                        gt_vals = gt_img[y, x].astype(np.float64)
                        out_vals = out_img[y, x].astype(np.float64)
                        err = out_vals - gt_vals

                        blind_abs = np.abs(err)
                        blind_sq = err ** 2

                        blind_abs_sum += float(blind_abs.sum())
                        blind_sq_sum += float(blind_sq.sum())
                        blind_pix_sum += int(len(err))

                        in_path = input_map.get(img_name)
                        in_mae = None
                        if in_path and os.path.exists(in_path):
                            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                            if in_img is not None:
                                if in_img.shape != gt_img.shape:
                                    in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                                in_vals = in_img[y, x].astype(np.float64)
                                in_err = in_vals - gt_vals
                                in_abs = np.abs(in_err)
                                in_sq = in_err ** 2
                                blind_abs_in_sum += float(in_abs.sum())
                                blind_sq_in_sum += float(in_sq.sum())
                                in_mae = float(in_abs.mean())

                        row.update({
                            'blind_mae': float(blind_abs.mean()),
                            'blind_rmse': float(np.sqrt(blind_sq.mean())),
                            'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                            'blind_mae_input': in_mae,
                            'blind_count': int(len(err))
                        })
                        if in_mae is not None:
                            row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                            row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)
                per_image_logs.append(row)

    # print summary
    # aggregate blind metrics
    if blind_coords is not None and blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))

        print("===> Blind-Pixel Focused Metrics")
        print(f"BlindCount(total sampled): {blind_pix_sum}")
        print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}")

        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse_in, 1e-12)))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(f"Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)")

    # save per-image csv
    save_blind_dir = args.save_dir if args.save_dir is not None else os.path.join(args.output_dir, 'blind_eval')
    os.makedirs(save_blind_dir, exist_ok=True)
    save_blind_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
        with open(save_blind_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_logs:
                writer.writerow(row)
        print(f"Per-image test metrics saved to: {save_blind_csv}")


if __name__ == '__main__':
    main()

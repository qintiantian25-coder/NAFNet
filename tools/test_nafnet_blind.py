#!/usr/bin/env python3
"""
测试 NAFNet 模型（支持盲元评估）
参考 SwinIR 测试脚本改写，适配 NAFNet 结构。
支持灰度图输入/输出，合并静态盲元和闪光盲元，输出分组评估指标和对比图。

Usage:
    python test_nafnet_blind.py \
        --data_root /path/to/data_new \
        --checkpoint /path/to/net_g.pth \
        --in_chans 1 \
        --save_dir results/NAFNet_test \
        --test_mask_csv /path/to/test_mask \
        --image_border 0
"""

import os
import argparse
import re
import csv
import cv2
import numpy as np
import torch
from collections import defaultdict

# 添加 NAFNet 路径（根据你的实际项目路径修改）
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from basicsr.models.archs.NAFNet_arch import NAFNetLocal   # 或者 NAFNet

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_blind_coords(csv_path):
    """加载静态盲元坐标 CSV，返回 Nx2 numpy 数组（x, y）"""
    if not csv_path or not os.path.exists(csv_path):
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

def load_flash_map(csv_path):
    """加载闪光盲元 CSV（带 frame_name 列），返回 {frame_name: [(x,y),...]}"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        if 'frame_name' not in reader.fieldnames or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                fname = os.path.basename(str(row['frame_name']))
                x = int(float(row['x']))
                y = int(float(row['y']))
            except Exception:
                continue
            flash_map.setdefault(fname, set()).add((x, y))
    normalized = {k: list(v) for k, v in flash_map.items()}
    return normalized

def resolve_csv_path(csv_path, data_root):
    """解析 mask 路径（可能是目录或文件）"""
    if not csv_path:
        return None
    if os.path.isdir(csv_path):
        return csv_path
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path
    if os.path.exists(csv_path):
        return csv_path
    if data_root:
        candidate = os.path.join(data_root, csv_path)
        if os.path.exists(candidate):
            return candidate
    return csv_path

def resolve_group_mask_paths(mask_base_path, data_root, group_name):
    """根据组名（如 001）获取该组对应的 blind_coords.csv 和 flash CSV 路径"""
    candidates = []
    if mask_base_path:
        if os.path.isdir(mask_base_path):
            group_dir = os.path.join(mask_base_path, group_name)
            blind_csv_candidates = [
                os.path.join(group_dir, 'blind_coords.csv'),
                os.path.join(group_dir, 'blind_pixel_coords.csv'),
            ]
            flash_csv = os.path.join(group_dir, 'flash_pixel_coords.csv')
        else:
            candidates.append(mask_base_path)
            if data_root and not os.path.isabs(mask_base_path):
                candidates.append(os.path.join(data_root, mask_base_path))
            base_dir = os.path.dirname(mask_base_path)
            if base_dir:
                candidates.append(os.path.join(base_dir, group_name, os.path.basename(mask_base_path)))
            blind_csv_candidates = candidates
            flash_csv = os.path.join(base_dir, group_name, 'flash_pixel_coords.csv') if base_dir else None
    else:
        blind_csv_candidates = []
        flash_csv = None

    if data_root:
        default_group_dir = os.path.join(data_root, 'test_mask', group_name)
        blind_csv_candidates.extend([
            os.path.join(default_group_dir, 'blind_coords.csv'),
            os.path.join(default_group_dir, 'blind_pixel_coords.csv'),
        ])
        if flash_csv is None:
            flash_csv = os.path.join(default_group_dir, 'flash_pixel_coords.csv')

    seen = set()
    blind_csv = None
    for candidate in blind_csv_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            blind_csv = candidate
            break

    if flash_csv and not os.path.exists(flash_csv):
        flash_csv = None

    return {'blind_csv': blind_csv, 'flash_csv': flash_csv}

def get_group_name(rel_path):
    """从相对路径中提取组名（第一级目录）"""
    parts = os.path.normpath(rel_path).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return 'root'

# ----------------------------------------------------------------------
# 模型构建（NAFNet）
# ----------------------------------------------------------------------
def build_model(device, in_chans=1, width=64, enc_blk_nums=[1,1,1,28],
                middle_blk_num=1, dec_blk_nums=[1,1,1,1]):
    """
    构建 NAFNetLocal 模型（与你测试配置 NAFNet-blind.yml 一致）
    可根据需要修改为 NAFNet 完整版
    """
    model = NAFNetLocal(
        img_channel=in_chans,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
        # NAFNetLocal internally runs a dummy forward in __init__ via convert().
        # Keep channel count consistent with in_chans to avoid 1ch/3ch mismatch.
        train_size=(1, in_chans, 256, 256)
    )
    return model.to(device)

# ----------------------------------------------------------------------
# 图像预处理/后处理
# ----------------------------------------------------------------------
def to_tensor_gray(img_gray, in_chans=1):
    """
    将灰度图（H x W, uint8 或 float [0,255]）转换为模型输入的 tensor。
    - 若 in_chans == 1: 输出 (1, H, W) float32 [0,1]
    - 若 in_chans == 3: 输出 (3, H, W) float32 [0,1] (灰度复制三通道)
    """
    if img_gray.dtype != np.float32:
        img = img_gray.astype(np.float32)
    else:
        img = img_gray
    if img.max() > 1.0:
        img = img / 255.0
    if in_chans == 3:
        img3 = np.stack([img, img, img], axis=2)   # H, W, 3
        img3 = img3.transpose(2, 0, 1)             # 3, H, W
        return img3
    else:  # in_chans == 1
        return img[np.newaxis, ...]                # 1, H, W

def tensor_to_uint8_gray(out_tensor):
    """将模型输出的单通道 tensor [0,1] 转换为 uint8 灰度图"""
    out_np = out_tensor.squeeze(0).detach().cpu().numpy()
    if out_np.ndim == 3 and out_np.shape[0] == 1:
        out_np = out_np[0]
    elif out_np.ndim == 3 and out_np.shape[0] != 1:
        raise ValueError(f'Expected 1-channel output, got shape {out_np.shape}')
    # 确保值在 [0,1] 范围内
    out_np = np.clip(out_np, 0, 1)
    return (out_np * 255).round().astype(np.uint8)

# ----------------------------------------------------------------------
# 评估报告类（简单记录全图指标）
# ----------------------------------------------------------------------
class TestReport:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border
        self.total_rgb_psnr = []
        self.total_ssim = []

    def update_metric(self, gt_img, out_img, img_name=None):
        from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
        # use HWC order (or let the function infer from 2D images); 'HW' is invalid
        psnr = calculate_psnr(out_img, gt_img, crop_border=self.crop_border, input_order='HWC')
        ssim = calculate_ssim(out_img, gt_img, crop_border=self.crop_border, input_order='HWC')
        self.total_rgb_psnr.append(float(psnr))
        self.total_ssim.append(float(ssim))

    def print_final_result(self):
        if len(self.total_rgb_psnr) == 0:
            print('No valid images were evaluated.')
            return
        print(f'Average PSNR: {np.mean(self.total_rgb_psnr):.4f} dB')
        print(f'Average SSIM: {np.mean(self.total_ssim):.4f}')

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录，包含 test_sharp, test_blur 等')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件 (.pth)')
    parser.add_argument('--in_chans', type=int, default=1, help='模型输入通道数 (1 或 3)')
    parser.add_argument('--save_dir', type=str, default='results/nafnet_test', help='输出根目录')
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu')
    parser.add_argument('--test_mask_csv', type=str, default=None, help='盲元 CSV 路径或 test_mask 目录')
    parser.add_argument('--image_border', type=int, default=0, help='PSNR/SSIM 计算时裁剪边界')
    # 可选网络参数（覆盖默认值）
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--enc_blk_nums', type=str, default='1,1,1,28')
    parser.add_argument('--middle_blk_num', type=int, default=1)
    parser.add_argument('--dec_blk_nums', type=str, default='1,1,1,1')
    args = parser.parse_args()

    # 解析列表参数
    enc_blk_nums = [int(x) for x in args.enc_blk_nums.split(',')]
    dec_blk_nums = [int(x) for x in args.dec_blk_nums.split(',')]

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_triple = os.path.join(args.save_dir, 'triple_comparison')
    save_pure = os.path.join(args.save_dir, 'test')
    save_blind_dir = os.path.join(args.save_dir, 'blind_eval')
    os.makedirs(save_triple, exist_ok=True)
    os.makedirs(save_pure, exist_ok=True)
    os.makedirs(save_blind_dir, exist_ok=True)

    # 构建模型并加载权重
    model = build_model(device, in_chans=args.in_chans, width=args.width,
                        enc_blk_nums=enc_blk_nums, middle_blk_num=args.middle_blk_num,
                        dec_blk_nums=dec_blk_nums)
    # 加载 checkpoint（兼容 DataParallel 和旧格式）
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
    except Exception:
        # 如果遇到 weights_only 限制，放宽安全校验
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    state = ckpt.get('params', ckpt)   # 尝试取 'params' 键，否则整个字典视为 state_dict
    # 去除 'module.' 前缀
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state[new_k] = v
        state = new_state
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded model from {args.checkpoint}')

    # 构建 GT 映射（相对路径 -> 绝对路径）
    gt_root = os.path.join(args.data_root, 'test_sharp')
    gt_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                full = os.path.join(root, f)
                rel = os.path.normpath(os.path.relpath(full, gt_root))
                gt_map[rel] = full
                if f not in gt_map:     # fallback by basename
                    gt_map[f] = full

    # 收集输入图像（test_blur）
    input_root = os.path.join(args.data_root, 'test_blur')
    input_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith('.png'):
                input_files.append(os.path.join(root, f))
    input_files = sorted(input_files, key=natural_sort_key)

    # 按组分组
    grouped_inputs = defaultdict(list)
    for in_path in input_files:
        rel_in = os.path.normpath(os.path.relpath(in_path, input_root))
        grouped_inputs[get_group_name(rel_in)].append(in_path)

    # 解析 mask 根路径
    resolved_test_mask_csv = resolve_csv_path(args.test_mask_csv, args.data_root)

    # 全局统计
    report = TestReport(crop_border=args.image_border)
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []   # 所有图像的记录

    print(f'===> 开始定量打分，准备比对 {len(input_files)} 张图片...')
    with torch.no_grad():
        for group_name, group_files in grouped_inputs.items():
            print(f'===> Processing group {group_name} ({len(group_files)} images) ...')
            group_rows = []
            group_pure_dir = os.path.join(save_pure, group_name)
            group_triple_dir = os.path.join(save_triple, group_name)
            os.makedirs(group_pure_dir, exist_ok=True)
            os.makedirs(group_triple_dir, exist_ok=True)

            # 获取该组的盲元 CSV
            masks = resolve_group_mask_paths(resolved_test_mask_csv, args.data_root, group_name)
            blind_coords = load_blind_coords(masks['blind_csv']) if masks['blind_csv'] else None
            flash_map = load_flash_map(masks['flash_csv']) if masks['flash_csv'] else {}

            if masks['blind_csv'] and blind_coords is not None:
                print(f"Loaded blind coords for group {group_name} from: {masks['blind_csv']} ({len(blind_coords)} unique points)")
            elif masks['blind_csv']:
                print(f"WARN: blind coords CSV not loaded for group {group_name}: {masks['blind_csv']}")
                print('WARN: blind metrics will stay empty until the CSV path is correct and the file has x,y columns.')
            else:
                print(f'WARN: no blind coords CSV found for group {group_name}')

            if masks['flash_csv'] and len(flash_map) > 0:
                print(f"Loaded flash coords map for group {group_name} from: {masks['flash_csv']} ({len(flash_map)} frames)")
            elif masks['flash_csv']:
                print(f"WARN: flash CSV has no valid frame_name/x/y entries for group {group_name}: {masks['flash_csv']}")

            for idx, in_path in enumerate(group_files):
                name = os.path.basename(in_path)
                rel_in = os.path.normpath(os.path.relpath(in_path, input_root))

                # 读取输入灰度图
                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is None:
                    print('WARN: failed to load', in_path)
                    continue

                # 预处理
                inp_np = to_tensor_gray(in_img, in_chans=args.in_chans)
                inp_tensor = torch.from_numpy(inp_np).float().unsqueeze(0).to(device)

                # 前向
                out = model(inp_tensor)
                out = out.clamp(0, 1)
                out_gray = tensor_to_uint8_gray(out)

                # 查找对应的 GT
                gt_path = gt_map.get(rel_in, gt_map.get(name))
                if gt_path and os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_img is None:
                        print('WARN: failed to load gt for', name)
                        # 仍然保存输出图像
                        cv2.imwrite(os.path.join(group_pure_dir, name), out_gray)
                        continue

                    # 尺寸对齐
                    if out_gray.shape != gt_img.shape:
                        out_gray = cv2.resize(out_gray, (gt_img.shape[1], gt_img.shape[0]))
                    if in_img.shape != gt_img.shape:
                        in_resized = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                    else:
                        in_resized = in_img

                    # 保存三合一对比图
                    triple = np.concatenate([in_resized, out_gray, gt_img], axis=1)
                    cv2.imwrite(os.path.join(group_triple_dir, f'triple_{name}'), triple)
                    cv2.imwrite(os.path.join(group_pure_dir, name), out_gray)

                    # 全图指标
                    report.update_metric(gt_img, out_gray, name)
                    full_psnr = report.total_rgb_psnr[-1]
                    full_ssim = report.total_ssim[-1]

                    # 盲元评估
                    row = {
                        'image': name,
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

                    # 合并所有盲元坐标（静态 + 帧级闪光）
                    merged_coords = []
                    if blind_coords is not None and blind_coords.size > 0:
                        h, w = gt_img.shape[:2]
                        xs = blind_coords[:, 0]
                        ys = blind_coords[:, 1]
                        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.any(valid):
                            merged_coords.extend(zip(xs[valid].tolist(), ys[valid].tolist()))

                    if len(flash_map) > 0:
                        frame_flash = flash_map.get(name, [])
                        merged_coords.extend(frame_flash)

                    if len(merged_coords) > 0:
                        coords_arr = np.unique(np.array(merged_coords, dtype=np.int32), axis=0)
                        h, w = gt_img.shape[:2]
                        xs = coords_arr[:, 0]
                        ys = coords_arr[:, 1]
                        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.any(valid):
                            xs = xs[valid]
                            ys = ys[valid]
                            gt_vals = gt_img[ys, xs].astype(np.float64)
                            out_vals = out_gray[ys, xs].astype(np.float64)
                            err = out_vals - gt_vals
                            abs_err = np.abs(err)
                            sq_err = err ** 2

                            blind_abs_sum += float(abs_err.sum())
                            blind_sq_sum += float(sq_err.sum())
                            blind_pix_sum += len(err)

                            in_vals = in_resized[ys, xs].astype(np.float64)
                            in_err = in_vals - gt_vals
                            in_abs = np.abs(in_err)
                            in_sq = in_err ** 2
                            blind_abs_in_sum += float(in_abs.sum())
                            blind_sq_in_sum += float(in_sq.sum())

                            row.update({
                                'blind_mae': float(abs_err.mean()),
                                'blind_rmse': float(np.sqrt(sq_err.mean())),
                                'blind_psnr': 10.0 * np.log10(255.0*255.0 / max(sq_err.mean(), 1e-12)),
                                'blind_mae_input': float(in_abs.mean()),
                                'blind_count': int(len(err))
                            })
                            if row['blind_mae_input'] is not None:
                                row['blind_mae_gain_abs'] = row['blind_mae_input'] - row['blind_mae']
                                row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (row['blind_mae_input'] + 1e-12)

                    per_image_logs.append(row)
                    group_rows.append(row)
                else:
                    # 没有 GT，只保存输出
                    cv2.imwrite(os.path.join(group_pure_dir, name), out_gray)

                if (idx + 1) % 10 == 0:
                    print(f'Processed {idx+1}/{len(group_files)} in group {group_name}')

            # 保存该组的 CSV
            if len(group_rows) > 0:
                group_csv_dir = os.path.join(save_blind_dir, group_name)
                os.makedirs(group_csv_dir, exist_ok=True)
                group_csv = os.path.join(group_csv_dir, 'test_blind_metrics.csv')
                keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr',
                        'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
                with open(group_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(group_rows)
                print(f'Per-image test metrics saved to: {group_csv}')

    # 输出全图平均指标
    report.print_final_result()

    # 保存全局 CSV
    global_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr',
                'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
        with open(global_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_image_logs)
        print(f'Per-image test metrics saved to: {global_csv}')

    # 盲元总体统计
    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = np.sqrt(blind_mse)
        blind_psnr = 10.0 * np.log10(255.0*255.0 / max(blind_mse, 1e-12))
        print('===> Blind-Pixel Focused Metrics')
        print(f'BlindCount(total sampled): {blind_pix_sum}')
        print(f'Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}')
        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = 10.0 * np.log10(255.0*255.0 / max(blind_mse_in, 1e-12))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(f'Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)')
        if len(per_image_logs) > 0:
            print(f'Blind per-image metrics saved to: {global_csv}')

if __name__ == '__main__':
    main()
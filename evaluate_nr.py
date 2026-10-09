"""
盲元修复质量评估 (输出 vs 输入 vs GT).

指标:
  Residual>X  — 盲元区绝对误差超过 X 灰度级的像素占比, 越小越好
  LocalStd   — 局部标准差均值 (5×5), 越小越好 (盲元引入局部突变)
  EstSNR     — 估计信噪比 (Immerkaer), 越大越好 (盲元去除 → 噪声降低)

不依赖 RGB 预训练模型, 纯数学计算, 天然适配灰度红外图像.

用法:
  python evaluate_nr.py
"""

import os
import re
import csv
import cv2
import numpy as np


# =====================================================================
# 配置
# =====================================================================

OUTPUT_DIR = r"/home/student_server/Qtt/NAFNet/results/LWIR_test"
INPUT_DIR  = r"/home/student_server/Qtt/NAFNet/LWIR_test"
SAVE_DIR   = r"/home/student_server/Qtt/NAFNet/results/LWIR_test/nr_eval"

# Residual 阈值: 盲元绝对误差超过此灰度级即视为残余缺陷
RESIDUAL_THRESHOLDS = [5, 10, 20, 30, 50]


# =====================================================================
# 指标实现
# =====================================================================

def compute_residual(img: np.ndarray, thresh: int, kernel_size: int = 5) -> float:
    """像素与自身邻域中值的偏差 > thresh 的占比 (%), 越小越好.
    盲元 → 局部突变 → 残差率高; 修复后邻域一致性恢复 → 残差率降低."""
    f = img.astype(np.float64)
    median = cv2.medianBlur(img, kernel_size).astype(np.float64)
    dev = np.abs(f - median)
    return float(100.0 * (dev > thresh).sum() / img.size)


def compute_local_std(img: np.ndarray, kernel_size: int = 5) -> float:
    """局部标准差均值, 越小越好. 盲元 → 局部突变 → 标准差增大."""
    f = img.astype(np.float64)
    mean = cv2.blur(f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(f ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    return float(np.mean(local_std))


def compute_est_snr(img: np.ndarray) -> float:
    """估计信噪比 (Immerkaer), 越大越好. 盲元去除 → 噪声方差降低 → SNR 提升."""
    f = img.astype(np.float64)
    lap = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    laplacian = cv2.filter2D(f, -1, lap)
    noise_var = np.var(laplacian) / 72.0
    signal_var = max(np.var(f) - noise_var, 1e-10)
    return float(10.0 * np.log10(signal_var / noise_var))


def compute_roughness(img: np.ndarray) -> float:
    """粗糙度 (IR 非均匀性经典指标), 越小越好.
    归一化 Laplacian 高通能量, 盲元 → 高频伪影 → 粗糙度增大."""
    f = img.astype(np.float64)
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    hf = np.abs(cv2.filter2D(f, -1, lap))
    return float(np.mean(hf) / (np.mean(f) + 1e-10))


def compute_nu(img: np.ndarray, block_size: int = 32) -> float:
    """非均匀性 (局部均值变异系数), 越小越好.
    std(block_means) / mean(block_means) × 100%. 盲元 → 局部均值偏移 → NU 增大."""
    f = img.astype(np.float64)
    h, w = f.shape
    means = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = f[y:min(y + block_size, h), x:min(x + block_size, w)]
            if block.size > block_size:
                means.append(np.mean(block))
    means = np.array(means)
    return float(100.0 * np.std(means) / (np.mean(means) + 1e-10))


# =====================================================================
# 工具
# =====================================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def _resolve_path(base_dir, seq_name, rel_path, img_name):
    p = os.path.join(base_dir, rel_path)
    if os.path.exists(p):
        return p
    p = os.path.join(base_dir, seq_name, img_name)
    if os.path.exists(p):
        return p
    return None


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 扫描输出目录
    out_records = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if not f.endswith('.png'):
                continue
            op = os.path.join(root, f)
            rel = os.path.relpath(op, OUTPUT_DIR).replace('\\', '/')
            out_records.append({
                'out_path': op, 'img_name': f, 'rel_path': rel,
                'seq': rel.split('/')[0] if '/' in rel else 'root',
            })
    out_records.sort(key=lambda r: natural_sort_key(r['rel_path']))
    print(f"找到 {len(out_records)} 张输出图片")

    seq_records = {}
    for r in out_records:
        seq_records.setdefault(r['seq'], []).append(r)

    # 构建列名: residual_{10,20,30,50}, localstd, estsnr 各 × out/in
    METRICS = ([f'residual_{t}' for t in RESIDUAL_THRESHOLDS] +
               ['localstd', 'estsnr'])
    DIRECTION = {}
    for t in RESIDUAL_THRESHOLDS:
        DIRECTION[f'residual_{t}'] = '↓'
    DIRECTION['localstd'] = '↓'
    DIRECTION['estsnr'] = '↑'

    keys = ['image', 'seq']
    for m in METRICS:
        keys += [f'{m}_out', f'{m}_in']

    per_image_rows = []
    seq_stats = {}
    global_vals = {}
    for m in METRICS:
        global_vals[f'{m}_out'] = []
        global_vals[f'{m}_in'] = []

    print("===> 开始评估...")

    for seq_name in sorted(seq_records, key=natural_sort_key):
        seq_recs = sorted(seq_records[seq_name], key=lambda r: natural_sort_key(r['rel_path']))
        sm = {}
        for m in METRICS:
            sm[f'{m}_out'] = []
            sm[f'{m}_in'] = []

        for idx, rec in enumerate(seq_recs):
            out_path = rec['out_path']
            rel_path = rec['rel_path']

            in_path = _resolve_path(INPUT_DIR, seq_name, rel_path, rec['img_name'])

            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            if out_img is None:
                print(f"  警告: 无法读取 {out_path}, 跳过")
                continue

            # ---- 输出图 ----
            out_vals = {}
            for t in RESIDUAL_THRESHOLDS:
                out_vals[f'residual_{t}'] = compute_residual(out_img, t)
            out_vals['localstd'] = compute_local_std(out_img)
            out_vals['estsnr'] = compute_est_snr(out_img)

            # ---- 输入图 ----
            in_vals = {}
            if in_path and os.path.exists(in_path):
                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is not None:
                    for t in RESIDUAL_THRESHOLDS:
                        in_vals[f'residual_{t}'] = compute_residual(in_img, t)
                    in_vals['localstd'] = compute_local_std(in_img)
                    in_vals['estsnr'] = compute_est_snr(in_img)

            row = {'image': rel_path, 'seq': seq_name}
            for m in METRICS:
                row[f'{m}_out'] = round(out_vals[m], 6) if m in out_vals and out_vals[m] is not None else None
                row[f'{m}_in'] = round(in_vals[m], 6) if m in in_vals and in_vals[m] is not None else None
            per_image_rows.append(row)

            def _a(lst, v):
                if v is not None:
                    lst.append(v)
            for m in METRICS:
                _a(sm[f'{m}_out'], out_vals.get(m))
                _a(sm[f'{m}_in'], in_vals.get(m))
                _a(global_vals[f'{m}_out'], out_vals.get(m))
                _a(global_vals[f'{m}_in'], in_vals.get(m))

            if (idx + 1) % 10 == 0 or idx == len(seq_recs) - 1:
                parts = f"Res>10={out_vals.get('residual_10'):.1f}% vs {in_vals.get('residual_10')}%  "
                parts += f"Std={out_vals.get('localstd'):.1f} vs {in_vals.get('localstd')}  "
                parts += f"SNR={out_vals.get('estsnr'):.1f} vs {in_vals.get('estsnr')}"
                print(f"  [{seq_name}] {idx + 1}/{len(seq_recs)}  {parts}")

        if sm['localstd_out']:
            st = {'count': len([x for x in sm['localstd_out'] if x is not None])}
            for m in METRICS:
                vals_out = [x for x in sm[f'{m}_out'] if x is not None]
                vals_in = [x for x in sm[f'{m}_in'] if x is not None]
                st[f'{m}_out'] = float(np.mean(vals_out)) if vals_out else None
                st[f'{m}_in'] = float(np.mean(vals_in)) if vals_in else None
            seq_stats[seq_name] = st

    # ---- CSV ----
    csv_path = os.path.join(SAVE_DIR, 'nr_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

        for sn in sorted(seq_stats, key=natural_sort_key):
            st = seq_stats[sn]
            r = {'image': f'AVERAGE ({sn})', 'seq': sn}
            for m in METRICS:
                r[f'{m}_out'] = round(st[f'{m}_out'], 6) if st[f'{m}_out'] is not None else None
                r[f'{m}_in'] = round(st[f'{m}_in'], 6) if st[f'{m}_in'] is not None else None
            writer.writerow(r)

        def _avg(lst): return float(np.mean([x for x in lst if x is not None])) if lst else None
        r = {'image': 'AVERAGE', 'seq': ''}
        for m in METRICS:
            r[f'{m}_out'] = round(_avg(global_vals[f'{m}_out']), 6) if global_vals[f'{m}_out'] else None
            r[f'{m}_in'] = round(_avg(global_vals[f'{m}_in']), 6) if global_vals[f'{m}_in'] else None
        writer.writerow(r)

    def _avg(lst): return float(np.mean([x for x in lst if x is not None])) if lst else None

    print(f"\nCSV: {csv_path}")
    print(f"{'='*70}")
    print(f"总体平均 ({len([x for x in global_vals['localstd_out'] if x is not None])} 张):")
    print(f"{'':>14s} {'方向':>4s} {'输出(修复后)':>14s}  {'输入(模糊)':>14s}")
    for m in METRICS:
        so = f"{_avg(global_vals[f'{m}_out']):.4f}"
        si = f"{_avg(global_vals[f'{m}_in']):.4f}" if global_vals[f'{m}_in'] else "N/A"
        print(f"  {m:14s} {DIRECTION.get(m, ''):>4s} {so:>14s}  {si:>14s}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

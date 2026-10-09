"""
独立的测试指标计算脚本

用法:
    python evaluate.py --output <输出图目录> --gt <真值目录> --input <输入图目录> --mask <mask目录> --save <保存目录>

输出:
    <save_dir>/test_blind_metrics_001.csv       每个子文件夹逐帧
    <save_dir>/test_blind_metrics.csv           全局逐帧（末行 AVERAGE）
    <save_dir>/test_blind_summary_by_seq.csv    按序列汇总（末行 AVERAGE）

新增:
    blind_operable_rate_5 / 10 / 20：仅在盲元区域统计的有效像元率。
    论文主表建议使用 blind_operable_rate_10，5/20 可用于阈值敏感性分析。
"""

import os
import re
import csv
import argparse
import cv2
import math
import numpy as np


# =====================================================================
# 默认配置 (可通过命令行覆盖)
# =====================================================================

DEFAULTS = {
    'output': r"/home/student_server/Qtt/NAFNet/results/experiment_test/test",
    'gt':     r"/home/student_server/Qtt/NAFNet/data/test_sharp",
    'input':  r"/home/student_server/Qtt/NAFNet/data/test_blur",
    'mask':   r"/home/student_server/Qtt/NAFNet/data/test_mask",
    'save':   r"/home/student_server/Qtt/NAFNet/results/experiment_test/blind_eval",
}

OPERABLE_THRESHOLD = 10.0
BLIND_OPERABLE_THRESHOLDS = [5.0, 10.0, 20.0]

# =====================================================================
# 工具函数
# =====================================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def threshold_tag(v):
    """将阈值转成适合作为 CSV 字段名的字符串。"""
    v = float(v)
    if v.is_integer():
        return str(int(v))
    return str(v).replace('.', 'p')


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den.clip(min=1e-12)).mean()


def load_blind_coords(csv_path):
    if not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if not coords:
        return None
    return np.unique(np.array(coords, dtype=np.int32), axis=0)


def load_flash_map(csv_path):
    if not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                flash_map.setdefault(os.path.basename(row['frame_name']), []).append(
                    (int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    return flash_map


def build_path_maps(base_dir):
    """递归扫描目录，返回 {rel_path: abs_path} 和 {filename: [abs_path, ...]}"""
    rel_map, name_map = {}, {}
    if not os.path.exists(base_dir):
        return rel_map, name_map
    for root, _, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.png'):
                continue
            p = os.path.join(root, f)
            rel_map[os.path.relpath(p, base_dir).replace('\\', '/')] = p
            name_map.setdefault(f, []).append(p)
    return rel_map, name_map


def resolve_by_name(name_map, img_name, seq_hint, base_dir):
    candidates = name_map.get(img_name, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and seq_hint:
        for c in candidates:
            if os.path.relpath(c, base_dir).replace('\\', '/').split('/')[0] == seq_hint:
                return c
    return None


def blind_psnr_from_stats(abs_sum, sq_sum, pix_sum):
    mse = sq_sum / pix_sum if pix_sum > 0 else 1.0
    return 20.0 * math.log10(255.0 / max(math.sqrt(mse), 1e-12))


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f'--{k}', default=v)
    parser.add_argument('--threshold', type=float, default=OPERABLE_THRESHOLD,
                        help='全图有效像元率阈值，默认 10')
    parser.add_argument('--blind-thresholds', nargs='+', type=float,
                        default=BLIND_OPERABLE_THRESHOLDS,
                        help='盲元有效像元率阈值列表，默认 5 10 20')
    args = parser.parse_args()

    operable_threshold = float(args.threshold)
    blind_thresholds = sorted(set(float(t) for t in args.blind_thresholds))

    OUTPUT_DIR = args.output
    GT_DIR     = args.gt
    INPUT_DIR  = args.input
    MASK_ROOT  = args.mask
    SAVE_DIR   = args.save

    # ---- 1. 扫描输出目录 ----
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

    # ---- 2. 建立查找表 ----
    gt_rel_map, gt_name_map = build_path_maps(GT_DIR)
    input_rel_map, input_name_map = build_path_maps(INPUT_DIR)

    # ---- 3. 盲元坐标缓存 ----
    seq_mask_cache = {}

    def get_seq_masks(seq_name):
        key = seq_name or '__root__'
        if key not in seq_mask_cache:
            b = os.path.join(MASK_ROOT, seq_name, 'blind_pixel_coords.csv') if seq_name else os.path.join(MASK_ROOT, '001', 'blind_pixel_coords.csv')
            fl = os.path.join(MASK_ROOT, seq_name, 'flash_pixel_coords.csv') if seq_name else os.path.join(MASK_ROOT, '001', 'flash_pixel_coords.csv')
            seq_mask_cache[key] = {'blind_coords': load_blind_coords(b), 'flash_map': load_flash_map(fl)}
        return seq_mask_cache[key]

    # ---- 4. 遍历打分 ----
    os.makedirs(SAVE_DIR, exist_ok=True)

    blind_or_fields = [f'blind_operable_rate_{threshold_tag(t)}' for t in blind_thresholds]

    keys = ['image', 'seq', 'psnr', 'ssim', 'blind_mae', 'blind_rmse',
            'blind_psnr', 'blind_mae_input', 'blind_mae_gain_abs',
            'blind_mae_gain_pct', 'blind_count', 'operable_rate',
            *blind_or_fields]
    summary_keys = ['seq', 'images', 'psnr', 'ssim', 'blind_count',
                    'blind_mae', 'blind_rmse', 'blind_psnr',
                    'input_blind_mae', 'input_blind_psnr',
                    'blind_mae_gain_abs', 'blind_mae_gain_pct', 'operable_rate',
                    *blind_or_fields]

    per_image_logs = []
    seq_logs_all, seq_stats_all = {}, {}
    global_stats = {'psnr': [], 'ssim': [],
                    'blind_abs': 0.0, 'blind_sq': 0.0, 'blind_pix': 0,
                    'blind_abs_in': 0.0, 'blind_sq_in': 0.0,
                    'operable_count': 0, 'total_pixels': 0,
                    'blind_operable_count': {t: 0 for t in blind_thresholds}}

    print("===> 开始定量打分...")

    for seq_name in sorted(seq_records, key=natural_sort_key):
        seq_recs = sorted(seq_records[seq_name], key=lambda r: natural_sort_key(r['rel_path']))
        seq_logs = []
        sst = {'blind_abs': 0.0, 'blind_sq': 0.0, 'blind_abs_in': 0.0,
               'blind_sq_in': 0.0, 'blind_pix': 0, 'psnr': [], 'ssim': [],
               'operable_count': 0, 'total_pixels': 0,
               'blind_operable_count': {t: 0 for t in blind_thresholds}}

        for rec in seq_recs:
            img_name, rel_name, out_path = rec['img_name'], rec['rel_path'], rec['out_path']
            seq_hint = seq_name if seq_name != 'root' else ''

            gt_path = gt_rel_map.get(rel_name) or gt_rel_map.get(f"{seq_hint}/{img_name}") or resolve_by_name(gt_name_map, img_name, seq_hint, GT_DIR)
            if not (gt_path and os.path.exists(out_path)):
                continue

            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is None or out_img is None:
                continue
            if out_img.shape != gt_img.shape:
                out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

            psnr_v = calculate_psnr(gt_img, out_img)
            ssim_v = calculate_ssim(gt_img, out_img)
            global_stats['psnr'].append(psnr_v)
            global_stats['ssim'].append(ssim_v)
            sst['psnr'].append(psnr_v)
            sst['ssim'].append(ssim_v)

            row = {k: None for k in keys}
            row.update({'image': rel_name, 'seq': seq_name,
                        'psnr': round(psnr_v, 4), 'ssim': round(ssim_v, 6), 'blind_count': 0})

            # 盲元指标
            h, w = gt_img.shape[:2]
            masks = get_seq_masks(seq_hint)
            merged = []
            if masks['blind_coords'] is not None:
                x, y = masks['blind_coords'][:, 0], masks['blind_coords'][:, 1]
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                if np.any(ok):
                    merged.extend(zip(x[ok].tolist(), y[ok].tolist()))
            for fx, fy in masks['flash_map'].get(img_name, []):
                if 0 <= fx < w and 0 <= fy < h:
                    merged.append((fx, fy))

            if merged:
                arr = np.unique(np.array(merged, dtype=np.int32), axis=0)
                if arr.size > 0:
                    x, y = arr[:, 0], arr[:, 1]
                    err = out_img[y, x].astype(np.float64) - gt_img[y, x].astype(np.float64)
                    abs_sum, sq_sum = float(np.abs(err).sum()), float((err ** 2).sum())
                    n = int(len(err))

                    for d, k in [(abs_sum, 'blind_abs'), (sq_sum, 'blind_sq'), (n, 'blind_pix')]:
                        sst[k] += d
                        global_stats[k] += d

                    mae = abs_sum / n
                    row.update({
                        'blind_mae': round(mae, 6),
                        'blind_rmse': round(math.sqrt(sq_sum / n), 6),
                        'blind_psnr': round(blind_psnr_from_stats(abs_sum, sq_sum, n), 4),
                        'blind_count': n,
                    })

                    # 盲元有效像元率：仅在盲元区域统计 |output-GT| < threshold
                    abs_err = np.abs(err)
                    for t in blind_thresholds:
                        blind_operable = int((abs_err < t).sum())
                        sst['blind_operable_count'][t] += blind_operable
                        global_stats['blind_operable_count'][t] += blind_operable
                        row[f'blind_operable_rate_{threshold_tag(t)}'] = round(
                            100.0 * blind_operable / n, 2
                        )

                    # 输入图对比
                    in_path = input_rel_map.get(rel_name) or input_rel_map.get(f"{seq_hint}/{img_name}") or resolve_by_name(input_name_map, img_name, seq_hint, INPUT_DIR)
                    if in_path and os.path.exists(in_path):
                        in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                        if in_img is not None:
                            if in_img.shape != gt_img.shape:
                                in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                            in_err = in_img[y, x].astype(np.float64) - gt_img[y, x].astype(np.float64)
                            in_abs, in_sq = float(np.abs(in_err).sum()), float((in_err ** 2).sum())
                            for d, k in [(in_abs, 'blind_abs_in'), (in_sq, 'blind_sq_in')]:
                                sst[k] += d
                                global_stats[k] += d
                            in_mae = in_abs / n
                            row['blind_mae_input'] = round(in_mae, 6)
                            row['blind_mae_gain_abs'] = round(in_mae - mae, 6)
                            row['blind_mae_gain_pct'] = round(100.0 * (in_mae - mae) / (in_mae + 1e-12), 4)

            # 有效像元率 (全图, 非仅盲区)
            full_err = out_img.astype(np.float64) - gt_img.astype(np.float64)
            operable_all = int((np.abs(full_err) < operable_threshold).sum())
            row['operable_rate'] = round(100.0 * operable_all / (h * w), 2)
            sst['operable_count'] += operable_all
            sst['total_pixels'] += (h * w)
            global_stats['operable_count'] += operable_all
            global_stats['total_pixels'] += (h * w)

            seq_logs.append(row)
            per_image_logs.append(row)

        seq_logs_all[seq_name] = seq_logs
        seq_stats_all[seq_name] = sst

        if seq_logs:
            scsv = os.path.join(SAVE_DIR, f'test_blind_metrics_{seq_name}.csv')
            with open(scsv, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(seq_logs)
            print(f"  [{seq_name}] 已保存: {scsv}")

    # ---- 5. 预计算全局平均值 ----
    psnr_vals = [x for x in global_stats['psnr'] if not math.isinf(x)]
    ssim_vals = [x for x in global_stats['ssim'] if x is not None and not math.isnan(x)]
    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else None
    avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else None

    def make_avg_row(img_key=True):
        """生成全局平均行，img_key=True 用于逐帧 CSV，False 用于汇总 CSV"""
        pix = global_stats['blind_pix']
        total_pix = global_stats.get('total_pixels', 0)
        oper_rate = round(100.0 * global_stats.get('operable_count', 0) / total_pix, 2) if total_pix > 0 else None
        if img_key:
            r = {'image': 'AVERAGE', 'seq': '',
                 'psnr': round(avg_psnr, 4) if avg_psnr else None,
                 'ssim': round(avg_ssim, 6) if avg_ssim else None,
                 'blind_count': pix if pix > 0 else None,
                 'blind_mae': None, 'blind_rmse': None, 'blind_psnr': None,
                 'blind_mae_input': None, 'blind_mae_gain_abs': None, 'blind_mae_gain_pct': None,
                 'operable_rate': oper_rate}
        else:
            r = {'seq': 'AVERAGE', 'images': len(per_image_logs),
                 'psnr': round(avg_psnr, 4) if avg_psnr else None,
                 'ssim': round(avg_ssim, 6) if avg_ssim else None,
                 'blind_count': pix if pix > 0 else None,
                 'blind_mae': None, 'blind_rmse': None, 'blind_psnr': None,
                 'input_blind_mae': None, 'input_blind_psnr': None,
                 'blind_mae_gain_abs': None, 'blind_mae_gain_pct': None,
                 'operable_rate': oper_rate}
        if pix > 0:
            mae = global_stats['blind_abs'] / pix
            mse = global_stats['blind_sq'] / pix
            bpsnr = blind_psnr_from_stats(global_stats['blind_abs'], global_stats['blind_sq'], pix)
            r.update({'blind_mae': round(mae, 6),
                       'blind_rmse': round(math.sqrt(mse), 6),
                       'blind_psnr': round(bpsnr, 4)})
            if img_key:
                r['blind_count'] = pix
            if global_stats['blind_abs_in'] > 0:
                in_mae = global_stats['blind_abs_in'] / pix
                gain = in_mae - mae
                if img_key:
                    r.update({'blind_mae_input': round(in_mae, 6),
                               'blind_mae_gain_abs': round(gain, 6),
                               'blind_mae_gain_pct': round(100.0 * gain / (in_mae + 1e-12), 4)})
                else:
                    in_mse = global_stats['blind_sq_in'] / pix
                    r.update({'input_blind_mae': round(in_mae, 6),
                               'input_blind_psnr': round(blind_psnr_from_stats(global_stats['blind_abs_in'], global_stats['blind_sq_in'], pix), 4),
                               'blind_mae_gain_abs': round(gain, 6),
                               'blind_mae_gain_pct': round(100.0 * gain / (in_mae + 1e-12), 4)})
        if pix > 0:
            for t in blind_thresholds:
                r[f'blind_operable_rate_{threshold_tag(t)}'] = round(
                    100.0 * global_stats['blind_operable_count'][t] / pix, 2
                )
        return r

    # ---- 6. 写全局逐帧 CSV ----
    pcsv = os.path.join(SAVE_DIR, 'test_blind_metrics.csv')
    with open(pcsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_image_logs)
        w.writerow(make_avg_row(img_key=True))
    print(f"全局逐帧已保存: {pcsv}")

    # ---- 7. 写按序列汇总 CSV ----
    scsv = os.path.join(SAVE_DIR, 'test_blind_summary_by_seq.csv')
    with open(scsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_keys)
        w.writeheader()
        for seq_name in sorted(seq_logs_all, key=natural_sort_key):
            logs = seq_logs_all[seq_name]
            st = seq_stats_all[seq_name]
            pix = st['blind_pix']
            row = {
                'seq': seq_name, 'images': len(logs),
                'psnr': round(float(np.mean(st['psnr'])), 4) if st['psnr'] else None,
                'ssim': round(float(np.mean(st['ssim'])), 6) if st['ssim'] else None,
                'blind_count': pix,
                'blind_mae': None, 'blind_rmse': None, 'blind_psnr': None,
                'input_blind_mae': None, 'input_blind_psnr': None,
                'blind_mae_gain_abs': None, 'blind_mae_gain_pct': None,
                'operable_rate': None,
            }
            if pix > 0:
                row.update({
                    'blind_mae': round(st['blind_abs'] / pix, 6),
                    'blind_rmse': round(math.sqrt(st['blind_sq'] / pix), 6),
                    'blind_psnr': round(blind_psnr_from_stats(st['blind_abs'], st['blind_sq'], pix), 4),
                })
                for t in blind_thresholds:
                    row[f'blind_operable_rate_{threshold_tag(t)}'] = round(
                        100.0 * st['blind_operable_count'][t] / pix, 2
                    )
            total_pix = st.get('total_pixels', 0)
            if total_pix > 0:
                row['operable_rate'] = round(100.0 * st['operable_count'] / total_pix, 2)
                if st['blind_abs_in'] > 0:
                    in_m = st['blind_abs_in'] / pix
                    in_mse = st['blind_sq_in'] / pix
                    row.update({
                        'input_blind_mae': round(in_m, 6),
                        'input_blind_psnr': round(blind_psnr_from_stats(st['blind_abs_in'], st['blind_sq_in'], pix), 4),
                        'blind_mae_gain_abs': round(in_m - st['blind_abs'] / pix, 6),
                        'blind_mae_gain_pct': round(100.0 * (in_m - st['blind_abs'] / pix) / (in_m + 1e-12), 4),
                    })
            w.writerow(row)
        w.writerow(make_avg_row(img_key=False))
    print(f"按序列汇总已保存: {scsv}")

    # ---- 8. 控制台打印 ----
    print("=" * 60)
    print(f"全图指标: PSNR = {avg_psnr:.4f} dB  |  SSIM = {avg_ssim:.4f}" if avg_psnr else "全图指标: 无有效数据")
    pix = global_stats['blind_pix']
    if pix > 0:
        mae = global_stats['blind_abs'] / pix
        rmse = math.sqrt(global_stats['blind_sq'] / pix)
        bpsnr = blind_psnr_from_stats(global_stats['blind_abs'], global_stats['blind_sq'], pix)
        print(f"盲元指标: Blind MAE = {mae:.6f}  |  Blind RMSE = {rmse:.6f}  |  Blind PSNR = {bpsnr:.4f}")
        for t in blind_thresholds:
            blind_oper_rate = 100.0 * global_stats['blind_operable_count'][t] / pix
            print(f"盲元有效像元率 Blind Operable Rate@{t:g}: {blind_oper_rate:.2f}%")
    total_pix = global_stats.get('total_pixels', 0)
    if total_pix > 0:
        oper_rate = 100.0 * global_stats['operable_count'] / total_pix
        print(f"有效像元率 (全图, 阈值={operable_threshold:.0f}): {oper_rate:.2f}%")
        if global_stats['blind_abs_in'] > 0:
            in_mae = global_stats['blind_abs_in'] / pix
            in_psnr = blind_psnr_from_stats(global_stats['blind_abs_in'], global_stats['blind_sq_in'], pix)
            gain = in_mae - mae
            print(f"输入图盲元: MAE = {in_mae:.6f}  |  PSNR = {in_psnr:.4f}  |  MAE Gain = {gain:.6f} ({100*gain/(in_mae+1e-12):.2f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()

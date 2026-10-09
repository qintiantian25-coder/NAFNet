import os
import csv
import re
import cv2
import math
import argparse
import numpy as np

BLUR_ROOT = "/home/student_server/Qtt/NAFNet/data/test_blur"
SHARP_ROOT = "/home/student_server/Qtt/NAFNet/data/test_sharp"
MASK_ROOT = "/home/student_server/Qtt/NAFNet/data/test_mask"
OUTPUT_ROOT = "/home/student_server/Qtt/NAFNet/results/traditional"

BLIND_THRESHOLDS = [5.0, 10.0, 20.0]
FULL_THRESHOLD = 10.0


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"([0-9]+)", s)]


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(d * d)
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.T)

    mu1 = cv2.filter2D(a, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, w)[5:-5, 5:-5]

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(a ** 2, -1, w)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b ** 2, -1, w)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, w)[5:-5, 5:-5] - mu12

    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float((num / np.maximum(den, 1e-12)).mean())


def median3_reflect(img):
    pad = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    return cv2.medianBlur(pad, 3)[1:-1, 1:-1]


def load_static_coords(path):
    if not os.path.exists(path):
        return None

    pts = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                pts.append((int(float(r["x"])), int(float(r["y"]))))
            except Exception:
                pass

    if not pts:
        return None
    return np.unique(np.asarray(pts, dtype=np.int32), axis=0)


def load_flash_map(path):
    out = {}
    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                name = os.path.basename(r["frame_name"])
                x, y = int(float(r["x"])), int(float(r["y"]))
                out.setdefault(name, []).append((x, y))
            except Exception:
                pass
    return out


def frame_coords(static_coords, flash_map, frame_name, h, w):
    pts = []

    if static_coords is not None:
        for x, y in static_coords:
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))

    for x, y in flash_map.get(frame_name, []):
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))

    if not pts:
        return None
    return np.unique(np.asarray(pts, dtype=np.int32), axis=0)


# ============================================================
# Gu et al. 2010
# five-frame adaptation
# ============================================================
def gu2010_restore(blur_seq, m=0.10, sigma_thresh=3.0):
    """
    blur_seq: [T,H,W]，本文实验统一 T=5

    顾国华 2010：
    1. 多帧对应像素序列计算 Xmax、Xmin、去极值均值 Xbar
    2. Delta=max(|Xmax-Xbar|/Xbar, |Xmin-Xbar|/Xbar)
    3. Delta>m 检测闪烁盲元
    4. 多帧均值图进行 5x5 局部 3sigma 检测
    5. 两类检测结果合并
    6. 8邻域均值替换盲元
    7. 3x3 中值二次修正
    """
    seq = blur_seq.astype(np.float64)
    T, H, W = seq.shape
    if T < 3:
        raise ValueError("Gu 2010 多帧偏差计算至少需要3帧")

    center = seq[T // 2]

    # ---------- 多帧偏差检测 ----------
    xmax = seq.max(axis=0)
    xmin = seq.min(axis=0)
    xbar = (seq.sum(axis=0) - xmax - xmin) / float(T - 2)

    denom = np.maximum(np.abs(xbar), 1e-6)
    delta = np.maximum(np.abs(xmax - xbar) / denom,
                       np.abs(xmin - xbar) / denom)
    temporal_mask = delta > m

    # ---------- 均值图上的 5×5 局部 3σ ----------
    mean_img = seq.mean(axis=0)

    k5 = np.ones((5, 5), np.float64) / 25.0
    local_mean = cv2.filter2D(mean_img, -1, k5, borderType=cv2.BORDER_REFLECT)
    local_sq = cv2.filter2D(mean_img ** 2, -1, k5, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))

    spatial_mask = (
        (local_std > 1e-6) &
        (np.abs(mean_img - local_mean) > sigma_thresh * local_std)
    )

    blind_mask = temporal_mask | spatial_mask

    # ---------- 8邻域均值预替换 ----------
    k8 = np.ones((3, 3), np.float64)
    k8[1, 1] = 0.0
    k8 /= 8.0

    neighbor_mean = cv2.filter2D(
        center, -1, k8, borderType=cv2.BORDER_REFLECT
    )

    tmp = center.copy()
    tmp[blind_mask] = neighbor_mean[blind_mask]

    # ---------- 3×3 中值二次修正 ----------
    median_img = median3_reflect(
        np.clip(tmp, 0, 255).astype(np.uint8)
    ).astype(np.float64)

    restored = center.copy()
    restored[blind_mask] = median_img[blind_mask]

    return np.clip(restored, 0, 255).astype(np.uint8)


# ============================================================
# Threshold + Median
# ============================================================
def threshold_median_restore(blur_seq, threshold=25):
    center = blur_seq[len(blur_seq) // 2].astype(np.uint8)
    mask = center <= threshold

    med = median3_reflect(center).astype(np.float64)
    out = center.astype(np.float64)
    out[mask] = med[mask]

    # 3x3 全部异常时，用5帧同位置中值兜底
    valid = (~mask).astype(np.uint8)
    valid_count = cv2.filter2D(
        valid, -1, np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT
    )

    bad = mask & (valid_count == 0)
    if np.any(bad):
        tmed = np.median(blur_seq.astype(np.float64), axis=0)
        out[bad] = tmed[bad]

    return np.clip(out, 0, 255).astype(np.uint8)


class Stats:
    def __init__(self):
        self.psnr = []
        self.ssim = []
        self.input_psnr = []
        self.input_ssim = []

        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

        self.input_abs_sum = 0.0
        self.input_sq_sum = 0.0
        self.input_count = 0

        self.blind_or = {t: 0 for t in BLIND_THRESHOLDS}
        self.full_or = 0
        self.full_count = 0

    def update(self, out, inp, gt, coords):
        self.psnr.append(psnr(gt, out))
        self.ssim.append(ssim(gt, out))
        self.input_psnr.append(psnr(gt, inp))
        self.input_ssim.append(ssim(gt, inp))

        err_full = np.abs(out.astype(np.float64) - gt.astype(np.float64))
        self.full_or += int((err_full < FULL_THRESHOLD).sum())
        self.full_count += gt.size

        if coords is None:
            return

        xs, ys = coords[:, 0], coords[:, 1]
        g = gt[ys, xs].astype(np.float64)
        o = out[ys, xs].astype(np.float64)
        i = inp[ys, xs].astype(np.float64)

        e = o - g
        ie = i - g

        ae = np.abs(e)
        iae = np.abs(ie)

        self.abs_sum += float(ae.sum())
        self.sq_sum += float((e ** 2).sum())
        self.count += len(e)

        self.input_abs_sum += float(iae.sum())
        self.input_sq_sum += float((ie ** 2).sum())
        self.input_count += len(ie)

        for t in BLIND_THRESHOLDS:
            self.blind_or[t] += int((ae < t).sum())

    def summary(self):
        mae = self.abs_sum / self.count
        mse = self.sq_sum / self.count
        rmse = math.sqrt(mse)
        bpsnr = 10.0 * math.log10(255.0 ** 2 / mse)

        in_mae = self.input_abs_sum / self.input_count
        in_mse = self.input_sq_sum / self.input_count
        in_bpsnr = 10.0 * math.log10(255.0 ** 2 / in_mse)

        return {
            "psnr": float(np.mean(self.psnr)),
            "ssim": float(np.mean(self.ssim)),
            "blind_mae": mae,
            "blind_rmse": rmse,
            "blind_psnr": bpsnr,
            "or5": 100.0 * self.blind_or[5.0] / self.count,
            "or10": 100.0 * self.blind_or[10.0] / self.count,
            "or20": 100.0 * self.blind_or[20.0] / self.count,
            "full_or10": 100.0 * self.full_or / self.full_count,
            "input_blind_mae": in_mae,
            "input_blind_psnr": in_bpsnr,
            "gain": in_mae - mae,
            "gain_pct": 100.0 * (in_mae - mae) / in_mae,
            "blind_count": self.count,
        }


def evaluate_method(blur_root, sharp_root, mask_root,
                    method_fn, method_name, output_root, **kwargs):

    out_root = os.path.join(output_root, method_name)
    eval_root = os.path.join(output_root, method_name + "_blind_eval")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)

    seqs = sorted(
        [x for x in os.listdir(blur_root)
         if os.path.isdir(os.path.join(blur_root, x))],
        key=natural_key
    )

    global_stats = Stats()
    rows = []
    seq_summary = []
    total_images = 0

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {method_name}")
    print(f"{'=' * 80}")

    for seq_name in seqs:
        blur_dir = os.path.join(blur_root, seq_name)
        sharp_dir = os.path.join(sharp_root, seq_name)
        mask_dir = os.path.join(mask_root, seq_name)

        names = sorted(
            [x for x in os.listdir(blur_dir) if x.lower().endswith(".png")],
            key=natural_key
        )

        static_path = os.path.join(mask_dir, "blind_pixel_coords.csv")
        if not os.path.exists(static_path):
            static_path = os.path.join(mask_dir, "blind_coords.csv")

        static_coords = load_static_coords(static_path)
        flash_map = load_flash_map(
            os.path.join(mask_dir, "flash_pixel_coords.csv")
        )

        seq_stats = Stats()
        seq_rows = []
        seq_out = os.path.join(out_root, seq_name)
        os.makedirs(seq_out, exist_ok=True)

        for ci in range(2, len(names) - 2):
            frames = []
            for off in [-2, -1, 0, 1, 2]:
                p = os.path.join(blur_dir, names[ci + off])
                im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    raise RuntimeError(f"读取失败: {p}")
                frames.append(im)

            blur_seq = np.stack(frames, axis=0)
            name = names[ci]
            inp = blur_seq[2]

            gt_path = os.path.join(sharp_dir, name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise RuntimeError(f"GT读取失败: {gt_path}")

            out = method_fn(blur_seq, **kwargs)
            if out.shape != gt.shape:
                raise RuntimeError(f"{seq_name}/{name}: Output/GT尺寸不一致")

            cv2.imwrite(os.path.join(seq_out, name), out)

            coords = frame_coords(
                static_coords, flash_map, name, gt.shape[0], gt.shape[1]
            )

            global_stats.update(out, inp, gt, coords)
            seq_stats.update(out, inp, gt, coords)

            row = {
                "image": f"{seq_name}/{name}",
                "psnr": psnr(gt, out),
                "ssim": ssim(gt, out)
            }

            if coords is not None:
                xs, ys = coords[:, 0], coords[:, 1]
                e = out[ys, xs].astype(np.float64) - gt[ys, xs].astype(np.float64)
                ae = np.abs(e)
                mse = np.mean(e ** 2)

                row.update({
                    "blind_mae": float(ae.mean()),
                    "blind_rmse": float(np.sqrt(mse)),
                    "blind_psnr": float(10 * np.log10(255 ** 2 / mse)),
                    "blind_or5": float(100 * np.mean(ae < 5)),
                    "blind_or10": float(100 * np.mean(ae < 10)),
                    "blind_or20": float(100 * np.mean(ae < 20)),
                    "blind_count": len(ae)
                })

            rows.append(row)
            seq_rows.append(row)
            total_images += 1

        s = seq_stats.summary()
        s["seq"] = seq_name
        s["images"] = len(seq_rows)
        seq_summary.append(s)

        seq_csv = os.path.join(eval_root, f"test_blind_metrics_{seq_name}.csv")
        if seq_rows:
            with open(seq_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=seq_rows[0].keys())
                writer.writeheader()
                writer.writerows(seq_rows)

    if rows:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(os.path.join(eval_root, "test_blind_metrics.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    if seq_summary:
        keys = list(seq_summary[0].keys())
        with open(os.path.join(eval_root, "test_blind_summary_by_seq.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(seq_summary)

    r = global_stats.summary()

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {method_name}")
    print("=" * 80)
    print(f"全图指标: PSNR = {r['psnr']:.4f} dB  |  SSIM = {r['ssim']:.4f}")
    print(
        f"盲元指标: Blind MAE = {r['blind_mae']:.6f}  |  "
        f"Blind RMSE = {r['blind_rmse']:.6f}  |  "
        f"Blind PSNR = {r['blind_psnr']:.4f}"
    )
    print(f"盲元有效像元率 Blind Operable Rate@5: {r['or5']:.2f}%")
    print(f"盲元有效像元率 Blind Operable Rate@10: {r['or10']:.2f}%")
    print(f"盲元有效像元率 Blind Operable Rate@20: {r['or20']:.2f}%")
    print(f"有效像元率 (全图, 阈值=10): {r['full_or10']:.2f}%")
    print(
        f"输入图盲元: MAE = {r['input_blind_mae']:.6f}  |  "
        f"PSNR = {r['input_blind_psnr']:.4f}  |  "
        f"MAE Gain = {r['gain']:.6f} ({r['gain_pct']:.2f}%)"
    )
    print(f"Images used: {total_images}")
    print(f"Blind pixels used: {r['blind_count']}")
    print("=" * 80)

    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur-root", default=BLUR_ROOT)
    parser.add_argument("--sharp-root", default=SHARP_ROOT)
    parser.add_argument("--mask-root", default=MASK_ROOT)
    parser.add_argument("--output", default=OUTPUT_ROOT)
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        choices=["all", "gu2010", "threshold_median"]
    )
    parser.add_argument(
        "--gu-m", type=float, default=None,
        help="Gu 2010 多帧相对偏差阈值m；应在验证集确定"
    )
    parser.add_argument("--threshold", type=int, default=25)
    args = parser.parse_args()

    methods = args.methods
    if "all" in methods:
        methods = ["gu2010", "threshold_median"]

    if "gu2010" in methods and args.gu_m is None:
        raise ValueError(
            "运行 Gu 2010 时必须显式指定 --gu-m，"
            "请先在验证集确定该参数。"
        )

    results = {}

    if "gu2010" in methods:
        results["Gu2010"] = evaluate_method(
            args.blur_root, args.sharp_root, args.mask_root,
            gu2010_restore, "gu2010", args.output,
            m=args.gu_m, sigma_thresh=3.0
        )

    if "threshold_median" in methods:
        results["Threshold-Median"] = evaluate_method(
            args.blur_root, args.sharp_root, args.mask_root,
            threshold_median_restore, "threshold_median", args.output,
            threshold=args.threshold
        )

    print("\n" + "=" * 120)
    print(
        f"{'Method':20s} {'PSNR':>9s} {'SSIM':>9s} "
        f"{'B-MAE':>10s} {'B-RMSE':>10s} {'B-PSNR':>10s} "
        f"{'OR@5':>8s} {'OR@10':>8s} {'OR@20':>8s}"
    )
    print("-" * 120)

    for name, r in results.items():
        print(
            f"{name:20s} "
            f"{r['psnr']:9.3f} "
            f"{r['ssim']:9.4f} "
            f"{r['blind_mae']:10.3f} "
            f"{r['blind_rmse']:10.3f} "
            f"{r['blind_psnr']:10.3f} "
            f"{r['or5']:8.2f} "
            f"{r['or10']:8.2f} "
            f"{r['or20']:8.2f}"
        )
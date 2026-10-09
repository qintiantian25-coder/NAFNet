#!/usr/bin/env python3
"""
NAFNet 测试脚本
统一采用 FGAF-Net evaluate.py 的评价方式。

评价指标：
1. PSNR / SSIM
   - 逐帧计算后取平均
2. Blind MAE / Blind RMSE / Blind PSNR
   - 合并静态盲元和当前帧闪烁盲元
   - 对全部测试帧中的盲元像素统一累计后计算
3. Operable Rate
   - 全图 |output - GT| < 10 的像素比例
4. Blind Operable Rate
   - 仅在盲元区域统计
   - 默认计算 @5、@10、@20
5. 同时统计 Input 对应指标

Usage:
    python tools/test_nafnet_blind.py \
        --data_root /home/student_server/Qtt/NAFNet/data_new \
        --checkpoint experiments/models/best_model.pt \
        --in_chans 3 \
        --save_dir results/experiment_test \
        --test_mask_csv /home/student_server/Qtt/NAFNet/data_new/test_mask \
        --image_border 0
"""

import os
import sys
import re
import csv
import math
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch


# ============================================================
# NAFNet
# ============================================================
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from basicsr.models.archs.NAFNet_arch import NAFNetLocal


# ============================================================
# 默认评价阈值
# ============================================================

# 全图有效像元率
OPERABLE_THRESHOLD = 10.0

# 盲元有效像元率
# 论文主表建议使用 @10
BLIND_OPERABLE_THRESHOLDS = [
    5.0,
    10.0,
    20.0
]


# ============================================================
# 工具函数
# ============================================================

def natural_sort_key(s):
    return [
        int(t) if t.isdigit()
        else t.lower()
        for t in re.split(
            r'([0-9]+)',
            s
        )
    ]


def threshold_tag(v):
    """
    阈值转换成 CSV 字段后缀。

    5.0  -> 5
    10.0 -> 10
    2.5  -> 2p5
    """
    v = float(v)

    if v.is_integer():
        return str(int(v))

    return str(v).replace(
        '.',
        'p'
    )


# ============================================================
# 与 evaluate.py 完全一致的 PSNR / SSIM
# ============================================================

def calculate_psnr(img1, img2):
    img1 = img1.astype(
        np.float64
    )

    img2 = img2.astype(
        np.float64
    )

    mse = np.mean(
        (img1 - img2) ** 2
    )

    if mse == 0:
        return float('inf')

    return 20.0 * math.log10(
        255.0 /
        math.sqrt(mse)
    )


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(
        np.float64
    )

    img2 = img2.astype(
        np.float64
    )

    kernel = cv2.getGaussianKernel(
        11,
        1.5
    )

    window = np.outer(
        kernel,
        kernel.transpose()
    )

    mu1 = cv2.filter2D(
        img1,
        -1,
        window
    )[5:-5, 5:-5]

    mu2 = cv2.filter2D(
        img2,
        -1,
        window
    )[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        cv2.filter2D(
            img1 ** 2,
            -1,
            window
        )[5:-5, 5:-5]
        - mu1_sq
    )

    sigma2_sq = (
        cv2.filter2D(
            img2 ** 2,
            -1,
            window
        )[5:-5, 5:-5]
        - mu2_sq
    )

    sigma12 = (
        cv2.filter2D(
            img1 * img2,
            -1,
            window
        )[5:-5, 5:-5]
        - mu1_mu2
    )

    num = (
        (2 * mu1_mu2 + C1)
        *
        (2 * sigma12 + C2)
    )

    den = (
        (mu1_sq + mu2_sq + C1)
        *
        (
            sigma1_sq
            + sigma2_sq
            + C2
        )
    )

    return float(
        (
            num /
            den.clip(
                min=1e-12
            )
        ).mean()
    )


def blind_psnr_from_stats(
    sq_sum,
    pix_sum
):
    if pix_sum <= 0:
        return None

    mse = (
        sq_sum /
        pix_sum
    )

    if mse <= 0:
        return float('inf')

    return (
        10.0
        *
        math.log10(
            (255.0 * 255.0)
            /
            mse
        )
    )


def safe_mean(values):
    valid = []

    for v in values:

        if v is None:
            continue

        if (
            math.isnan(v)
            or math.isinf(v)
        ):
            continue

        valid.append(v)

    if not valid:
        return None

    return float(
        np.mean(valid)
    )


# ============================================================
# Mask读取
# ============================================================

def load_blind_coords(csv_path):
    if (
        not csv_path
        or
        not os.path.exists(csv_path)
    ):
        return None

    coords = []

    with open(
        csv_path,
        'r',
        encoding='utf-8-sig',
        newline=''
    ) as f:

        reader = csv.DictReader(f)

        if (
            reader.fieldnames is None
            or
            'x' not in reader.fieldnames
            or
            'y' not in reader.fieldnames
        ):
            return None

        for row in reader:

            try:
                x = int(
                    float(row['x'])
                )

                y = int(
                    float(row['y'])
                )

                coords.append(
                    (x, y)
                )

            except Exception:
                continue

    if not coords:
        return None

    return np.unique(
        np.array(
            coords,
            dtype=np.int32
        ),
        axis=0
    )


def load_flash_map(csv_path):
    if (
        not csv_path
        or
        not os.path.exists(csv_path)
    ):
        return {}

    flash_map = {}

    with open(
        csv_path,
        'r',
        encoding='utf-8-sig',
        newline=''
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return {}

        if (
            'frame_name'
            not in reader.fieldnames
            or
            'x'
            not in reader.fieldnames
            or
            'y'
            not in reader.fieldnames
        ):
            return {}

        for row in reader:

            try:
                frame_name = os.path.basename(
                    str(
                        row[
                            'frame_name'
                        ]
                    )
                )

                x = int(
                    float(
                        row['x']
                    )
                )

                y = int(
                    float(
                        row['y']
                    )
                )

            except Exception:
                continue

            flash_map.setdefault(
                frame_name,
                set()
            ).add(
                (x, y)
            )

    return {
        k: list(v)
        for k, v
        in flash_map.items()
    }


# ============================================================
# Mask路径处理
# ============================================================

def resolve_csv_path(
    csv_path,
    data_root
):
    if not csv_path:
        return None

    if os.path.isdir(
        csv_path
    ):
        return csv_path

    if (
        os.path.isabs(csv_path)
        and
        os.path.exists(csv_path)
    ):
        return csv_path

    if os.path.exists(
        csv_path
    ):
        return csv_path

    if data_root:

        candidate = os.path.join(
            data_root,
            csv_path
        )

        if os.path.exists(
            candidate
        ):
            return candidate

    return csv_path


def resolve_group_mask_paths(
    mask_base_path,
    data_root,
    group_name
):
    blind_csv_candidates = []
    flash_csv = None

    if mask_base_path:

        if os.path.isdir(
            mask_base_path
        ):

            group_dir = os.path.join(
                mask_base_path,
                group_name
            )

            blind_csv_candidates.extend([
                os.path.join(
                    group_dir,
                    'blind_coords.csv'
                ),

                os.path.join(
                    group_dir,
                    'blind_pixel_coords.csv'
                )
            ])

            flash_csv = os.path.join(
                group_dir,
                'flash_pixel_coords.csv'
            )

        else:

            blind_csv_candidates.append(
                mask_base_path
            )

            if (
                data_root
                and
                not os.path.isabs(
                    mask_base_path
                )
            ):

                blind_csv_candidates.append(
                    os.path.join(
                        data_root,
                        mask_base_path
                    )
                )

            base_dir = os.path.dirname(
                mask_base_path
            )

            if base_dir:

                blind_csv_candidates.append(
                    os.path.join(
                        base_dir,
                        group_name,
                        os.path.basename(
                            mask_base_path
                        )
                    )
                )

                flash_csv = os.path.join(
                    base_dir,
                    group_name,
                    'flash_pixel_coords.csv'
                )

    if data_root:

        default_group_dir = os.path.join(
            data_root,
            'test_mask',
            group_name
        )

        blind_csv_candidates.extend([
            os.path.join(
                default_group_dir,
                'blind_coords.csv'
            ),

            os.path.join(
                default_group_dir,
                'blind_pixel_coords.csv'
            )
        ])

        if flash_csv is None:

            flash_csv = os.path.join(
                default_group_dir,
                'flash_pixel_coords.csv'
            )

    blind_csv = None
    seen = set()

    for candidate in blind_csv_candidates:

        if not candidate:
            continue

        if candidate in seen:
            continue

        seen.add(
            candidate
        )

        if os.path.exists(
            candidate
        ):

            blind_csv = candidate
            break

    if (
        flash_csv
        and
        not os.path.exists(
            flash_csv
        )
    ):
        flash_csv = None

    return {
        'blind_csv':
            blind_csv,

        'flash_csv':
            flash_csv
    }


def get_group_name(rel_path):
    parts = os.path.normpath(
        rel_path
    ).split(
        os.sep
    )

    if len(parts) > 1:
        return parts[0]

    return 'root'


# ============================================================
# 模型
# ============================================================

def build_model(
    device,
    in_chans=3,
    width=64,
    enc_blk_nums=None,
    middle_blk_num=1,
    dec_blk_nums=None
):

    if enc_blk_nums is None:
        enc_blk_nums = [
            1,
            1,
            1,
            28
        ]

    if dec_blk_nums is None:
        dec_blk_nums = [
            1,
            1,
            1,
            1
        ]

    model = NAFNetLocal(
        img_channel=in_chans,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=
            middle_blk_num,
        dec_blk_nums=
            dec_blk_nums,
        train_size=(
            1,
            in_chans,
            256,
            256
        )
    )

    return model.to(
        device
    )


# ============================================================
# 图像预处理
# ============================================================

def preprocess_image(
    img,
    in_chans=3
):
    if img is None:
        raise ValueError(
            "Empty image"
        )

    if img.dtype != np.float32:
        img = img.astype(
            np.float32
        )

    if img.max() > 1.0:
        img = (
            img /
            255.0
        )

    if img.ndim == 2:

        if in_chans == 3:

            img = np.stack(
                [
                    img,
                    img,
                    img
                ],
                axis=2
            )

        else:

            img = img[
                ...,
                np.newaxis
            ]

    elif (
        img.ndim == 3
        and
        img.shape[2] == 3
    ):

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    else:

        raise ValueError(
            f"Unexpected image shape: "
            f"{img.shape}"
        )

    img = img.transpose(
        2,
        0,
        1
    )

    return torch.from_numpy(
        img
    ).float()


def postprocess_output(
    tensor
):
    """
    网络输出转换为 8-bit 灰度图。
    三通道输出取通道平均。
    """

    out_np = (
        tensor
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    if out_np.ndim == 3:

        if out_np.shape[0] == 1:

            out_np = out_np[0]

        elif out_np.shape[0] == 3:

            out_np = out_np.mean(
                axis=0
            )

        else:

            raise ValueError(
                f"Unexpected output channels: "
                f"{out_np.shape[0]}"
            )

    out_np = np.clip(
        out_np,
        0,
        1
    )

    return (
        out_np * 255
    ).round().astype(
        np.uint8
    )


# ============================================================
# 统计量
# ============================================================

def make_empty_stats(
    blind_thresholds
):

    return {
        # Output 全图
        'psnr': [],
        'ssim': [],

        # Input 全图
        'input_psnr': [],
        'input_ssim': [],

        # Output Blind
        'blind_abs': 0.0,
        'blind_sq': 0.0,
        'blind_pix': 0,

        # Input Blind
        'blind_abs_in': 0.0,
        'blind_sq_in': 0.0,
        'blind_pix_in': 0,

        # Output 全图有效像元
        'operable_count': 0,
        'total_pixels': 0,

        # Input 全图有效像元
        'input_operable_count': 0,
        'input_total_pixels': 0,

        # Output Blind Operable
        'blind_operable_count': {
            float(t): 0
            for t
            in blind_thresholds
        },

        # Input Blind Operable
        'input_blind_operable_count': {
            float(t): 0
            for t
            in blind_thresholds
        }
    }


# ============================================================
# 主函数
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root',
        type=str,
        required=True
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True
    )

    parser.add_argument(
        '--in_chans',
        type=int,
        default=3
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=
        'results/nafnet_test'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    parser.add_argument(
        '--test_mask_csv',
        type=str,
        default=None
    )

    parser.add_argument(
        '--image_border',
        type=int,
        default=0
    )

    parser.add_argument(
        '--width',
        type=int,
        default=64
    )

    parser.add_argument(
        '--enc_blk_nums',
        type=str,
        default='1,1,1,28'
    )

    parser.add_argument(
        '--middle_blk_num',
        type=int,
        default=1
    )

    parser.add_argument(
        '--dec_blk_nums',
        type=str,
        default='1,1,1,1'
    )

    # 全图有效像元阈值
    parser.add_argument(
        '--threshold',
        type=float,
        default=
        OPERABLE_THRESHOLD
    )

    # 盲元有效像元率阈值
    parser.add_argument(
        '--blind_thresholds',
        nargs='+',
        type=float,
        default=
        BLIND_OPERABLE_THRESHOLDS
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 参数
    # --------------------------------------------------------

    enc_blk_nums = [
        int(x)
        for x
        in args.enc_blk_nums.split(',')
    ]

    dec_blk_nums = [
        int(x)
        for x
        in args.dec_blk_nums.split(',')
    ]

    operable_threshold = float(
        args.threshold
    )

    blind_thresholds = sorted(
        set(
            float(t)
            for t
            in args.blind_thresholds
        )
    )

    device = torch.device(
        args.device
        if (
            torch.cuda.is_available()
            and
            'cuda' in args.device
        )
        else
        'cpu'
    )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    save_triple = os.path.join(
        args.save_dir,
        'triple_comparison'
    )

    save_pure = os.path.join(
        args.save_dir,
        'test'
    )

    save_eval = os.path.join(
        args.save_dir,
        'blind_eval'
    )

    os.makedirs(
        save_triple,
        exist_ok=True
    )

    os.makedirs(
        save_pure,
        exist_ok=True
    )

    os.makedirs(
        save_eval,
        exist_ok=True
    )

    print("=" * 80)

    print(
        "NAFNet TEST "
        "- UNIFIED EVALUATION"
    )

    print("=" * 80)

    print(
        f"Data root : "
        f"{args.data_root}"
    )

    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )

    print(
        f"Device    : "
        f"{device}"
    )

    print(
        f"Full-image threshold: "
        f"|error| < "
        f"{operable_threshold:g}"
    )

    print(
        "Blind thresholds: "
        + ", ".join(
            f"{t:g}"
            for t
            in blind_thresholds
        )
    )

    print("=" * 80)

    # ========================================================
    # 构建模型
    # ========================================================

    model = build_model(
        device=device,
        in_chans=
            args.in_chans,
        width=
            args.width,
        enc_blk_nums=
            enc_blk_nums,
        middle_blk_num=
            args.middle_blk_num,
        dec_blk_nums=
            dec_blk_nums
    )

    # ========================================================
    # 加载权重
    # ========================================================

    try:

        ckpt = torch.load(
            args.checkpoint,
            map_location='cpu'
        )

    except Exception:

        try:
            torch.serialization.add_safe_globals([
                np._core.multiarray.scalar
            ])
        except Exception:
            pass

        ckpt = torch.load(
            args.checkpoint,
            map_location='cpu',
            weights_only=False
        )

    state = ckpt.get(
        'params',
        ckpt.get(
            'model',
            ckpt
        )
    )

    if isinstance(
        state,
        dict
    ):

        new_state = {}

        for k, v in state.items():

            new_k = (
                k[7:]
                if k.startswith('module.')
                else k
            )

            new_state[new_k] = v

        state = new_state

    model.load_state_dict(
        state
    )

    model.eval()

    print(
        f"Loaded model from: "
        f"{args.checkpoint}"
    )

    # ========================================================
    # GT索引
    # ========================================================

    gt_root = os.path.join(
        args.data_root,
        'test_sharp'
    )

    input_root = os.path.join(
        args.data_root,
        'test_blur'
    )

    gt_map = {}

    for root, _, files in os.walk(
        gt_root
    ):

        for filename in files:

            if not filename.lower().endswith(
                '.png'
            ):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            rel = os.path.normpath(
                os.path.relpath(
                    full_path,
                    gt_root
                )
            )

            gt_map[rel] = full_path

            if filename not in gt_map:
                gt_map[
                    filename
                ] = full_path

    # ========================================================
    # 输入图像
    # ========================================================

    input_files = []

    for root, _, files in os.walk(
        input_root
    ):

        for filename in files:

            if filename.lower().endswith(
                '.png'
            ):

                input_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    input_files = sorted(
        input_files,
        key=natural_sort_key
    )

    grouped_inputs = defaultdict(
        list
    )

    for in_path in input_files:

        rel_in = os.path.normpath(
            os.path.relpath(
                in_path,
                input_root
            )
        )

        group_name = get_group_name(
            rel_in
        )

        grouped_inputs[
            group_name
        ].append(
            in_path
        )

    # ========================================================
    # Mask根目录
    # ========================================================

    resolved_mask_root = (
        resolve_csv_path(
            args.test_mask_csv,
            args.data_root
        )
    )

    # ========================================================
    # CSV字段
    # ========================================================

    blind_or_fields = [
        f'blind_operable_rate_'
        f'{threshold_tag(t)}'
        for t
        in blind_thresholds
    ]

    input_blind_or_fields = [
        f'input_blind_operable_rate_'
        f'{threshold_tag(t)}'
        for t
        in blind_thresholds
    ]

    keys = [
        'image',
        'seq',

        'psnr',
        'ssim',

        'blind_mae',
        'blind_rmse',
        'blind_psnr',

        'operable_rate',

        *blind_or_fields,

        'input_psnr',
        'input_ssim',

        'input_blind_mae',
        'input_blind_rmse',
        'input_blind_psnr',

        'input_operable_rate',

        *input_blind_or_fields,

        'blind_mae_gain_abs',
        'blind_mae_gain_pct',

        'blind_count'
    ]

    summary_keys = [
        'seq',
        'images',

        'psnr',
        'ssim',

        'blind_mae',
        'blind_rmse',
        'blind_psnr',

        'operable_rate',

        *blind_or_fields,

        'input_psnr',
        'input_ssim',

        'input_blind_mae',
        'input_blind_rmse',
        'input_blind_psnr',

        'input_operable_rate',

        *input_blind_or_fields,

        'blind_mae_gain_abs',
        'blind_mae_gain_pct',

        'blind_count'
    ]

    # ========================================================
    # 全局统计
    # ========================================================

    global_stats = make_empty_stats(
        blind_thresholds
    )

    per_image_logs = []

    seq_logs_all = {}

    seq_stats_all = {}

    print(
        f"\n===> 开始测试，共 "
        f"{len(input_files)} 张图像，"
        f"{len(grouped_inputs)} 个序列"
    )

    # ========================================================
    # 推理
    # ========================================================

    with torch.no_grad():

        for group_name in sorted(
            grouped_inputs,
            key=natural_sort_key
        ):

            group_files = sorted(
                grouped_inputs[
                    group_name
                ],
                key=natural_sort_key
            )

            print(
                f"\n===> 处理序列 "
                f"{group_name} "
                f"({len(group_files)} 张)"
            )

            group_pure_dir = os.path.join(
                save_pure,
                group_name
            )

            group_triple_dir = os.path.join(
                save_triple,
                group_name
            )

            os.makedirs(
                group_pure_dir,
                exist_ok=True
            )

            os.makedirs(
                group_triple_dir,
                exist_ok=True
            )

            # ------------------------------------------------
            # Mask
            # ------------------------------------------------

            mask_paths = (
                resolve_group_mask_paths(
                    resolved_mask_root,
                    args.data_root,
                    group_name
                )
            )

            blind_coords = (
                load_blind_coords(
                    mask_paths[
                        'blind_csv'
                    ]
                )
                if
                mask_paths[
                    'blind_csv'
                ]
                else None
            )

            flash_map = (
                load_flash_map(
                    mask_paths[
                        'flash_csv'
                    ]
                )
                if
                mask_paths[
                    'flash_csv'
                ]
                else {}
            )

            seq_stats = make_empty_stats(
                blind_thresholds
            )

            seq_logs = []

            # =================================================
            # 当前序列逐帧
            # =================================================

            for idx, in_path in enumerate(
                group_files
            ):

                name = os.path.basename(
                    in_path
                )

                rel_in = os.path.normpath(
                    os.path.relpath(
                        in_path,
                        input_root
                    )
                )

                # =============================================
                # 输入
                # =============================================

                if args.in_chans == 3:

                    img_bgr = cv2.imread(
                        in_path,
                        cv2.IMREAD_COLOR
                    )

                    if img_bgr is None:

                        print(
                            f"WARN: cannot read "
                            f"{in_path}"
                        )

                        continue

                    inp_tensor = (
                        preprocess_image(
                            img_bgr,
                            in_chans=
                                args.in_chans
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                else:

                    img_gray = cv2.imread(
                        in_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img_gray is None:
                        continue

                    inp_tensor = (
                        preprocess_image(
                            img_gray,
                            in_chans=
                                args.in_chans
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                # =============================================
                # 前向
                # =============================================

                out = model(
                    inp_tensor
                )

                out = out.clamp(
                    0,
                    1
                )

                out_gray = postprocess_output(
                    out
                )

                # =============================================
                # GT
                # =============================================

                gt_path = gt_map.get(
                    rel_in,
                    gt_map.get(
                        name
                    )
                )

                if (
                    not gt_path
                    or
                    not os.path.exists(
                        gt_path
                    )
                ):

                    cv2.imwrite(
                        os.path.join(
                            group_pure_dir,
                            name
                        ),
                        out_gray
                    )

                    continue

                gt_img = cv2.imread(
                    gt_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if gt_img is None:

                    cv2.imwrite(
                        os.path.join(
                            group_pure_dir,
                            name
                        ),
                        out_gray
                    )

                    continue

                # =============================================
                # 尺寸
                # =============================================

                if (
                    out_gray.shape
                    != gt_img.shape
                ):

                    out_gray = cv2.resize(
                        out_gray,
                        (
                            gt_img.shape[1],
                            gt_img.shape[0]
                        )
                    )

                if args.in_chans == 3:

                    in_resized_bgr = cv2.resize(
                        img_bgr,
                        (
                            gt_img.shape[1],
                            gt_img.shape[0]
                        )
                    )

                    in_gray = cv2.cvtColor(
                        in_resized_bgr,
                        cv2.COLOR_BGR2GRAY
                    )

                else:

                    in_gray = cv2.resize(
                        img_gray,
                        (
                            gt_img.shape[1],
                            gt_img.shape[0]
                        )
                    )

                # =============================================
                # 保存输出
                # =============================================

                cv2.imwrite(
                    os.path.join(
                        group_pure_dir,
                        name
                    ),
                    out_gray
                )

                triple = np.concatenate(
                    [
                        in_gray,
                        out_gray,
                        gt_img
                    ],
                    axis=1
                )

                cv2.imwrite(
                    os.path.join(
                        group_triple_dir,
                        f'triple_{name}'
                    ),
                    triple
                )

                # =============================================
                # Row
                # =============================================

                row = {
                    k: None
                    for k
                    in keys
                }

                row[
                    'image'
                ] = rel_in

                row[
                    'seq'
                ] = group_name

                row[
                    'blind_count'
                ] = 0

                # =============================================
                # Output全图 PSNR / SSIM
                # =============================================

                psnr_v = calculate_psnr(
                    gt_img,
                    out_gray
                )

                ssim_v = calculate_ssim(
                    gt_img,
                    out_gray
                )

                row[
                    'psnr'
                ] = round(
                    psnr_v,
                    4
                )

                row[
                    'ssim'
                ] = round(
                    ssim_v,
                    6
                )

                global_stats[
                    'psnr'
                ].append(
                    psnr_v
                )

                global_stats[
                    'ssim'
                ].append(
                    ssim_v
                )

                seq_stats[
                    'psnr'
                ].append(
                    psnr_v
                )

                seq_stats[
                    'ssim'
                ].append(
                    ssim_v
                )

                # =============================================
                # Input全图 PSNR / SSIM
                # =============================================

                input_psnr_v = (
                    calculate_psnr(
                        gt_img,
                        in_gray
                    )
                )

                input_ssim_v = (
                    calculate_ssim(
                        gt_img,
                        in_gray
                    )
                )

                row[
                    'input_psnr'
                ] = round(
                    input_psnr_v,
                    4
                )

                row[
                    'input_ssim'
                ] = round(
                    input_ssim_v,
                    6
                )

                global_stats[
                    'input_psnr'
                ].append(
                    input_psnr_v
                )

                global_stats[
                    'input_ssim'
                ].append(
                    input_ssim_v
                )

                seq_stats[
                    'input_psnr'
                ].append(
                    input_psnr_v
                )

                seq_stats[
                    'input_ssim'
                ].append(
                    input_ssim_v
                )

                # =============================================
                # 盲元坐标
                # =============================================

                h, w = gt_img.shape[:2]

                merged_coords = []

                # 静态盲元
                if (
                    blind_coords
                    is not None
                    and
                    blind_coords.size > 0
                ):

                    xs = blind_coords[
                        :,
                        0
                    ]

                    ys = blind_coords[
                        :,
                        1
                    ]

                    valid = (
                        (xs >= 0)
                        &
                        (xs < w)
                        &
                        (ys >= 0)
                        &
                        (ys < h)
                    )

                    if np.any(
                        valid
                    ):

                        merged_coords.extend(
                            zip(
                                xs[
                                    valid
                                ].tolist(),

                                ys[
                                    valid
                                ].tolist()
                            )
                        )

                # 闪烁盲元
                frame_flash = flash_map.get(
                    name,
                    []
                )

                for fx, fy in frame_flash:

                    if (
                        0 <= fx < w
                        and
                        0 <= fy < h
                    ):

                        merged_coords.append(
                            (
                                fx,
                                fy
                            )
                        )

                # =============================================
                # Blind指标
                # =============================================

                if merged_coords:

                    coords_arr = np.unique(
                        np.array(
                            merged_coords,
                            dtype=np.int32
                        ),
                        axis=0
                    )

                    xs = coords_arr[
                        :,
                        0
                    ]

                    ys = coords_arr[
                        :,
                        1
                    ]

                    valid = (
                        (xs >= 0)
                        &
                        (xs < w)
                        &
                        (ys >= 0)
                        &
                        (ys < h)
                    )

                    xs = xs[
                        valid
                    ]

                    ys = ys[
                        valid
                    ]

                    if len(xs) > 0:

                        gt_vals = gt_img[
                            ys,
                            xs
                        ].astype(
                            np.float64
                        )

                        out_vals = out_gray[
                            ys,
                            xs
                        ].astype(
                            np.float64
                        )

                        # ------------------------------------
                        # Output blind
                        # ------------------------------------

                        err = (
                            out_vals
                            -
                            gt_vals
                        )

                        abs_err = np.abs(
                            err
                        )

                        sq_err = (
                            err ** 2
                        )

                        n = int(
                            len(err)
                        )

                        abs_sum = float(
                            abs_err.sum()
                        )

                        sq_sum = float(
                            sq_err.sum()
                        )

                        mae = (
                            abs_sum /
                            n
                        )

                        rmse = math.sqrt(
                            sq_sum /
                            n
                        )

                        bpsnr = (
                            blind_psnr_from_stats(
                                sq_sum,
                                n
                            )
                        )

                        row[
                            'blind_mae'
                        ] = round(
                            mae,
                            6
                        )

                        row[
                            'blind_rmse'
                        ] = round(
                            rmse,
                            6
                        )

                        row[
                            'blind_psnr'
                        ] = round(
                            bpsnr,
                            4
                        )

                        row[
                            'blind_count'
                        ] = n

                        seq_stats[
                            'blind_abs'
                        ] += abs_sum

                        seq_stats[
                            'blind_sq'
                        ] += sq_sum

                        seq_stats[
                            'blind_pix'
                        ] += n

                        global_stats[
                            'blind_abs'
                        ] += abs_sum

                        global_stats[
                            'blind_sq'
                        ] += sq_sum

                        global_stats[
                            'blind_pix'
                        ] += n

                        # ------------------------------------
                        # Output Blind Operable Rate
                        # ------------------------------------

                        for t in blind_thresholds:

                            count = int(
                                (
                                    abs_err
                                    <
                                    t
                                ).sum()
                            )

                            field = (
                                f'blind_operable_rate_'
                                f'{threshold_tag(t)}'
                            )

                            row[
                                field
                            ] = round(
                                100.0
                                *
                                count
                                /
                                n,
                                2
                            )

                            seq_stats[
                                'blind_operable_count'
                            ][t] += count

                            global_stats[
                                'blind_operable_count'
                            ][t] += count

                        # ------------------------------------
                        # Input blind
                        # ------------------------------------

                        in_vals = in_gray[
                            ys,
                            xs
                        ].astype(
                            np.float64
                        )

                        in_err = (
                            in_vals
                            -
                            gt_vals
                        )

                        in_abs_err = np.abs(
                            in_err
                        )

                        in_sq_err = (
                            in_err ** 2
                        )

                        in_abs_sum = float(
                            in_abs_err.sum()
                        )

                        in_sq_sum = float(
                            in_sq_err.sum()
                        )

                        in_mae = (
                            in_abs_sum /
                            n
                        )

                        in_rmse = math.sqrt(
                            in_sq_sum /
                            n
                        )

                        in_bpsnr = (
                            blind_psnr_from_stats(
                                in_sq_sum,
                                n
                            )
                        )

                        row[
                            'input_blind_mae'
                        ] = round(
                            in_mae,
                            6
                        )

                        row[
                            'input_blind_rmse'
                        ] = round(
                            in_rmse,
                            6
                        )

                        row[
                            'input_blind_psnr'
                        ] = round(
                            in_bpsnr,
                            4
                        )

                        seq_stats[
                            'blind_abs_in'
                        ] += in_abs_sum

                        seq_stats[
                            'blind_sq_in'
                        ] += in_sq_sum

                        seq_stats[
                            'blind_pix_in'
                        ] += n

                        global_stats[
                            'blind_abs_in'
                        ] += in_abs_sum

                        global_stats[
                            'blind_sq_in'
                        ] += in_sq_sum

                        global_stats[
                            'blind_pix_in'
                        ] += n

                        # ------------------------------------
                        # Input Blind Operable Rate
                        # ------------------------------------

                        for t in blind_thresholds:

                            count = int(
                                (
                                    in_abs_err
                                    <
                                    t
                                ).sum()
                            )

                            field = (
                                f'input_blind_operable_rate_'
                                f'{threshold_tag(t)}'
                            )

                            row[
                                field
                            ] = round(
                                100.0
                                *
                                count
                                /
                                n,
                                2
                            )

                            seq_stats[
                                'input_blind_operable_count'
                            ][t] += count

                            global_stats[
                                'input_blind_operable_count'
                            ][t] += count

                        # ------------------------------------
                        # Gain
                        # ------------------------------------

                        gain_abs = (
                            in_mae
                            -
                            mae
                        )

                        gain_pct = (
                            100.0
                            *
                            gain_abs
                            /
                            (
                                in_mae
                                +
                                1e-12
                            )
                        )

                        row[
                            'blind_mae_gain_abs'
                        ] = round(
                            gain_abs,
                            6
                        )

                        row[
                            'blind_mae_gain_pct'
                        ] = round(
                            gain_pct,
                            4
                        )

                # =============================================
                # Output 全图 Operable Rate
                # =============================================

                total_pixels = (
                    h * w
                )

                full_err = (
                    out_gray.astype(
                        np.float64
                    )
                    -
                    gt_img.astype(
                        np.float64
                    )
                )

                operable_count = int(
                    (
                        np.abs(
                            full_err
                        )
                        <
                        operable_threshold
                    ).sum()
                )

                row[
                    'operable_rate'
                ] = round(
                    100.0
                    *
                    operable_count
                    /
                    total_pixels,
                    2
                )

                seq_stats[
                    'operable_count'
                ] += operable_count

                seq_stats[
                    'total_pixels'
                ] += total_pixels

                global_stats[
                    'operable_count'
                ] += operable_count

                global_stats[
                    'total_pixels'
                ] += total_pixels

                # =============================================
                # Input 全图 Operable Rate
                # =============================================

                input_full_err = (
                    in_gray.astype(
                        np.float64
                    )
                    -
                    gt_img.astype(
                        np.float64
                    )
                )

                input_operable_count = int(
                    (
                        np.abs(
                            input_full_err
                        )
                        <
                        operable_threshold
                    ).sum()
                )

                row[
                    'input_operable_rate'
                ] = round(
                    100.0
                    *
                    input_operable_count
                    /
                    total_pixels,
                    2
                )

                seq_stats[
                    'input_operable_count'
                ] += input_operable_count

                seq_stats[
                    'input_total_pixels'
                ] += total_pixels

                global_stats[
                    'input_operable_count'
                ] += input_operable_count

                global_stats[
                    'input_total_pixels'
                ] += total_pixels

                # =============================================
                # 日志
                # =============================================

                seq_logs.append(
                    row
                )

                per_image_logs.append(
                    row
                )

                if (
                    idx + 1
                ) % 10 == 0:

                    print(
                        f"Processed "
                        f"{idx + 1}/"
                        f"{len(group_files)}"
                    )

            # =================================================
            # 保存序列信息
            # =================================================

            seq_logs_all[
                group_name
            ] = seq_logs

            seq_stats_all[
                group_name
            ] = seq_stats

            # 序列逐帧 CSV
            if seq_logs:

                seq_csv = os.path.join(
                    save_eval,
                    f'test_blind_metrics_'
                    f'{group_name}.csv'
                )

                with open(
                    seq_csv,
                    'w',
                    encoding='utf-8',
                    newline=''
                ) as f:

                    writer = csv.DictWriter(
                        f,
                        fieldnames=keys
                    )

                    writer.writeheader()

                    writer.writerows(
                        seq_logs
                    )

                print(
                    f"Sequence metrics saved: "
                    f"{seq_csv}"
                )

    # ========================================================
    # 全局平均
    # ========================================================

    avg_psnr = safe_mean(
        global_stats[
            'psnr'
        ]
    )

    avg_ssim = safe_mean(
        global_stats[
            'ssim'
        ]
    )

    avg_input_psnr = safe_mean(
        global_stats[
            'input_psnr'
        ]
    )

    avg_input_ssim = safe_mean(
        global_stats[
            'input_ssim'
        ]
    )

    # ========================================================
    # 生成全局 AVERAGE
    # ========================================================

    def make_average_row(
        for_summary=False
    ):

        fields = (
            summary_keys
            if for_summary
            else keys
        )

        row = {
            k: None
            for k
            in fields
        }

        if for_summary:

            row[
                'seq'
            ] = 'AVERAGE'

            row[
                'images'
            ] = len(
                per_image_logs
            )

        else:

            row[
                'image'
            ] = 'AVERAGE'

            row[
                'seq'
            ] = ''

        if avg_psnr is not None:

            row[
                'psnr'
            ] = round(
                avg_psnr,
                4
            )

        if avg_ssim is not None:

            row[
                'ssim'
            ] = round(
                avg_ssim,
                6
            )

        if avg_input_psnr is not None:

            row[
                'input_psnr'
            ] = round(
                avg_input_psnr,
                4
            )

        if avg_input_ssim is not None:

            row[
                'input_ssim'
            ] = round(
                avg_input_ssim,
                6
            )

        # ----------------------------------------------------
        # Output blind
        # ----------------------------------------------------

        pix = global_stats[
            'blind_pix'
        ]

        if pix > 0:

            mae = (
                global_stats[
                    'blind_abs'
                ]
                /
                pix
            )

            mse = (
                global_stats[
                    'blind_sq'
                ]
                /
                pix
            )

            row[
                'blind_mae'
            ] = round(
                mae,
                6
            )

            row[
                'blind_rmse'
            ] = round(
                math.sqrt(
                    mse
                ),
                6
            )

            row[
                'blind_psnr'
            ] = round(
                blind_psnr_from_stats(
                    global_stats[
                        'blind_sq'
                    ],
                    pix
                ),
                4
            )

            row[
                'blind_count'
            ] = pix

            for t in blind_thresholds:

                field = (
                    f'blind_operable_rate_'
                    f'{threshold_tag(t)}'
                )

                count = (
                    global_stats[
                        'blind_operable_count'
                    ][t]
                )

                row[
                    field
                ] = round(
                    100.0
                    *
                    count
                    /
                    pix,
                    2
                )

        # ----------------------------------------------------
        # Input blind
        # ----------------------------------------------------

        input_pix = global_stats[
            'blind_pix_in'
        ]

        if input_pix > 0:

            input_mae = (
                global_stats[
                    'blind_abs_in'
                ]
                /
                input_pix
            )

            input_mse = (
                global_stats[
                    'blind_sq_in'
                ]
                /
                input_pix
            )

            row[
                'input_blind_mae'
            ] = round(
                input_mae,
                6
            )

            row[
                'input_blind_rmse'
            ] = round(
                math.sqrt(
                    input_mse
                ),
                6
            )

            row[
                'input_blind_psnr'
            ] = round(
                blind_psnr_from_stats(
                    global_stats[
                        'blind_sq_in'
                    ],
                    input_pix
                ),
                4
            )

            for t in blind_thresholds:

                field = (
                    f'input_blind_operable_rate_'
                    f'{threshold_tag(t)}'
                )

                count = (
                    global_stats[
                        'input_blind_operable_count'
                    ][t]
                )

                row[
                    field
                ] = round(
                    100.0
                    *
                    count
                    /
                    input_pix,
                    2
                )

            if (
                pix > 0
                and
                pix == input_pix
            ):

                output_mae = (
                    global_stats[
                        'blind_abs'
                    ]
                    /
                    pix
                )

                gain_abs = (
                    input_mae
                    -
                    output_mae
                )

                gain_pct = (
                    100.0
                    *
                    gain_abs
                    /
                    (
                        input_mae
                        +
                        1e-12
                    )
                )

                row[
                    'blind_mae_gain_abs'
                ] = round(
                    gain_abs,
                    6
                )

                row[
                    'blind_mae_gain_pct'
                ] = round(
                    gain_pct,
                    4
                )

        # ----------------------------------------------------
        # 全图有效像元率
        # ----------------------------------------------------

        if (
            global_stats[
                'total_pixels'
            ]
            > 0
        ):

            row[
                'operable_rate'
            ] = round(
                100.0
                *
                global_stats[
                    'operable_count'
                ]
                /
                global_stats[
                    'total_pixels'
                ],
                2
            )

        if (
            global_stats[
                'input_total_pixels'
            ]
            > 0
        ):

            row[
                'input_operable_rate'
            ] = round(
                100.0
                *
                global_stats[
                    'input_operable_count'
                ]
                /
                global_stats[
                    'input_total_pixels'
                ],
                2
            )

        return row

    # ========================================================
    # 全局逐帧 CSV
    # ========================================================

    global_csv = os.path.join(
        save_eval,
        'test_blind_metrics.csv'
    )

    with open(
        global_csv,
        'w',
        encoding='utf-8',
        newline=''
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(
            per_image_logs
        )

        writer.writerow(
            make_average_row(
                for_summary=False
            )
        )

    # ========================================================
    # 按序列汇总 CSV
    # ========================================================

    summary_csv = os.path.join(
        save_eval,
        'test_blind_summary_by_seq.csv'
    )

    with open(
        summary_csv,
        'w',
        encoding='utf-8',
        newline=''
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_keys
        )

        writer.writeheader()

        for seq_name in sorted(
            seq_logs_all,
            key=natural_sort_key
        ):

            st = seq_stats_all[
                seq_name
            ]

            logs = seq_logs_all[
                seq_name
            ]

            row = {
                k: None
                for k
                in summary_keys
            }

            row[
                'seq'
            ] = seq_name

            row[
                'images'
            ] = len(
                logs
            )

            # -----------------------------------------------
            # PSNR / SSIM
            # -----------------------------------------------

            seq_psnr = safe_mean(
                st[
                    'psnr'
                ]
            )

            seq_ssim = safe_mean(
                st[
                    'ssim'
                ]
            )

            seq_input_psnr = safe_mean(
                st[
                    'input_psnr'
                ]
            )

            seq_input_ssim = safe_mean(
                st[
                    'input_ssim'
                ]
            )

            if seq_psnr is not None:

                row[
                    'psnr'
                ] = round(
                    seq_psnr,
                    4
                )

            if seq_ssim is not None:

                row[
                    'ssim'
                ] = round(
                    seq_ssim,
                    6
                )

            if (
                seq_input_psnr
                is not None
            ):

                row[
                    'input_psnr'
                ] = round(
                    seq_input_psnr,
                    4
                )

            if (
                seq_input_ssim
                is not None
            ):

                row[
                    'input_ssim'
                ] = round(
                    seq_input_ssim,
                    6
                )

            # -----------------------------------------------
            # Output blind
            # -----------------------------------------------

            pix = st[
                'blind_pix'
            ]

            row[
                'blind_count'
            ] = pix

            if pix > 0:

                mae = (
                    st[
                        'blind_abs'
                    ]
                    /
                    pix
                )

                mse = (
                    st[
                        'blind_sq'
                    ]
                    /
                    pix
                )

                row[
                    'blind_mae'
                ] = round(
                    mae,
                    6
                )

                row[
                    'blind_rmse'
                ] = round(
                    math.sqrt(
                        mse
                    ),
                    6
                )

                row[
                    'blind_psnr'
                ] = round(
                    blind_psnr_from_stats(
                        st[
                            'blind_sq'
                        ],
                        pix
                    ),
                    4
                )

                for t in blind_thresholds:

                    field = (
                        f'blind_operable_rate_'
                        f'{threshold_tag(t)}'
                    )

                    row[
                        field
                    ] = round(
                        100.0
                        *
                        st[
                            'blind_operable_count'
                        ][t]
                        /
                        pix,
                        2
                    )

            # -----------------------------------------------
            # Input blind
            # -----------------------------------------------

            input_pix = st[
                'blind_pix_in'
            ]

            if input_pix > 0:

                input_mae = (
                    st[
                        'blind_abs_in'
                    ]
                    /
                    input_pix
                )

                input_mse = (
                    st[
                        'blind_sq_in'
                    ]
                    /
                    input_pix
                )

                row[
                    'input_blind_mae'
                ] = round(
                    input_mae,
                    6
                )

                row[
                    'input_blind_rmse'
                ] = round(
                    math.sqrt(
                        input_mse
                    ),
                    6
                )

                row[
                    'input_blind_psnr'
                ] = round(
                    blind_psnr_from_stats(
                        st[
                            'blind_sq_in'
                        ],
                        input_pix
                    ),
                    4
                )

                for t in blind_thresholds:

                    field = (
                        f'input_blind_operable_rate_'
                        f'{threshold_tag(t)}'
                    )

                    row[
                        field
                    ] = round(
                        100.0
                        *
                        st[
                            'input_blind_operable_count'
                        ][t]
                        /
                        input_pix,
                        2
                    )

                if (
                    pix > 0
                    and
                    pix == input_pix
                ):

                    output_mae = (
                        st[
                            'blind_abs'
                        ]
                        /
                        pix
                    )

                    gain_abs = (
                        input_mae
                        -
                        output_mae
                    )

                    gain_pct = (
                        100.0
                        *
                        gain_abs
                        /
                        (
                            input_mae
                            +
                            1e-12
                        )
                    )

                    row[
                        'blind_mae_gain_abs'
                    ] = round(
                        gain_abs,
                        6
                    )

                    row[
                        'blind_mae_gain_pct'
                    ] = round(
                        gain_pct,
                        4
                    )

            # -----------------------------------------------
            # 全图有效像元率
            # -----------------------------------------------

            if (
                st[
                    'total_pixels'
                ] > 0
            ):

                row[
                    'operable_rate'
                ] = round(
                    100.0
                    *
                    st[
                        'operable_count'
                    ]
                    /
                    st[
                        'total_pixels'
                    ],
                    2
                )

            if (
                st[
                    'input_total_pixels'
                ] > 0
            ):

                row[
                    'input_operable_rate'
                ] = round(
                    100.0
                    *
                    st[
                        'input_operable_count'
                    ]
                    /
                    st[
                        'input_total_pixels'
                    ],
                    2
                )

            writer.writerow(
                row
            )

        writer.writerow(
            make_average_row(
                for_summary=True
            )
        )

    # ========================================================
    # 最终控制台输出
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL NAFNET METRICS"
    )

    print(
        "=" * 80
    )

    if avg_psnr is not None:

        print(
            f"PSNR                         : "
            f"{avg_psnr:.4f} dB"
        )

    if avg_ssim is not None:

        print(
            f"SSIM                         : "
            f"{avg_ssim:.6f}"
        )

    pix = global_stats[
        'blind_pix'
    ]

    if pix > 0:

        blind_mae = (
            global_stats[
                'blind_abs'
            ]
            /
            pix
        )

        blind_mse = (
            global_stats[
                'blind_sq'
            ]
            /
            pix
        )

        blind_rmse = math.sqrt(
            blind_mse
        )

        blind_psnr = (
            blind_psnr_from_stats(
                global_stats[
                    'blind_sq'
                ],
                pix
            )
        )

        print(
            f"Blind MAE                    : "
            f"{blind_mae:.6f}"
        )

        print(
            f"Blind RMSE                   : "
            f"{blind_rmse:.6f}"
        )

        print(
            f"Blind PSNR                   : "
            f"{blind_psnr:.4f} dB"
        )

        for t in blind_thresholds:

            blind_operable_rate = (
                100.0
                *
                global_stats[
                    'blind_operable_count'
                ][t]
                /
                pix
            )

            print(
                f"Blind Operable Rate@{t:g}"
                f"         : "
                f"{blind_operable_rate:.2f}%"
            )

    if (
        global_stats[
            'total_pixels'
        ] > 0
    ):

        operable_rate = (
            100.0
            *
            global_stats[
                'operable_count'
            ]
            /
            global_stats[
                'total_pixels'
            ]
        )

        print(
            f"Operable Rate (full image)   : "
            f"{operable_rate:.2f}%"
        )

    print(
        "-" * 80
    )

    # Input结果

    if avg_input_psnr is not None:

        print(
            f"Input PSNR                   : "
            f"{avg_input_psnr:.4f} dB"
        )

    if avg_input_ssim is not None:

        print(
            f"Input SSIM                   : "
            f"{avg_input_ssim:.6f}"
        )

    input_pix = global_stats[
        'blind_pix_in'
    ]

    if input_pix > 0:

        input_mae = (
            global_stats[
                'blind_abs_in'
            ]
            /
            input_pix
        )

        input_mse = (
            global_stats[
                'blind_sq_in'
            ]
            /
            input_pix
        )

        input_rmse = math.sqrt(
            input_mse
        )

        input_bpsnr = (
            blind_psnr_from_stats(
                global_stats[
                    'blind_sq_in'
                ],
                input_pix
            )
        )

        print(
            f"Input Blind MAE              : "
            f"{input_mae:.6f}"
        )

        print(
            f"Input Blind RMSE             : "
            f"{input_rmse:.6f}"
        )

        print(
            f"Input Blind PSNR             : "
            f"{input_bpsnr:.4f} dB"
        )

        for t in blind_thresholds:

            rate = (
                100.0
                *
                global_stats[
                    'input_blind_operable_count'
                ][t]
                /
                input_pix
            )

            print(
                f"Input Blind Operable Rate@{t:g}"
                f"   : "
                f"{rate:.2f}%"
            )

    if (
        global_stats[
            'input_total_pixels'
        ] > 0
    ):

        input_operable_rate = (
            100.0
            *
            global_stats[
                'input_operable_count'
            ]
            /
            global_stats[
                'input_total_pixels'
            ]
        )

        print(
            f"Input Operable Rate           : "
            f"{input_operable_rate:.2f}%"
        )

    print(
        "-" * 80
    )

    print(
        f"Blind pixels used             : "
        f"{pix}"
    )

    print(
        f"Global CSV                    : "
        f"{global_csv}"
    )

    print(
        f"Summary CSV                   : "
        f"{summary_csv}"
    )

    print(
        f"Saved outputs                 : "
        f"{save_pure}"
    )

    print(
        "=" * 80
    )


if __name__ == '__main__':
    main()
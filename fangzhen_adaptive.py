import cv2
import numpy as np
import random
import os
import glob
import csv
import re

# ========================= 全局设置 =========================
DIFFICULTY = 2           # 1.0=原始；1.3=中等；1.5=较难；2.0=很难
FLASH_MIN_ACTIVE_RATIO = 0.65 # 每帧至少激活多少比例的闪元


# ========================= 辅助函数 =========================
def get_random_dark_color():
    """通用暗色采样 (0-100灰度)"""
    return random.randint(0, 10) if random.random() < 0.3 else random.randint(10, 100)


def get_mostly_black_color():
    """暗主导块深黑色采样 (0-15灰度)"""
    return random.randint(0, 5) if random.random() < 0.9 else random.randint(5, 15)


def natural_sort_key(s):
    """自然排序，例如 1.png, 2.png, 10.png"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def renumber_images_from_one(src_dir, img_paths, mask_dir):
    """
    将源图像按自然顺序重新编号为 1,2,3,...
    使用两阶段重命名避免文件名冲突。
    同时保存 filename_mapping.csv。
    """
    if not img_paths:
        return img_paths

    img_paths = sorted(img_paths, key=lambda p: natural_sort_key(os.path.basename(p)))

    already_numbered = all(
        os.path.splitext(os.path.basename(p))[0] == str(i)
        for i, p in enumerate(img_paths, start=1)
    )
    if already_numbered:
        print(f"[编号检查] {src_dir} 已连续编号为 1~{len(img_paths)}，无需修改。")
        return img_paths

    print(f"[重新编号] {src_dir}：共 {len(img_paths)} 张")

    # 第一阶段：全部改成临时文件名，避免 3.png -> 1.png 时发生冲突
    temp_records = []
    for i, old_path in enumerate(img_paths, start=1):
        old_name = os.path.basename(old_path)
        _, ext = os.path.splitext(old_name)
        temp_path = os.path.join(src_dir, f"__renumber_tmp__{i:06d}{ext}")

        counter = 0
        while os.path.exists(temp_path):
            counter += 1
            temp_path = os.path.join(src_dir, f"__renumber_tmp__{i:06d}_{counter}{ext}")

        os.rename(old_path, temp_path)
        temp_records.append((temp_path, old_name, ext))

    # 第二阶段：重新编号为 1,2,3,...
    new_paths, mapping_records = [], []
    for i, (temp_path, old_name, ext) in enumerate(temp_records, start=1):
        new_name = f"{i}{ext}"
        new_path = os.path.join(src_dir, new_name)

        if os.path.exists(new_path):
            raise RuntimeError(f"重新编号失败，目标文件已存在：{new_path}")

        os.rename(temp_path, new_path)
        new_paths.append(new_path)
        mapping_records.append([i, old_name, new_name])

    # 保存映射表
    os.makedirs(mask_dir, exist_ok=True)
    mapping_csv = os.path.join(mask_dir, "filename_mapping.csv")
    with open(mapping_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "original_name", "new_name"])
        writer.writerows(mapping_records)

    new_paths = sorted(new_paths, key=lambda p: natural_sort_key(os.path.basename(p)))
    print(f"[重新编号] 完成：1~{len(new_paths)}")
    print(f"[重新编号] 映射表：{mapping_csv}")
    return new_paths


# ========================= 盲元生成 =========================
def gen_mostly_black_tight_blob(w, h, target_pts, dominant_type, forbidden, margin_block):
    """生成高度粘合的白主导/黑主导块"""
    pts_dict = {}

    for _ in range(50):
        tx = random.randint(margin_block, w - margin_block)
        ty = random.randint(margin_block, h - margin_block)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx = random.randint(margin_block, w - margin_block)
        cy = random.randint(margin_block, h - margin_block)

    current_pts = [(cx, cy)]
    pts_dict[(cx, cy)] = 255 if dominant_type == "white" else get_mostly_black_color()

    while len(pts_dict) < target_pts:
        base_x, base_y = random.choice(current_pts)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = base_x + dx, base_y + dy

        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in pts_dict:
            if dominant_type == "white":
                c = 255 if random.random() < 0.8 else get_random_dark_color()
            else:
                c = get_mostly_black_color() if random.random() < 0.90 else 255
            pts_dict[(nx, ny)] = c
            current_pts.append((nx, ny))

    return [(x, y, c) for (x, y), c in pts_dict.items()], (cx, cy)


def gen_extra_long_lines(w, h, line_start_margin, line_len_min, line_len_max):
    """生成超长破碎线"""
    pts = []

    for _ in range(random.randint(2, 3)):
        cx = random.randint(line_start_margin, w // 2)
        cy = random.randint(line_start_margin, h // 2)
        dx, dy = random.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)])
        length = random.randint(line_len_min, line_len_max)

        for _ in range(length):
            if 5 <= cx < w - 5 and 5 <= cy < h - 5:
                c = get_random_dark_color() if random.random() < 0.4 else 255
                pts.append((cx, cy, c))

            if random.random() < 0.05:
                sx = cx + random.randint(-1, 1)
                sy = cy + random.randint(-1, 1)
                if 5 <= sx < w - 5 and 5 <= sy < h - 5:
                    pts.append((sx, sy, get_random_dark_color()))

            cx += dx
            cy += dy
            if not (0 <= cx < w and 0 <= cy < h):
                break

    return pts


def grow_compact_blob(w, h, target_w, target_d, forbidden, margin_block):
    """大型/中型块：白色核心 + 暗色污染边缘"""
    pts = []

    for _ in range(50):
        tx = random.randint(margin_block, w - margin_block)
        ty = random.randint(margin_block, h - margin_block)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx = random.randint(margin_block, w - margin_block)
        cy = random.randint(margin_block, h - margin_block)

    # 白色核心
    w_pts = {(cx, cy)}
    bnd = [(cx, cy)]

    while len(w_pts) < target_w and bnd:
        px, py = random.choice(bnd)
        dirs = ([(1, 0)] * 4 + [(-1, 0)] * 4 +
                [(0, 1)] * 4 + [(0, -1)] * 4 + [(1, 1)])
        dx, dy = random.choice(dirs)
        nx, ny = px + dx, py + dy

        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in w_pts:
            w_pts.add((nx, ny))
            bnd.append((nx, ny))

        if len(bnd) > target_w // 2:
            bnd.pop(0)

    # 暗色外围
    d_pts = set()
    bnd_d = list(w_pts)

    while len(d_pts) < target_d and bnd_d:
        px, py = random.choice(bnd_d)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])
        nx, ny = px + dx, py + dy

        if (5 <= nx < w - 5 and 5 <= ny < h - 5 and
                (nx, ny) not in w_pts and (nx, ny) not in d_pts):
            d_pts.add((nx, ny))
            bnd_d.append((nx, ny))

    for x, y in d_pts:
        pts.append((x, y, get_random_dark_color()))

    for x, y in w_pts:
        c = get_random_dark_color() if random.random() < 0.05 else 255
        pts.append((x, y, c))

    return pts, (cx, cy)


def gen_cross_invalid_pixels(w, h, num_crosses, forbidden, occupied, margin_cross, retry=80):
    """生成十字形盲元"""
    pts, centers = [], []
    arms = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    for _ in range(num_crosses):
        cx = cy = None

        for _ in range(retry):
            tx = random.randint(margin_cross, w - 1 - margin_cross)
            ty = random.randint(margin_cross, h - 1 - margin_cross)

            if not (1 <= tx < w - 1 and 1 <= ty < h - 1):
                continue
            if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
                continue

            cross_coords = [(tx + dx, ty + dy) for dx, dy in arms]
            if any((x, y) in occupied for x, y in cross_coords):
                continue

            cx, cy = tx, ty
            break

        if cx is None:
            continue

        pts.append((cx, cy, random.randint(0, 5)))
        occupied.add((cx, cy))

        for dx, dy in arms[1:]:
            x, y = cx + dx, cy + dy
            pts.append((x, y, random.randint(100, 150)))
            occupied.add((x, y))

        centers.append((cx, cy))

    return pts, centers, occupied


def gen_flash_anchor_positions(w, h, num_anchors, forbidden, occupied,
                               margin_flash, min_dist, block_size=1, retry=120):
    """生成闪元候选位置"""
    anchors = []
    tries = 0

    while len(anchors) < num_anchors and tries < retry * max(1, num_anchors):
        tries += 1
        tx = random.randint(margin_flash, w - 1 - margin_flash)
        ty = random.randint(margin_flash, h - 1 - margin_flash)

        block_coords = [
            (tx + dx, ty + dy)
            for dx in range(block_size)
            for dy in range(block_size)
        ]

        if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            continue
        if any((x, y) in occupied for x, y in block_coords):
            continue
        if any(abs(tx - ax) < min_dist and abs(ty - ay) < min_dist for ax, ay in anchors):
            continue

        anchors.append((tx, ty))
        occupied.update(block_coords)

    return anchors, occupied


def sample_flash_pixels_for_frame(img, flash_anchors, min_active, max_active, block_size=1):
    """每帧随机激活闪元"""
    if not flash_anchors:
        return [], []

    active_num = min(len(flash_anchors), random.randint(min_active, max_active))
    active_positions = random.sample(flash_anchors, active_num)
    pts, records = [], []

    for x, y in active_positions:
        if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
            continue

        orig_gray = int(img[y, x][0])

        # 暂时保持原始灰度破坏方式，只增加空间/数量难度
        if random.random() < 0.5:
            delta = random.randint(40, 120)
            c = random.randint(0, 30) if random.random() < 0.7 else max(0, orig_gray - delta)
            mode = "darken"
        else:
            delta = random.randint(40, 120)
            c = random.randint(225, 255) if random.random() < 0.7 else min(255, orig_gray + delta)
            mode = "brighten"

        for dx in range(block_size):
            for dy in range(block_size):
                px, py = x + dx, y + dy
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    pts.append((px, py, c))

        records.append([x, y, orig_gray, c, mode])

    return pts, records


# ========================= 主仿真函数 =========================
def run_consistent_simulation(src_dir, dst_dir, mask_dir):
    """
    根据图像尺寸自动调整盲元规模。
    DIFFICULTY 控制整体难度。
    同时将 sharp 图像编号统一为 1,2,3,...。
    """
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # ---------- 读取并自然排序 ----------
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp",
            "*.PNG", "*.JPG", "*.JPEG", "*.BMP")
    img_paths = []

    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(src_dir, ext)))

    img_paths = sorted(
        set(img_paths),
        key=lambda p: natural_sort_key(os.path.basename(p))
    )

    if not img_paths:
        print(f"警告：{src_dir} 中没有支持的图片文件")
        return

    # sharp 统一重新编号
    img_paths = renumber_images_from_one(src_dir, img_paths, mask_dir)

    first_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        print(f"无法读取第一张图像：{img_paths[0]}")
        return

    h, w = first_img.shape[:2]

    print("=" * 70)
    print(f"图像尺寸：{w} × {h}")
    print(f"总帧数：{len(img_paths)}")
    print(f"仿真难度：DIFFICULTY = {DIFFICULTY}")
    print(f"闪元最低激活比例：{FLASH_MIN_ACTIVE_RATIO:.0%}")
    print("=" * 70)

    # ---------- 自适应参数 ----------
    ref_w, ref_h = 640, 512
    ref_short = min(ref_w, ref_h)
    ref_area = ref_w * ref_h

    short = min(w, h)
    area = w * h
    size_ratio = short / ref_short
    area_ratio = area / ref_area

    # 数量随 DIFFICULTY 线性增加
    base_cnt1 = max(150, int(800 * area_ratio * DIFFICULTY))
    base_cnt2 = max(400, int(2000 * area_ratio * DIFFICULTY))

    # 块尺寸采用较温和的增长，避免难度增加过猛
    block_scale = DIFFICULTY ** 0.7
    tight_target = max(8, int(32 * area_ratio * block_scale))
    large_w = max(15, int(130 * area_ratio * block_scale))
    large_d = max(15, int(150 * area_ratio * block_scale))
    medium_w = max(10, int(45 * area_ratio * block_scale))
    medium_d = max(10, int(60 * area_ratio * block_scale))

    # 长线
    margin_block = max(20, int(120 * size_ratio))
    line_start_margin = max(10, int(30 * size_ratio))
    line_len_min = max(40, int(120 * size_ratio * block_scale))
    line_len_max = max(60, int(180 * size_ratio * block_scale))

    # 十字盲元
    cross_num = max(8, int(40 * area_ratio * DIFFICULTY))
    margin_cross = max(4, int(8 * size_ratio))

    # 动态闪元
    flash_num = max(3, int(10 * area_ratio * DIFFICULTY))
    margin_flash = max(4, int(8 * size_ratio))

    # 难度越高，不同异常结构可更加靠近
    overlap_scale = 1.0 / (DIFFICULTY ** 0.25)
    init_forbidden_radius = max(1, int(180 * size_ratio * overlap_scale))
    blob_forbidden_radius = max(1, int(80 * size_ratio * overlap_scale))
    large_forbidden_radius = max(1, int(100 * size_ratio * overlap_scale))
    cross_forbidden_radius = max(1, int(20 * size_ratio * overlap_scale))
    flash_forbidden_radius = max(1, int(12 * size_ratio * overlap_scale))

    print(
        f"散点参数：{base_cnt1}, {base_cnt2} | "
        f"大块：{large_w}+{large_d} | "
        f"中块：{medium_w}+{medium_d} | "
        f"十字：{cross_num} | 闪元候选：{flash_num}"
    )

    # ---------- 生成静态盲元 ----------
    all_static_blind_params = []
    forbidden = [(w // 4, h // 4, init_forbidden_radius)]

    # 1. 基础散布点
    for wl, hl, cnt in [(w, h, base_cnt1), (w // 2, h // 2, base_cnt2)]:
        for _ in range(cnt):
            tx = random.randint(0, wl - 2)
            ty = random.randint(0, hl - 2)

            all_static_blind_params.append(
                (tx, ty, get_random_dark_color())
            )

            all_static_blind_params.append((
                tx + random.choice([0, 1]),
                ty + random.choice([0, 1]),
                255
            ))

    # 2. 超长破碎线
    all_static_blind_params += gen_extra_long_lines(
        w, h, line_start_margin, line_len_min, line_len_max
    )

    # 3. 粘合不规则块：两白两黑
    for dominant_type in ["white", "white", "dark", "dark"]:
        p, center = gen_mostly_black_tight_blob(
            w, h, tight_target, dominant_type, forbidden, margin_block
        )
        all_static_blind_params += p
        forbidden.append((center[0], center[1], blob_forbidden_radius))

    # 4. 大型 / 中型块
    for wt, dt, n in [(large_w, large_d, 2), (medium_w, medium_d, 2)]:
        for _ in range(n):
            p, center = grow_compact_blob(
                w, h, wt, dt, forbidden, margin_block
            )
            all_static_blind_params += p
            forbidden.append((center[0], center[1], large_forbidden_radius))

    # 5. 十字盲元
    occupied = {(x, y) for x, y, _ in all_static_blind_params}

    cross_pts, cross_centers, occupied = gen_cross_invalid_pixels(
        w, h, cross_num, forbidden, occupied, margin_cross, retry=80
    )

    all_static_blind_params += cross_pts

    for cx, cy in cross_centers:
        forbidden.append((cx, cy, cross_forbidden_radius))

    # 6. 动态闪元候选
    flash_anchors, occupied = gen_flash_anchor_positions(
        w, h, flash_num, forbidden, occupied,
        margin_flash, min_dist=24, block_size=1, retry=120
    )

    for ax, ay in flash_anchors:
        forbidden.append((ax, ay, flash_forbidden_radius))

    # ---------- 渲染 ----------
    mask_img = np.zeros((h, w), dtype=np.uint8)
    csv_records = []
    flash_records = []
    successfully_processed = 0

    for idx, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)

        if img is None:
            print(f"无法读取图像：{p}，跳过")
            continue

        if img.shape[:2] != (h, w):
            raise ValueError(
                f"图像尺寸不一致：{p}\n"
                f"期望：{w} × {h}\n"
                f"实际：{img.shape[1]} × {img.shape[0]}"
            )

        out_img = img.copy()

        # 静态盲元
        for x, y, c in all_static_blind_params:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = [c, c, c]

                if idx == 0:
                    mask_img[y, x] = 255
                    orig_gray = int(img[y, x][0])
                    csv_records.append([x, y, orig_gray, c])

        # 动态闪元
        min_active = max(1, int(flash_num * FLASH_MIN_ACTIVE_RATIO))
        max_active = flash_num

        flash_pts, frame_flash_records = sample_flash_pixels_for_frame(
            img,
            flash_anchors,
            min_active=min_active,
            max_active=max_active,
            block_size=1
        )

        for x, y, c in flash_pts:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = [c, c, c]

        # 输出统一为 PNG
        frame_index = os.path.splitext(os.path.basename(p))[0]
        out_filename = f"{frame_index}.png"
        output_path = os.path.join(dst_dir, out_filename)

        # flash CSV 使用输出文件名，确保始终与 blur 文件名一致
        for rec in frame_flash_records:
            flash_records.append([out_filename] + rec)

        if not cv2.imwrite(output_path, out_img):
            raise RuntimeError(f"图像保存失败：{output_path}")

        successfully_processed += 1
        print(
            f"[{successfully_processed:4d}/{len(img_paths):4d}] "
            f"{os.path.basename(p)} -> {out_filename}"
        )

    # ---------- 保存 Mask / CSV ----------
    mask_path = os.path.join(mask_dir, "blind_pixel_mask.png")
    static_csv_path = os.path.join(mask_dir, "blind_pixel_coords.csv")
    flash_csv_path = os.path.join(mask_dir, "flash_pixel_coords.csv")

    cv2.imwrite(mask_path, mask_img)

    with open(static_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "original_gray", "simulated_gray"])
        writer.writerows(csv_records)

    if flash_records:
        with open(flash_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_name", "x", "y",
                "original_gray", "simulated_gray", "mode"
            ])
            writer.writerows(flash_records)

    # ---------- 汇总 ----------
    unique_static = len({(x, y) for x, y, _ in all_static_blind_params})
    static_ratio = 100.0 * unique_static / (w * h)

    print("=" * 70)
    print(f"仿真完成：{successfully_processed} 张")
    print(f"仿真难度：{DIFFICULTY}")
    print(f"Sharp 编号：1 ~ {len(img_paths)}")
    print(f"Blur 编号：1 ~ {successfully_processed}")
    print(f"静态盲元唯一像素数：{unique_static}")
    print(f"静态盲元占比：{static_ratio:.4f}%")
    print(f"闪元候选点数：{len(flash_anchors)}")
    print(f"每帧闪元激活范围：{min_active} ~ {max_active}")
    print(f"Blind Mask：{mask_path}")
    print(f"静态盲元 CSV：{static_csv_path}")
    if flash_records:
        print(f"闪元 CSV：{flash_csv_path}")
    print("=" * 70)


# ========================= 主程序 =========================
if __name__ == "__main__":
    DATA_BASE = r"/home/student_server/Qtt/NAFNet/data"
    SEQ = "006"

    run_consistent_simulation(
        src_dir=os.path.join(DATA_BASE, "test_sharp", "005"),
        dst_dir=os.path.join(DATA_BASE, "test_blur", "005"),
        mask_dir=os.path.join(DATA_BASE, "test_mask", "005")
    )

    print("所有仿真任务完成。")
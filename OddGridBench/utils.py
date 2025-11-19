import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import os
import json
import numpy
import shutil
import random
import math
import numpy as np
from PIL import Image, ImageDraw
import itertools
import cv2

def generate_lab_color(l_range=(20, 70), a_range=(-40, 40), b_range=(-40, 40)):
    """随机生成一个更深的 LAB 颜色（避免白色或过亮），保留两位小数"""
    L = np.random.uniform(*l_range)
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    return np.round(np.array([L, a, b]), 2)


def perturb_color(base_lab, target_delta_e, step=1.0, max_iter=5000, tol=0.5):
    """
    生成与 base_lab 相差约 target_delta_e 的颜色 (ΔE2000)，保留两位小数
    """
    best_candidate = base_lab
    best_diff = 1e9
    for _ in range(max_iter):
        candidate = base_lab + np.random.uniform(-step, step, 3) * target_delta_e
        dE = color.deltaE_ciede2000(
            base_lab[np.newaxis, :], candidate[np.newaxis, :]
        )[0]
        diff = abs(dE - target_delta_e)
        if diff < best_diff:
            best_diff = diff
            best_candidate = candidate
        if diff < tol:
            break
    return np.round(best_candidate, 2)

def lab_to_rgb(lab):
    """LAB 转 RGB，并裁剪到 [0,1]"""
    rgb = color.lab2rgb(lab[np.newaxis, np.newaxis, :])
    return np.clip(rgb[0, 0, :], 0, 1)

def ensure_dirs(data_type: str):
    """
    创建 难度/image 与 难度/metadata 目录
    - 若已存在，则先清空再重新创建
    """
    img_dir = os.path.join(data_type, "image")
    meta_dir = os.path.join(data_type, "metadata")
    img_red_dir = os.path.join(data_type, "image_red")
    image_with_number_dir = os.path.join(data_type, "image_number")
    


    # 如果存在旧目录则先删除
    for d in [img_dir, meta_dir, img_red_dir, image_with_number_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print(f"Creating directories '{img_dir}', {img_red_dir}, and '{meta_dir}'...")

    # 重新创建空目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(img_red_dir, exist_ok=True)
    os.makedirs(image_with_number_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    

    return img_dir, meta_dir

def save_image_as_png(image: np.ndarray, path: str):
    """保存为 PNG 文件"""
    plt.imsave(path, image)


def save_pair(image, meta, img_dir, meta_dir, index, img_with_number, draw_bbox=False):
    img_name = f"image_{index}.png"
    meta_name = f"metadata_{index}.json"

    img_path = os.path.join(img_dir, img_name)
    meta_path = os.path.join(meta_dir, meta_name)

    meta = dict(meta)  # 复制一份
    meta["image_file"] = os.path.join("image", img_name)
    meta["metadata_file"] = os.path.join("metadata", meta_name)
    save_image_as_png(image, img_path)

    # ✅ 在这里画框
    if draw_bbox and "odd_bbox" in meta:
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        x, y, w, h = meta["odd_bbox"]["x"], meta["odd_bbox"]["y"], meta["odd_bbox"]["w"], meta["odd_bbox"]["h"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        image = np.asarray(img_pil, dtype=np.float32) / 255.0
        image_red_dir = img_dir.replace("image", "image_red")
        os.makedirs(image_red_dir, exist_ok=True)
        save_image_as_png(image, os.path.join(image_red_dir, img_name))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    if img_with_number is not None:
        image_with_number_dir = img_dir.replace("image", "image_number")
        os.makedirs(image_with_number_dir, exist_ok=True)
        save_image_as_png(img_with_number, os.path.join(image_with_number_dir, img_name))

    
    

def apply_odd_variations(base_shape, base_lab, base_rgb, base_angle, block_size, odd_type, args):
    """根据 odd_type 生成 odd 属性"""
    odd_lab, odd_rgb, odd_block_size, odd_angle = base_lab.copy(), base_rgb, block_size, base_angle

    if "color" in odd_type:
        odd_lab = perturb_color(base_lab, args.de)
        odd_rgb = lab_to_rgb(odd_lab)

    if "size" in odd_type:
        odd_block_size = int(round(block_size * args.size_ratio))
        # 随机决定是 odd 变大还是 odd 变小
        # if random.randint(0, 1) == 0:
            # block_size, odd_block_size = odd_block_size, block_size

    if "rotation" in odd_type:
        odd_angle = abs(base_angle + args.angle_sacle) % 360
        # if random.randint(0, 1) == 0:
            # base_angle, odd_angle = odd_angle, base_angle

    return odd_lab, odd_rgb, odd_block_size, block_size, base_angle, odd_angle

def get_block_position(i, j, block_size, gap, margin, img_w, img_h,
                       idx, odd_pos, odd_block_size, odd_type, args):
    """计算 block 的放置位置"""
    y0 = margin + i * (block_size + gap)
    x0 = margin + j * (block_size + gap)

    # odd_type: position 偏移
    if idx == odd_pos and "position" in odd_type:
        x0 = x0 + args.dx
        y0 = y0 + args.dy
 

    return int(x0), int(y0)


def rotate_block_keep_full(img_np, angle, bgcolor=(1,1,1)):
    """
    旋转整个 block，保持完整，不裁剪，绕小 block 中心旋转
    """
    block_size = img_np.shape[0]

    # numpy -> PIL
    img = Image.fromarray((img_np * 255).astype(np.uint8))

    # 扩大 √2 倍画布，保证旋转后完全容纳
    padded_size = int(math.ceil(block_size * math.sqrt(2)))
    big_img = Image.new("RGB", (padded_size, padded_size),
                        tuple(int(c*255) for c in bgcolor))

    # 居中放置小 block
    offset = (padded_size - block_size) // 2
    big_img.paste(img, (offset, offset))

    # 旋转：绕大画布中心，其实就是绕小 block中心
    rotated = big_img.rotate(angle, resample=Image.BILINEAR, expand=False,
                             fillcolor=tuple(int(c*255) for c in bgcolor))

    return np.asarray(rotated, dtype=np.float32) / 255.0

def compute_min_gap_rotation(block_size, base_angle, odd_angle):
    def scale(angle):
        theta = math.radians(angle % 180)   # 旋转对称性，周期 180°
        return abs(math.cos(theta)) + abs(math.sin(theta))

    f1 = scale(base_angle)
    f2 = scale(odd_angle)

    max_scale = max(f1, f2)  # 找放大效果最大的
    min_gap = math.ceil(block_size * (max_scale - 1)) + 2
    return min_gap


def get_safe_gap(block_size, odd_type, gap, args):
    min_gap = 0  # 默认值
    size_gap = 0
    if "size" in odd_type:
        block_size = math.ceil(max(block_size, block_size * args.size_ratio))
        size_gap = math.ceil(max(block_size * args.size_ratio - block_size, 0))

    # --- 单独情况 ---
    if "rotation" in odd_type and "size" not in odd_type and "position" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle)

    elif "size" in odd_type and "rotation" not in odd_type and "position" not in odd_type:
        min_gap = size_gap

    elif "position" in odd_type and "rotation" not in odd_type and "size" not in odd_type:
        min_gap = max(abs(args.dx), abs(args.dy))

    # --- 两两组合 ---
    elif "rotation" in odd_type and "size" in odd_type and "position" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) + size_gap

    elif "rotation" in odd_type and "position" in odd_type and "size" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) + max(abs(args.dx), abs(args.dy))

    elif "size" in odd_type and "position" in odd_type and "rotation" not in odd_type:
        min_gap = max(abs(args.dx), abs(args.dy)) + size_gap

    # --- 三个都要 ---
    elif "rotation" in odd_type and "size" in odd_type and "position" in odd_type:
        min_gap = (compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) +
                   size_gap +
                   max(abs(args.dx), abs(args.dy)))

    # --- gap 修正 ---
    if gap < min_gap:
        gap = min_gap
    return gap



def random_background_color(prob_white: float = 0.5,
                            light_range: tuple = (0.8, 1.0),
                            smooth: bool = True):
    """
    随机生成背景颜色（RGB）
    -------------------------------------
    Args:generate_odd_type_list
        prob_white: 保持白色背景的概率（默认 0.5）
        light_range: 淡色通道取值范围（默认 0.8~1.0）
        smooth: 若为 True，则三通道相差较小，颜色柔和

    Returns:
        tuple: (r, g, b)，范围在 [0, 1]
    """

    # 以 prob_white 概率返回白色
    if random.random() < prob_white:
        return (1.0, 1.0, 1.0)

    lo, hi = light_range

    if smooth:
        # 柔和淡色：三通道在 base±variation 范围内微调
        base = random.uniform(lo, hi)
        variation = 0.05
        r = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
        g = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
        b = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
    else:
        # 独立随机通道
        r = round(random.uniform(lo, hi), 2)
        g = round(random.uniform(lo, hi), 2)
        b = round(random.uniform(lo, hi), 2)

    return (r, g, b)



def generate_odd_type_list (base_types, total_number: int):
    odd_type_list = []

    # 平均分成7份
    n_per_group = total_number // 7
    remainder = total_number % 7  # 多退少补部分

    # 1️⃣ 单类别组合
    for t in base_types:
        odd_type_list.extend([[t]] * n_per_group)

    # 2️⃣ 随机两个类别组合
    two_combos = list(itertools.combinations(base_types, 2))
    for combo in random.choices(two_combos, k=n_per_group):
        odd_type_list.append(list(combo))

    # 3️⃣ 随机三个类别组合
    three_combos = list(itertools.combinations(base_types, 3))
    for combo in random.choices(three_combos, k=n_per_group):
        odd_type_list.append(list(combo))

    # 4️⃣ 四个类别组合
    four_combo = [base_types]
    odd_type_list.extend([four_combo[0]] * n_per_group)

    # 5️⃣ 补齐或裁剪到 total_number
    while len(odd_type_list) < total_number:
        odd_type_list.append(random.choice(odd_type_list))
    odd_type_list = odd_type_list[:total_number]

    random.shuffle(odd_type_list)
    return odd_type_list

def add_row_col_numbers(img, grid_size, block_size, gap, margin, background_rgb):
    import cv2, numpy as np
    h, w = grid_size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, block_size / 100)
    font_thickness = max(1, int(block_size / 30))
    text_color = (0, 0, 0) if np.mean(background_rgb) > 0.5 else (1, 1, 1)
    text_color = tuple(int(c * 255) for c in text_color)

    # --- 列编号（上方） ---
    for j in range(w):
        x0 = margin + j * (block_size + gap) + block_size // 2
        y0 = int(margin * 0.6)
        # 横向稍微往右偏移（+0.2），避免偏左
        x_shift = int(block_size * 0.1)
        # 不动纵向，因为是横行
        cv2.putText(
            img, str(j + 1),
            (x0 - int(block_size * 0.1) + x_shift, y0),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA
        )

    # --- 行编号（左侧） ---
    for i in range(h):
        x0 = int(margin * 0.3)
        y0 = margin + i * (block_size + gap) + block_size // 2
        # 纵向稍微往下偏移（+0.25），避免偏上
        y_shift = int(block_size * 0.2)
        cv2.putText(
            img, str(i + 1),
            (x0, y0 + y_shift),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA
        )

    return img




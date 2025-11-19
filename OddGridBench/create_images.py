import numpy as np
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
from configs import configs, randomize_config
from shapes import draw_random_shape, draw_shape_by_name


def generate_odd_one_out_image(
    grid_size,
    block_size,
    gap,
    delta_e,
    max_image_size,
    margin,
    background_rgb,
    odd_type,
    args
):
    h, w = grid_size
    gap = get_safe_gap(block_size, odd_type, gap, args)

    # 缩放 block 和 gap
    core_h = h * block_size + (h - 1) * gap
    core_w = w * block_size + (w - 1) * gap
    avail_h = max_image_size[0] - 2 * margin
    avail_w = max_image_size[1] - 2 * margin
    scale = min(avail_h / core_h, avail_w / core_w, 1.0)
    block_size = max(1, int(round(block_size * scale)))
    gap = max(0, int(round(gap * scale)))

    # base 属性
    base_lab = generate_lab_color()
    base_rgb = lab_to_rgb(base_lab)
    _, base_shape = draw_random_shape(
        block_size,
        color=base_rgb,
        bgcolor=background_rgb,
        exclude=args.rotation_banned if "rotation" in odd_type else None,
    )

    # odd 属性
    odd_lab, odd_rgb, odd_block_size, block_size, base_angle, odd_angle = apply_odd_variations(
        base_shape, base_lab, base_rgb, args.base_angle, block_size, odd_type, args
    )
    odd_pos = np.random.randint(0, h * w)
    odd_row, odd_col = odd_pos // w + 1, odd_pos % w + 1

    # 画布
    core_h = h * block_size + (h - 1) * gap
    core_w = w * block_size + (w - 1) * gap
    multi_mar = 3 if "rotation" in odd_type else 2
    img_h, img_w = core_h + multi_mar * margin, core_w + multi_mar * margin
    img = np.ones((img_h, img_w, 3), dtype=np.float32) * np.array(background_rgb, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if idx == odd_pos:
                block_img, _ = draw_shape_by_name(base_shape, odd_block_size, color=odd_rgb, bgcolor=background_rgb)
                if "rotation" in odd_type:
                    block_img = rotate_block_keep_full(block_img, odd_angle, background_rgb)
            else:
                block_img, _ = draw_shape_by_name(base_shape, block_size, color=base_rgb, bgcolor=background_rgb)
                if "rotation" in odd_type:
                    block_img = rotate_block_keep_full(block_img, base_angle, background_rgb)

            x0, y0 = get_block_position(
                i, j, block_size, gap, margin, img_w, img_h,
                idx, odd_pos, odd_block_size, odd_type, args
            )
            if idx == odd_pos:
                odd_bbox = {
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(block_img.shape[1]),
                    "h": int(block_img.shape[0]),
                }

            img[y0:y0+block_img.shape[0], x0:x0+block_img.shape[1], :] = block_img
            
    img_clean = img.copy()  # 保留原始图像
    if args.rowcol_image:
        img_with_number = img.copy()  # 副本用于加行列编号
        img_with_number = add_row_col_numbers(img_with_number, (h, w), block_size, gap, margin, background_rgb)
    else:
        img_with_number = None

    meta = {
        "grid_size": [h, w],
        "block_size": block_size,
        "odd_block_size": odd_block_size,
        "gap": gap,
        "margin": margin,
        "odd_position": {"row": odd_row, "col": odd_col},
        "odd_bbox": odd_bbox,
        "odd_type": odd_type,
        "base_color_lab": [float(x) for x in base_lab],
        "odd_color_lab": [float(x) for x in odd_lab],
        "category": base_shape.split("(&)")[0],
        "base_shape": base_shape.split("(&)")[1].lower(),
        "angle_sacle": getattr(args, "angle_sacle", "N/A") if "rotation" in odd_type else "N/A",
        "size_ratio": getattr(args, "size_ratio", "N/A") if "size" in odd_type else "N/A",
        "color_delta_e": float(delta_e) if "color" in odd_type else "N/A",
        "dx_dy": f"[{getattr(args,'dx',0)},{getattr(args,'dy',0)}]" if "position" in odd_type else "N/A",
        "image_size": f"{[img_h, img_w]}",
    }
    return img_clean, img_with_number, meta


def generate_single(idx, args, img_dir, meta_dir):
    args_copy = copy.deepcopy(args)
    cfg = randomize_config(configs)
    for k, v in cfg.items():
        setattr(args_copy, k, v)
    args_copy.de = cfg["de"]

    # 初始化缺失字段
    for k, v in {
        "base_angle": 0,
        "rotation_banned": [],
        "angle_sacle": 0,
        "size_ratio": 1.0,
        "dx": 0.0,
        "dy": 0.0,
    }.items():
        if not hasattr(args_copy, k):
            setattr(args_copy, k, v)

    try:
        img, img_with_number, meta = generate_odd_one_out_image(
            grid_size=(args_copy.grid_y, args_copy.grid_x),
            block_size=args_copy.block_size,
            gap=args_copy.gap,
            delta_e=args_copy.de,
            max_image_size=(args_copy.image_size, args_copy.image_size)
            if isinstance(args_copy.image_size, int)
            else tuple(args_copy.image_size),
            margin=args_copy.margin,
            odd_type=args_copy.odd_type_list[idx],
            background_rgb=random_background_color(),
            args=args_copy
        )
        meta["index"] = idx
        save_pair(img, meta, img_dir, meta_dir, idx, img_with_number, draw_bbox=args_copy.draw_bbox)
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


def build_dataset(args):
    img_dir, meta_dir = ensure_dirs(args.data_type)
    num_workers = max(1, args.num_workers)

    print(f"🚀 Starting generation with {num_workers} workers, total {args.number} samples...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_single, idx, args, img_dir, meta_dir)
            for idx in range(args.number) 
        ]

        for future in as_completed(futures):
            idx, success, msg = future.result()
            if success:
                print(f"[OK] Generated sample {idx}")
            else:
                print(f"[Warning] Sample {idx} failed: {msg}")

    print(f"✅ Finished generating {args.number} images into folder: {args.data_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--data_type", type=str, default="train_data", help="Output folder name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel processes")

    args = parser.parse_args()
    args.draw_bbox = True if args.data_type == "test_data" else False
    args.rowcol_image = True if args.data_type == "test_data" else False
    

    odd_type = ["color", "size", "position", "rotation"]
    args.odd_type_list = generate_odd_type_list(odd_type, args.number)

    build_dataset(args)

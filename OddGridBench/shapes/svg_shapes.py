import os, io
import numpy as np
from PIL import Image
import cairosvg
from .registry import register_shape, shape_registry


def rasterize_svg(svg_str, block_size, color=(0,0,0), bgcolor=(1,1,1)):
    """渲染 SVG → numpy 数组，两色模式：前景=color，背景=bgcolor"""
    png_data = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=block_size,
        output_height=block_size
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 4

    alpha = arr[..., 3:4]  # 透明度通道
    fg_color = np.array(color)[None, None, :]
    bg_color = np.array(bgcolor)[None, None, :]

    out = fg_color * alpha + bg_color * (1 - alpha)
    return out


def register_all_svg(folder):
    """递归扫描并注册 folder 下所有子目录中的 SVG 文件"""
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(".svg"):
                continue

            # 用子目录名 + 文件名作为注册名，避免重名
            rel_path = os.path.relpath(root, folder)  # 相对路径
            if rel_path == ".":
                shape_name = os.path.splitext(fname)[0]
            else:
                shape_name = f"{rel_path}(&){os.path.splitext(fname)[0]}"

            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                svg_str = f.read()

            # 注册函数
            def make_func(svg_str, shape_name):
                @register_shape(shape_name)
                def shape_func(block_size, color=(0,0,0), bgcolor=(1,1,1)):
                    return rasterize_svg(svg_str, block_size, color, bgcolor)
                return shape_func

            make_func(svg_str, shape_name)

    print(f"已注册 {len(shape_registry)} 个 SVG 图案: {list(shape_registry.keys())}")
register_all_svg("/data/wengtengjin/colorsense/create_data/svg_file_train/")
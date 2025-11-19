import random
import math
configs = {
    "image_size": (900, 900),
    "grid_x" : [5, 9],
    "grid_y" : [5, 9],
    "de": [5, 20],
    "margin": [30, 40],
    "block_size":[60, 80],
    "gap": [10, 20],
    "size_ratio": [0.85, 1.15],
    "size_ratio_min_offset": 0.05,
    # "odd_type": ["size"],
    "dx": [0.05,0.12],
    "dy": [0.05,0.12],
    "min_offset": 3,
    "base_angle":[25,335],
    "angle_sacle":[-25, 25],
    "angle_min_offset": 10,
    "rotation_banned": ["symbolic(&)circle,natural(&)snowflake"]
}
def randomize_config(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            # 整数范围
            if all(isinstance(x, int) for x in v):
                out[k] = random.randint(v[0], v[1])
            # 浮点范围
            elif all(isinstance(x, (int, float)) for x in v):
                out[k] = round(random.uniform(v[0], v[1]), 2)
            else:
                out[k] = v
        else:
            out[k] = v

    # # --- dx / dy: 至少一个超过 min_offset ---
    # min_offset = cfg.get("min_offset", 0)
    # if "dx" in cfg and "dy" in cfg:
    #     while abs(out["dx"]) <= min_offset and abs(out["dy"]) <= min_offset:
    #         out["dx"] = random.randint(cfg["dx"][0], cfg["dx"][1])
    #         out["dy"] = random.randint(cfg["dy"][0], cfg["dy"][1])
    
    # 设定偏移比例范围，例如 0.05~0.3 表示 5%~30% 的偏移
    
    block_size = out["block_size"]
    # 随机方向（左/右、上/下）
    out["dx"] = math.ceil(block_size * out["dx"]) * random.choice([-1, 1])
    out["dy"] = math.ceil(block_size * out["dy"]) * random.choice([-1, 1])
    
    # --- angle_scale: 强制排除 [-min_offset, min_offset] 区间 ---
    if "angle_sacle" in cfg and "angle_min_offset" in cfg:  # 注意 key 是 angle_sacle
        lo, hi = cfg["angle_sacle"]
        min_angle = cfg["angle_min_offset"]

        if random.random() < 0.5:
            out["angle_sacle"] = int(round(random.uniform(lo, -min_angle), 2))
        else:
            out["angle_sacle"] = int(round(random.uniform(min_angle, hi), 2))

    # --- 随机抽取 odd_type ---
    if "odd_type" in cfg:
        n = random.randint(1, len(cfg["odd_type"]))  # 随机选择 1~len 个
        out["odd_type"] = random.sample(cfg["odd_type"], n)

    # --- size_ratio: 避开中间区域 (例如 0.95~1.05) ---
    if "size_ratio" in cfg and "size_ratio_min_offset" in cfg:
        lo, hi = cfg["size_ratio"]
        min_offset = cfg["size_ratio_min_offset"]

        if random.random() < 0.5:
            out["size_ratio"] = round(random.uniform(lo, 1 - min_offset), 2)
        else:
            out["size_ratio"] = round(random.uniform(1 + min_offset, hi), 2)
    return out



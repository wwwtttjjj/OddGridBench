import json, math, numpy as np, os

INPUT_JSON = "/data/wengtengjin/colorsense/create_data/train_data.json"
RATIOS = [0.5, 0.333333333333334, 0.16]  # easy, medium, hard 比例

configs = {
    "grid_x": [5, 9],
    "grid_y": [5, 9],
    "de": [5, 20],
    "size_ratio": [0.85, 1.15],
    "angle_sacle": [-25, 25],
}

def parse_dxdy(val):
    if isinstance(val, str) and val.startswith("["):
        x, y = map(float, val.strip("[]").split(","))
        return math.sqrt(x**2 + y**2)
    return None

def norm(x, lo, hi): return max(0, min(1, (x - lo) / (hi - lo)))
def rev_norm(x, lo, hi): return 1 - norm(x, lo, hi)

def compute_difficulty(item, max_dxdy=20.0):
    rows, cols = item["grid_size"]
    grid_area = rows * cols
    minA = configs["grid_x"][0] * configs["grid_y"][0]
    maxA = configs["grid_x"][1] * configs["grid_y"][1]
    G = norm(grid_area, minA, maxA)
    n_types = len(item["odd_type"])
    T = 1 - (n_types - 1) / 3

    vals, weights = [], []

    if "color" in item["odd_type"] and item["color_delta_e"] != "N/A":
        val = rev_norm(float(item["color_delta_e"]), *configs["de"])
        vals.append(val); weights.append(0.1)

    if "size" in item["odd_type"] and item["size_ratio"] != "N/A":
        sr = float(item["size_ratio"])
        sr_range = max(abs(configs["size_ratio"][0]-1.0), abs(configs["size_ratio"][1]-1.0))
        val = 1 - min(abs(sr - 1.0)/sr_range, 1.0)
        vals.append(val); weights.append(0.3)

    if "rotation" in item["odd_type"] and item["angle_sacle"] != "N/A":
        ang = float(item["angle_sacle"])
        ang_range = max(abs(configs["angle_sacle"][0]), abs(configs["angle_sacle"][1]))
        val = 1 - min(abs(ang)/ang_range, 1.0)
        vals.append(val); weights.append(0.5)

    if "position" in item["odd_type"] and item["dx_dy"] != "N/A":
        dist = parse_dxdy(item["dx_dy"])
        val = rev_norm(dist, 0.0, max_dxdy)
        vals.append(val); weights.append(0.4)

    if vals:
        M = np.average(vals, weights=weights)
    else:
        M = 0.5

    return round(0.5 * G + 0.2 * T + 0.3 * M, 4)


# ===== main =====
data = json.load(open(INPUT_JSON))
for item in data:
    item["difficulty_score"] = compute_difficulty(item)

# 排序并分割
data_sorted = sorted(data, key=lambda x: x["difficulty_score"])
n = len(data_sorted)
i1 = int(RATIOS[0] * n)
i2 = int((RATIOS[0] + RATIOS[1]) * n)

splits = {
    "easy": data_sorted[:i1],
    "medium": data_sorted[i1:i2],
    "hard": data_sorted[i2:]
}

output_dir = "train_easy_med_hard"
os.makedirs(output_dir, exist_ok=True)

# 写出三个基本文件
for name, subset in splits.items():
    with open(os.path.join(output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

# ===== 新增混合集 =====
easy, med, hard = splits["easy"], splits["medium"], splits["hard"]

# 1️⃣ mix1 = easy 的 1/2 + medium 全部
mix1 = easy[: len(easy)//3 ] + med

# 2️⃣ mix2 = easy 的 1/8 + medium 的 1/4 + hard 全部
mix2 = easy[: len(easy)//3 ] + med[: len(med)//2 ] + hard

# 写出混合文件
with open(os.path.join(output_dir, "medium_easy.json"), "w", encoding="utf-8") as f:
    json.dump(mix1, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, "hard_medium_easy.json"), "w", encoding="utf-8") as f:
    json.dump(mix2, f, ensure_ascii=False, indent=4)

print(f"✅ 输出完成: easy={len(easy)}, medium={len(med)}, hard={len(hard)}, mix1={len(mix1)}, mix2={len(mix2)}")
print(f"📂 输出目录: {os.path.abspath(output_dir)}")

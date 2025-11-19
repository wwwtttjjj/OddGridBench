import os
import json
import pandas as pd
import re
from configs import Ablation_model_list


def parse_row_col(text):
    """
    从字符串中提取 Row / Column 数值，例如：
    "Row 3, Column 5" → (3, 5)
    """
    if not isinstance(text, str):
        return None, None
    match = re.findall(r"Row\s*(\d+).*?Column\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        row, col = match[0]
        return int(row), int(col)
    return None, None


def is_within_tolerance(pred_text, gold_text, tol=1):
    """判断预测坐标是否在容差范围内"""
    pr, pc = parse_row_col(pred_text)
    gr, gc = parse_row_col(gold_text)
    if pr is None or gr is None:
        return False
    return abs(pr - gr) <= tol and abs(pc - gc) <= tol
    # return abs(pr - gr)  + abs(pc - gc) <= tol
    


def compute_relatex_accuracy(data):
    """计算 Row/Column 容差1以内的准确率"""
    correct = 0
    total = 0
    for item in data:
        pred = item.get("extract_answer", "").strip()
        predict_answer = item.get("predict_answer", "").strip()
        gold = item.get("answer", "").strip()

        # 容差判断：extract_answer 或 predict_answer 只要一个在误差范围内就算对
        if is_within_tolerance(pred, gold) and pred != "Row 0, Column 0":
            correct += 1
        total += 1

    acc = round(correct / total * 100, 2) if total > 0 else 0.0
    return acc


def main(json_dir, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []

    for filename in os.listdir(json_dir):
        if filename not in Ablation_model_list or not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        relax_acc = compute_relatex_accuracy(data)
        rows.append({
            "file_name": filename.replace(".json", ""),
            "relax_acc(±1)": relax_acc
        })

    if not rows:
        print(f"No valid JSON files found in {json_dir}.")
        return

    df = pd.DataFrame(rows)
    df["file_name"] = pd.Categorical(df["file_name"], categories=[f.replace(".json", "") for f in Ablation_model_list], ordered=True)
    df = df.sort_values("file_name")

    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"✅ Saved relaxed accuracy results to {csv_path}")


if __name__ == "__main__":
    main("output", "final_result_table/accuracy_relatex.csv")

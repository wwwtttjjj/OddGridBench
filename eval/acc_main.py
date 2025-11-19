import os
import json
import pandas as pd
from collections import defaultdict
from configs import model_list, radar_model_list


def compute_total_accuracy(data):
    correct = 0
    total = 0
    for item in data:
        pred = item.get("extract_answer", "").strip()
        predict_answer = item.get("predict_answer", "").strip()
        gold = item.get("answer", "").strip()
        if pred == gold and pred != "Row 0, Column 0":
            correct += 1
        total += 1
    return round(correct / total * 100, 2) if total > 0 else 0.0


def compute_accuracy_by_odd_type(data):
    """仅统计单一 odd_type 的准确率"""
    counts = defaultdict(int)
    corrects = defaultdict(int)
    for item in data:
        odd_types = item.get("odd_type", [])
        if len(odd_types) != 1:
            continue
        odd = odd_types[0]
        pred = item.get("extract_answer", "").strip()
        predict_answer = item.get("predict_answer", "").strip()
        gold = item.get("answer", "").strip()
        if pred == gold or predict_answer == gold:
            corrects[odd] += 1
        counts[odd] += 1

    acc_by_type = {}
    for odd in ["color", "size", "rotation", "position"]:
        total = counts.get(odd, 0)
        acc = corrects.get(odd, 0) / total * 100 if total > 0 else 0.0
        acc_by_type[odd] = round(acc, 2) if total > 0 else "-"
    return acc_by_type


def compute_accuracy_by_odd_count(data):
    """按 odd_type 数量计算 (odd_count=1,2,3,4)"""
    counts = defaultdict(int)
    corrects = defaultdict(int)
    for item in data:
        odd_count = len(item.get("odd_type", []))
        pred = item.get("extract_answer", "").strip()
        predict_answer = item.get("predict_answer", "").strip()
        gold = item.get("answer", "").strip()
        if pred == gold or predict_answer == gold:
            corrects[odd_count] += 1
        counts[odd_count] += 1

    acc_by_group = {}
    for odd_count, total in counts.items():
        if odd_count == 1:
            continue
        acc = corrects[odd_count] / total * 100 if total > 0 else 0.0
        acc_by_group[odd_count] = round(acc, 2)
    return acc_by_group


def main(json_dir, csv_main):
    rows_main = []
    os.makedirs(os.path.dirname(csv_main), exist_ok=True)

    for filename in os.listdir(json_dir):
        if filename not in model_list or not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_acc = compute_total_accuracy(data)
        acc_by_odd_type = compute_accuracy_by_odd_type(data)
        acc_by_odd_count = compute_accuracy_by_odd_count(data)

        row = {"file_name": filename.replace(".json", "")}
        row.update(acc_by_odd_type)
        for odd_count, acc in sorted(acc_by_odd_count.items()):
            row[f"odd_count_{odd_count}"] = acc
        row["total_acc"] = total_acc
        rows_main.append(row)

    if not rows_main:
        print(f"No JSON files found in {json_dir}.")
        return

    df = pd.DataFrame(rows_main).fillna("-")
    df["file_name"] = pd.Categorical(df["file_name"], categories=[f.replace(".json", "") for f in model_list], ordered=True)
    df = df.sort_values("file_name")

    col_order = ["file_name", "color", "size", "rotation", "position"]
    col_order += sorted([c for c in df.columns if c.startswith("odd_count_")])
    col_order.append("total_acc")
    df = df[col_order]

    df.to_csv(csv_main, index=False, float_format="%.2f")
    print(f"✅ Saved main results to {csv_main}")
    

def radar_main(json_dir, csv_main):
    rows_main = []
    os.makedirs(os.path.dirname(csv_main), exist_ok=True)
    print(radar_model_list)
    for filename in os.listdir(json_dir):
        if filename not in radar_model_list or not filename.endswith(".json"):
            print(filename)
            continue

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_acc = compute_total_accuracy(data)
        acc_by_odd_type = compute_accuracy_by_odd_type(data)
        acc_by_odd_count = compute_accuracy_by_odd_count(data)

        row = {"file_name": filename.replace(".json", "")}
        row.update(acc_by_odd_type)
        for odd_count, acc in sorted(acc_by_odd_count.items()):
            row[f"odd_count_{odd_count}"] = acc
        row["total_acc"] = total_acc
        rows_main.append(row)

    if not rows_main:
        print(f"No JSON files found in {json_dir}.")
        return

    df = pd.DataFrame(rows_main).fillna("-")
    df["file_name"] = pd.Categorical(df["file_name"], categories=[f.replace(".json", "") for f in radar_model_list], ordered=True)
    df = df.sort_values("file_name")

    col_order = ["file_name", "color", "size", "rotation", "position"]
    col_order += sorted([c for c in df.columns if c.startswith("odd_count_")])
    col_order.append("total_acc")
    df = df[col_order]

    df.to_csv(csv_main, index=False, float_format="%.2f")
    print(f"✅ Saved main results to {csv_main}")


if __name__ == "__main__":
    main("output", "final_result_table/accuracy_main.csv")
    radar_main("output", "final_result_table/radar_main.csv")
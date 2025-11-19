import os
import json
import uuid
import argparse   # ✅ 新增

def main(arsg):
    # 目录路径

    # 结果列表
    merged_data = []

    # 遍历 metadata 目录下所有 .json 文件
    for filename in os.listdir(args.metadata_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(args.metadata_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 生成随机 id
        new_id = str(uuid.uuid4())

        # image 路径（如果要改相对路径可自行调整）
        image_path = data.get("image_file", "")

        # answer 转成文本（例如 'row 2, col 5'）
        answer_text = f"Row {data['odd_position']['row']}, Column {data['odd_position']['col']}"

        # 构造目标格式
        merged_data.append({
            "id": new_id,
            "image": image_path.split("/")[-1],
            "answer": answer_text,
            "answer_position": data.get("odd_position", None),
            "shape": data.get("base_shape", None),
            "grid_size": data["grid_size"],
            "odd_type": data["odd_type"],
            "angle_sacle": data["angle_sacle"],
            "size_ratio": data["size_ratio"],
            "color_delta_e": data.get("color_delta_e", None),
            "dx_dy": data.get("dx_dy", None),
            "odd_bbox": data.get("odd_bbox", []),
            "image_size": data.get("image_size", None),
        })

    # 保存为一个合并的 json
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 合并完成，共 {len(merged_data)} 条记录，保存到：{args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metadata JSON files.")
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",   # ✅ 默认值
        help="datatype of data."
    )
    args = parser.parse_args()
    args.output_file = f"{args.data_type}_data.json"
    args.metadata_dir = f"./{args.data_type}_data/metadata"
    
    main(args)

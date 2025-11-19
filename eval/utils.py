from pathlib import Path
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal, sys



def build_prompt(data: dict) -> str:
    rows, cols = data.get("grid_size", [0, 0])
    shape = data.get("shape", "object")
    odd_types = data.get("odd_type", [])
    odd_desc = ", ".join(odd_types) if odd_types else "appearance"

    prompt = f"""
    You are solving an odd-one-out visual perception task.
    You are given an image showing a {rows}×{cols} grid of {shape}s.
    All {shape}s appear the same, except one that is visually different in {odd_desc}.

    This is a **visual perception** task that does not require lengthy logical reasoning.

    ### Instructions
    - Carefully inspect the grid.
    - Identify the grid position (row and column) of the {shape} that is different.
    - Counting starts from the top-left corner, which is Row 1, Column 1.
    - Provide brief visual observations if necessary (no more than 300 words).

    ### Output Format Requirements
    - Provide observations in concise natural language, within 300 words.
    - End the response with the final answer strictly in LaTeX format:
      ```
      \\boxed{{Row X, Column Y}}
      ```
      where X and Y are integers (e.g., Row 2, Column 3).
    - Do **not** include any additional explanations or text after the \\boxed{{}} line.
    - If you believe no odd {shape} exists, return \\boxed{{Row 0, Column 0}}.
    """
    return prompt

def build_prompt_rl(data: dict) -> str:
    rows, cols = data.get("grid_size", [0, 0])
    shape = data.get("shape", "object")

    # 如果 shape 是纯数字且在 0~100000 之间，都归为 "object"
    if isinstance(shape, str) and shape.isdigit():
        num = int(shape)
        if 0 <= num <= 100000:
            shape = "object"
    shape = "object"
    odd_types = data.get("odd_type", [])
    odd_desc = ", ".join(odd_types) if odd_types else "appearance"

    prompt = (
        f"Identify the {shape} that differs from others in the {rows}×{cols} grid. "
        f"The difference is in {odd_desc}. "
        f"Count from the top-left as Row 1, Column 1. "
        f"Give your answer in the form: \\boxed{{Row X, Column Y}}."
    )
    return prompt


def build_prompt_director(data: dict) -> str:
    rows, cols = data.get("grid_size", [0, 0])
    shape = data.get("shape", "object")
        # 如果 shape 是纯数字且在 0~100000 之间，都归为 "object"
    if isinstance(shape, str) and shape.isdigit():
        num = int(shape)
        if 0 <= num <= 100000:
            shape = "object"
    odd_types = data.get("odd_type", [])
    
    odd_desc = ", ".join(odd_types) if odd_types else "appearance"

    prompt = f"""
    You are solving an **Odd-One-Out Visual Perception** task.

    You are given an image showing a {rows}×{cols} grid of {shape}s.
    All {shape}s look identical except one that differs in its {odd_desc}.

    This is a **pure visual perception** task — no reasoning or calculation is required.

    ### Task
    - Carefully inspect the grid.
    - Identify the {shape} that looks different.
    - Report its grid position (Row and Column), counting from the top-left corner as Row 1, Column 1.

    ### Output Format
    Only output the result in **exactly** this format:

    \\boxed{{Row X, Column Y}}

    Replace X and Y with the correct row and column numbers.
    For example:

    \\boxed{{Row 2, Column 3}}
    """
    return prompt


def build_prompt_red_box(data: dict) -> str:
    rows, cols = data.get("grid_size", [0, 0])
    shape = data.get("shape", "object")
    odd_types = data.get("odd_type", [])
    odd_desc = ", ".join(odd_types) if odd_types else "appearance"

    prompt = f"""
    You are given an image showing a {rows}×{cols} grid of {shape}s.
    Each grid cell contains one {shape}.
    - Counting starts from the top-left corner, which is Row 1, Column 1.
    - You only need to find the red box and return the answer.
    - End the response with the final answer strictly in LaTeX format:
      ```
      \\boxed{{Row X, Column Y}}
      ```
      where X and Y are integers (e.g.,  \\boxed{{Row 2, Column 3}}).
    - Do **not** include any additional explanations or text after the \\boxed{{}} line.
    """
    return prompt

def Extract_answer(predict_answer: str) -> str:
    """
    Extract the final boxed answer of the form \boxed{Row X, Column Y}.
    X and Y must be integers.
    """
    # 匹配 \boxed{Row X, Column Y}，允许空格，大小写忽略
    matches = re.findall(
        r"\\boxed\{\s*Row\s+(\d+)\s*,\s*Column\s+(\d+)\s*\}", 
        predict_answer, 
        flags=re.IGNORECASE
    )

    if len(matches) == 1:
        row, col = matches[0]
        return f"Row {row}, Column {col}"
    elif len(matches) > 1:
        return "Many answers found"
    else:
        return "No answer found"


def remove_existing_file(save_path):
    """
    Remove the file at the specified path if it exists.

    Args:
        save_path (str): Path of the file to remove.
    """
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Existing file removed: {save_path}")

def write_json(save_json_path, save_json_data):
    if os.path.exists(save_json_path):
        with open(save_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(save_json_data)
        else:
            data = [data, save_json_data]
    else:
        data = [save_json_data]

    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run_model_parallel(root_dir, save_json_path, max_workers, init_worker, worker_inference, args):
    """
    Run model inference in parallel and save results incrementally.

    Args:
        root_dir (str): Root directory containing images.
        save_json_path (str): Path to save aggregated JSON results.
        max_workers (int): Number of parallel worker processes.
        init_worker (callable): Initialization function for each worker (loads model).
        worker_inference (callable): Function to run inference on a single task.
        model_name (str): Model name passed to init_worker for loading.
    """
    # Prepare all tasks and corresponding raw data for result writing
    tasks = []
    raw_datas = []
            
    
    # 读取已有结果文件，避免重复处理
    processed_image_ids = set()
    if os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
        with open(save_json_path, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                for item in existing_results:
                    image_id = item.get("image_id")
                    if image_id:
                        processed_image_ids.add(image_id)
            except json.JSONDecodeError:
                print(f"[WARN] JSON 文件损坏或为空: {save_json_path}，已忽略。")
                existing_results = []
    else:
        existing_results = []

    for json_file in [args.json_path]:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if args.data_type == "with_number":
            json_data = json_data[:500]
        if args.red_box:
            json_data = json_data[:200]

        for data in json_data:
            image_id = data.get("id")
            if image_id in processed_image_ids:
                continue  # 已处理，跳过

            prompt = build_prompt(data) if not args.red_box else build_prompt_red_box(data)
            data["prompt"] = prompt
            image_name = data.get("image")
            image_path = os.path.join(root_dir, image_name)

            tasks.append((prompt, image_path))
            raw_datas.append(data)
    if not raw_datas:
        print("No new data to process. Exiting.")
        return
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(args.model_name,)
        ) as executor:

            futures = {executor.submit(worker_inference, task): idx for idx, task in enumerate(tasks)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    predict_answer = future.result()
                except Exception as e:
                    print(f"[ERROR] Task {idx} failed: {e}")
                    continue

                data = raw_datas[idx]
                extract_answer = Extract_answer(predict_answer)

                save_json_data = {
                    "image_id": data.get('id'),
                    "image_name": data.get("image"),
                    "prompt": data.get('prompt'),
                    "predict_answer": predict_answer,
                    "extract_answer": extract_answer,
                    "answer": data.get('answer'),
                    "grid_size": data["grid_size"],
                    "odd_type": data["odd_type"],
                    "angle_sacle": data["angle_sacle"],
                    "size_ratio": data["size_ratio"],
                    "color_delta_e": data.get("color_delta_e", None),
                    "dx_dy": data.get("dx_dy", None),
                }
                write_json(save_json_path, save_json_data)
                print(f"[INFO] Written result for {data.get('id')}")

    except KeyboardInterrupt:
        print("⚠️ 收到 Ctrl+C，准备终止所有进程...")
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        sys.exit(1)

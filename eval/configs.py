import os
models_dir = "/data/wengtengjin/models/"
max_new_tokens = 1024

def get_configs(data_type: str):
    # 目录定义

    image_dir = "/data/wengtengjin/colorsense/create_data/test_data/image"
    json_path = "/data/wengtengjin/colorsense/create_data/test_data.json"
    root_dir = image_dir

    # 输出路径
    Result_root = "output/"

    if data_type == "with_number":
        root_dir = root_dir.replace("image", "image_number")
        Result_root = "output_number/"
    if not os.path.exists(Result_root):
        os.mkdir(Result_root)
    return {
        "image_type": "colorsense",
        "image_dir": image_dir,
        "json_path": json_path,
        "root_dir": root_dir,
        "Result_root": Result_root,
        "models_dir": models_dir,
    }
    
    
Ablation_model_list = [
    "InternVL3_5-38B-Instruct.json",
    # "Qwen3-VL-30B-A3B-Instruct.json",
    # "Qwen3-VL-2B-Instruct.json",
    
    # "Qwen3-VL-4B-Instruct.json",
    "Qwen3-VL-32B-Instruct.json",
    "gemini-2.5-pro.json",
    "gpt-5.json",
]

radar_model_list = [
    "random.json",
    "Molmo-72B-0924.json",
    "InternVL3_5-38B-Instruct.json",
    # "Qwen3-VL-30B-A3B-Instruct.json",
    "Qwen3-VL-4B-Instruct.json",
    "Qwen3-VL-32B-Instruct.json",
    "gemini-2.5-pro.json",
    "gpt-5.json",
    "human.json"
]
    
model_list = [
    # 🟢 Random baseline
    "random.json",
    # 🟢 Open-source MLLMs
    "Phi-3.5-vision-instruct.json",
    "SAIL-VL2-2B.json",
    "SAIL-VL2-8B.json",
    "LLaVA-OneVision-1.5-4B-Instruct.json",
    "LLaVA-One-Vision-1.5-8B-Instruct.json",
    "llava-v1.6-34b.json",
    "InternVL3_5-38B-Instruct.json",
    "Molmo-72B-0924.json",
    "Qwen2.5-VL-7B-Instruct.json",
    "Qwen2.5-VL-72B-Instruct.json",
    "Qwen3-VL-2B-Instruct.json",
    "Qwen3-VL-4B-Instruct.json",
    "Qwen3-VL-8B-Instruct.json",
    "Qwen3-VL-30B-A3B-Instruct.json",
    "Qwen3-VL-32B-Instruct.json",
    # 🟢 Proprietary MLLMs
    "gemini-2.0-flash.json",
    "gemini-2.5-flash.json",
    # gemini-2.5-Pro 不存在，放最后
    "gemini-2.5-pro.json",
    # "GPT‑4-turbo.json",
    "gpt-5.json",
    "human.json",
    "GRPO.json",
    "OddRL.json",
    
    "SFT.json"
]
model_list_rl = [
    "Qwen3-VL-2B-Instruct.json",
    "GRPO.json",
    "OddRL_StepbyStep.json",
    "Grpo_stepbystep.json",
    "OddRL.json",
    "GSPO.json",
    "DAPO.json",
    "CISPO.json"
]
model_name_map = {
    # 🟢 Random baseline
    "random": "Random",

    # 🟢 Open-source MLLMs
    "Phi-3.5-vision-instruct": "Phi-3.5-vision",
    "SAIL-VL2-2B": "SAIL-VL2-2B",
    "SAIL-VL2-8B": "SAIL-VL2-8B",
    "LLaVA-OneVision-1.5-4B-Instruct": "LLaVA-OneVision-1.5-4B",
    "LLaVA-One-Vision-1.5-8B-Instruct": "LLaVA-OneVision-1.5-8B",
    "llava-v1.6-34b": "LLaVA-v1.6-34B",
    "InternVL3_5-38B-Instruct": "InternVL3.5-38B",
    "Molmo-72B-0924": "Molmo-72B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B",
    "Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
    # 🟡 Proprietary MLLMs
    "gemini-2.0-flash": "Gemini-2.0-flash",
    "gemini-2.5-flash": "Gemini-2.5-flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gpt-5": "GPT-5",

    # 🟠 Human
    "human": "Human"
}


# model_list = [
#     "Random.json",
#     "phi3_5.json",
#     "llava-v1.5-7b.json",
#     "llava-v1.5-13b.json",
#     "llava-v1.6-34b.json",
#     "llava-onevision-qwen2-7b-ov-hf.json",
#     "llava-onevision-qwen2-72b-si-hf.json",
#     "Internvl2_5-8B.json",
#     "Internvl2_5-38B.json",
#     "Internvl2_5-78B.json",
#     "Janus-Pro-7B.json",
#     "Qwen2-VL-2B-Instruct.json",
#     "Qwen2-VL-7B-Instruct.json",
#     "Qwen2-VL-72B-Instruct.json",
#     "Qwen2.5-VL-3B-Instruct.json",
#     "Qwen2.5-VL-7B-Instruct.json",
#     "Qwen2.5-VL-72B-Instruct.json",
#     "GPT-4o.json",
#     "gemini-1.5-flash.json",
#     "gemini-2.0-flash.json",
#     "gemini-1.5-pro.json",
#     "Human.json",
#     "Internvl-8B.json",
#     "Internvl-40B.json",
#     "InternVL2-8B-MPO.json",
#     "llava-v1.5-13b.json",
#     "Math-llava-13b.json",
#     "Llama-VL-3_2-11B.json",
#     "Llama-3.2V-11B-cot.json",
#     "Qwen2.5-VL-7B-Instruct.json",
#     "R1-Onevision-7B.json"
# ]
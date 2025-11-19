import os
import argparse
from qwen_vl_utils import process_vision_info
from configs import get_configs, max_new_tokens, models_dir
from utils import run_model_parallel
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image

# Global model and processor reused across processes
model = None
processor = None
tokenizer = None

import os, torch

def init_worker(model_name="SAIL-VL2-8B"):
    global model, processor, tokenizer
    model_path = os.path.join(models_dir, model_name)

    # 根据进程号决定用哪个 GPU
    world_size = torch.cuda.device_count()
    worker_id = os.getpid() % world_size
    device = torch.device(f"cuda:{worker_id}")

    print(f"[PID {os.getpid()}] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)


def worker_inference(task):
    """
    Perform inference on a single (prompt, image_path) pair using the Qwen2.5-VL model.
    """
    global model, processor,tokenizer
    prompt, image_path = task
    
    messages = [
        {"role": "user", "content": [{"type": "image", "image": 'image_path'}, 
        {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    image = Image.open(image_path)
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device).to(torch.bfloat16)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()
    return response


def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using SAIL-VL2-8B models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="SAIL-VL2-2B",
        help="Model name. Example: SAIL-VL2-2B, SAIL-VL2-8B"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers for inference"
    )
    
    parser.add_argument(
        "--red_box",
        action="store_true",
        help="Whether to use red box for inference (default: False)"
    )
    
    parser.add_argument(
        "--data_type",
        type=str,
        default="normal",
        help="normal or with_number"
    )

    args = parser.parse_args()
    configs_para = get_configs(args.data_type)
    Result_root = configs_para["Result_root"]
    args.json_path = configs_para["json_path"]
    if args.red_box:
        Result_root = Result_root.replace("output", "box_output")
        configs_para["root_dir"] = configs_para["root_dir"].replace("image", "image_red")
        
    save_json_path = os.path.join(Result_root, f"{args.model_name}.json")

    run_model_parallel(
        root_dir=configs_para["root_dir"],
        save_json_path=save_json_path,
        max_workers=args.max_workers,
        init_worker=init_worker,
        worker_inference=worker_inference,
        args=args
    )

if __name__ == "__main__":
    main()

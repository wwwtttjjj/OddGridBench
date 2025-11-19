import os
import argparse
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from configs import get_configs, max_new_tokens
from utils import run_model_parallel

# Global model and processor reused across processes
model = None
processor = None

def init_worker(model_name="Phi-3.5-vision-instruct"):
    """
    Initialize model and processor for a given model name.
    This is run once per worker process.
    """
    global model, processor

    model_path = "microsoft/" + model_name


    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    _attn_implementation='eager'    
    )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_path, 
    trust_remote_code=True, 
    num_crops=4
    ) 
    print(f"[INFO] Model loaded: {model_name}")


def worker_inference(task):
    """
    Perform inference on a single (prompt, image_path) pair using the Qwen2.5-VL model.
    """
    global model, processor
    prompt, image_path = task
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    images.append(Image.open(image_path))
    placeholder += f"<|image_{1}|>\n"

    messages = [
        {"role": "user", "content": placeholder + prompt},
    ]

    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": max_new_tokens, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0]
    
    return response


def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using Phi-3.5-vision-instruct models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Phi-3.5-vision-instruct",
        help="Model name. Example: Phi-3.5-vision-instruct"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=6,
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

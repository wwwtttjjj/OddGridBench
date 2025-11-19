import os
import argparse
from configs import get_configs, models_dir, max_new_tokens
from utils import run_model_parallel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch

# Global model and processor reused across processes
model = None
processor = None
def init_worker(model_name="Molmo-72B-0924"):
    # load the processor
    global model, processor
    model_path = f"allenai/{model_name}"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    print(f"[INFO] Model loaded: {model_name}")


def worker_inference(task):
    global model, processor
    prompt, image_path = task
    
    # process the image and text
    image = Image.open(image_path).convert("RGB")

    inputs = processor.process(
        images=[image],   # 列表或单张都行，看 processor 文档
        text=prompt
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 2048 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Molmo-72B-0924",
        help="Model name. Molmo-72B-0924"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
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
        


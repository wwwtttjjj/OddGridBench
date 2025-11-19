from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
import argparse
from configs import get_configs, models_dir, max_new_tokens
from utils import run_model_parallel

# Global model and processor reused across processes
model = None
processor = None
tokenizer =None
context_len = None
input_model_name = None
def init_worker(model_name="llava-v1.6-34b"):
    """
    Initialize model and processor for a given model name.
    This is run once per worker process.
    """
    global model, processor, tokenizer, context_len, input_model_name

    model_path = os.path.join(models_dir, model_name)
    input_model_name=get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=input_model_name
    )


def worker_inference(task):
    global model, processor, tokenizer, context_len, input_model_name
    prompt, image_path = task
    args = type('Args', (), {
        "model_name":input_model_name,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": processor,
        "context_len":context_len,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens
    })()

    return eval_model(args)
def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using Qwen2.5-VL models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-v1.6-34b",
        help="Model name. llava-v1.6-34b"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
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
        


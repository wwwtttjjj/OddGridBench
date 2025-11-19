import os
import argparse
from configs import get_configs, models_dir, max_new_tokens
from utils import run_model_parallel
from modelscope import Qwen3VLMoeForConditionalGeneration, AutoProcessor

from transformers import Qwen3VLForConditionalGeneration
# Global model and processor reused across processes
model = None
processor = None


def init_worker(model_name="Qwen3-VL-30B-A3B-Instruct"):
    """
    Initialize model and processor for a given model name.
    This is run once per worker process.
    """
    global model, processor
    if model_name in ["Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"]:
        model_path = "Qwen/" + model_name
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
        processor = AutoProcessor.from_pretrained(model_path)

    else:
        model_path = os.path.join(models_dir, model_name)
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )

        processor = AutoProcessor.from_pretrained(model_path)

    print(f"[INFO] Model loaded: {model_name}")


def worker_inference(task):
    """
    Perform inference on a single (prompt, image_path) pair using the Qwen2.5-VL model.
    """
    global model, processor
    prompt, image_path = task
    print(f'[INFO] Processing: {os.path.basename(image_path)}', end='\r')
    # Format the prompt with chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text[0]


def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using Qwen2.5-VL models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-VL-4B-Instruct",
        help="Model name. Example: Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-4B-Instruct,Qwen3-VL-8B-Instruct"
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

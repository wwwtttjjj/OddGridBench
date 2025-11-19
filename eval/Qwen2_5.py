import os
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from configs import get_configs, models_dir, max_new_tokens
from utils import run_model_parallel

# Global model and processor reused across processes
model = None
processor = None


def init_worker(model_name="Qwen2.5-VL-72B-Instruct"):
    """
    Initialize model and processor for a given model name.
    This is run once per worker process.
    """
    global model, processor

    if model_name == "R1-Onevision-7B":
        model_path = "Fancy-MLLM/R1-Onevision-7B"
    if model_name in ["different-Qwen2_5_3B-model","different-Qwen2_5_7B-model"]:
        model_path = os.path.join(models_dir, model_name)
    else:
        model_path = f"Qwen/{model_name}"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
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

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Preprocess inputs (image/video + prompt)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Remove prompt tokens from the output
    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode and return text
    response_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(response_text[0])

    return response_text[0]


def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using Qwen2.5-VL models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="different-Qwen2_5_3B-model",
        help="Model name. Example: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, R1-Onevision-7B"
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
        print(configs_para["root_dir"], Result_root)
        
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

import os
import argparse
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from configs import get_configs, models_dir, max_new_tokens
from utils import run_model_parallel

# Constants for ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Global variables for model/tokenizer reuse
model = None
tokenizer = None


def build_transform(input_size):
    """
    Returns a composed transform for preprocessing the image:
    - Convert to RGB if needed
    - Resize to fixed input size
    - Normalize using ImageNet stats
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, candidate_ratios, width, height, image_size):
    """
    Find the grid layout (rows x cols) with the closest aspect ratio to the image.
    Prefer layouts with higher effective area coverage if tied.
    """
    best_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in candidate_ratios:
        target_ratio = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ratio)

        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]):
            best_diff = diff
            best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_tiles=1, max_tiles=12, image_size=448, use_thumbnail=False):
    """
    Dynamically splits image into multiple tiles based on its aspect ratio.
    Optionally adds a thumbnail at the end.
    Returns a list of PIL Images.
    """
    width, height = image.size
    aspect_ratio = width / height

    candidate_ratios = {
        (i, j)
        for n in range(min_tiles, max_tiles + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_tiles <= i * j <= max_tiles
    }
    candidate_ratios = sorted(candidate_ratios, key=lambda x: x[0] * x[1])

    target_ratio = find_closest_aspect_ratio(aspect_ratio, candidate_ratios, width, height, image_size)

    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    num_blocks = target_ratio[0] * target_ratio[1]

    resized_image = image.resize((target_width, target_height))
    tiles_per_row = target_ratio[0]
    tiles = []

    for i in range(num_blocks):
        left = (i % tiles_per_row) * image_size
        top = (i // tiles_per_row) * image_size
        box = (left, top, left + image_size, top + image_size)
        tiles.append(resized_image.crop(box))

    if use_thumbnail and num_blocks > 1:
        thumbnail = image.resize((image_size, image_size))
        tiles.append(thumbnail)

    return tiles


def load_image(image_path, input_size=448, max_tiles=12):
    """
    Load an image file and return a tensor of tiled and normalized images.
    Shape: (num_tiles, 3, input_size, input_size)
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_tiles=max_tiles, use_thumbnail=True)
    tensors = [transform(tile) for tile in tiles]
    return torch.stack(tensors)


def init_worker(model_name="Internvl2_5-78B"):
    """
    Initialize the model and tokenizer once per worker process.
    """
    global model, tokenizer

    model_path = os.path.join(models_dir, model_name)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print(f"[INFO] Model loaded: {model_name}")


def worker_inference(task, timeout=300):   # timeout单位秒，这里默认5分钟
    """
    Perform inference on a single (prompt, image_path) pair using the loaded model.
    超时timeout秒未返回则直接返回OUT OF TIME
    """
    global model, tokenizer
    prompt, image_path = task

    image_tensor = load_image(image_path).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    def _call():
        return model.chat(tokenizer, image_tensor, prompt, generation_config)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"[WARN] inference超时>{timeout}s，返回None")
            return "OUT OF TIME"
        except Exception as e:
            print(f"[ERROR] inference失败: {e}")
            return "OUT OF TIME"


def main():
    global Result_root  # 告诉 Python 使用全局变量
    """
    Parse arguments and run the model inference in parallel.
    """
    parser = argparse.ArgumentParser(description="Run InternVL2 model inference")

    parser.add_argument(
        "--model_name",
        type=str,
        default="InternVL3_5-38B-Instruct",
        help="Choose model from: Internvl-2B, Internvl-8B, Internvl2_5-8B, Internvl2_5-38B, Internvl2_5-78B"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers"
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

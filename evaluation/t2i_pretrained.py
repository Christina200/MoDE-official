"""Text-to-image evaluation: generate images from prompts and score CLIP image alignment."""
import argparse
import json
import os

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import ChameleonProcessor, MMoEChameleonForConditionalGeneration

from peft import MMoELoraConfig, get_peft_model

from evaluation.eval_utils import (
    add_results_to_json,
    calculate_accuracy_and_stderr,
    get_clip_model,
    list_images,
    log_samples,
    set_seeds,
)

ADAPTER_NAME = "MMOELORA"


def clip_image_eval(model, processor, save_dir, save_name, text_input_name, target_path,
                    dataname="concept101"):
    similarity_calculator = get_clip_model()

    image_samples = []

    with open(text_input_name, "r") as f:
        data = json.load(f)

    curr_dir = f"{save_dir}/inference"
    os.makedirs(curr_dir, exist_ok=True)

    for idx, caption in enumerate(data["prompt"]):
        inputs = processor(caption, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)
        generate_ids = model.generate_mm(
            **inputs,
            multimodal_generation_mode="image-only",
            # 1026 = image_start_token + 1024 image tokens + image_end_token
            max_new_tokens=1026,
            # Most image tokens seen in training are "empty" patches, so greedy decoding
            # tends to produce a blank image.
            do_sample=True,
        )

        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        images = processor.postprocess_pixel_values(pixel_values)
        images = [to_pil_image(img.detach().cpu()) for img in images]
        image_path = os.path.join(curr_dir, f"{idx}.png")
        images[0].save(image_path)

        image_samples.append({"caption": caption, "image": image_path})

    # CLIP image alignment: every generated image against the real target images.
    generated_paths = [sample["image"] for sample in image_samples]
    generation_similarity_scores = similarity_calculator.calculate_image_alignment(
        generated_paths, list_images(target_path)
    )
    for sample, score in zip(image_samples, generation_similarity_scores):
        sample["score"] = score

    avg_generation_similarity, generation_stderr = calculate_accuracy_and_stderr(generation_similarity_scores)

    results_dict = {
        "image_generation": {
            dataname: {
                "generation_similarity": avg_generation_similarity,
                "generation_stderr": generation_stderr,
            }
        }
    }

    log_samples(f"{save_dir}/logs/{save_name}", "image_generation", image_samples)
    add_results_to_json(f"{save_dir}/results/{save_name}.json", results_dict)


def load_model(args):
    model = MMoEChameleonForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.mode = "inference-image"
    peft_config = MMoELoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_nums=args.lora_num,
        target_modules=["up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.load_adapter(args.ckpt, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    return model.to(torch.bfloat16)


def main(args):
    set_seeds(args.seed)
    processor = ChameleonProcessor.from_pretrained(args.base_model)
    model = load_model(args)
    clip_image_eval(model, processor, args.save_dir, args.save_name, args.prompt_file, args.target_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the trained MoDE adapter")
    parser.add_argument("--base_model", default="leloy/Anole-7b-v0.1-hf", type=str,
                        help="Base model the MoDE experts are attached to")
    parser.add_argument("--save_dir", default="evaluation/outputs", type=str,
                        help="Directory for generated images, logs and results")
    parser.add_argument("--save_name", default="eval_results", type=str, help="Name of the saved file")
    parser.add_argument("--prompt_file", default="data/concept101/text_input.json", type=str,
                        help="JSON file with a 'prompt' list of text prompts")
    parser.add_argument("--target_path", default="data/concept101/target_images", type=str,
                        help="Directory of real target images to measure CLIP image alignment against")
    parser.add_argument('--lora_num', type=int, default=4, help="number of LoRA experts")
    parser.add_argument('--lora_r', type=int, default=8, help="lora r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="lora dropout")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())

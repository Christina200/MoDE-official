"""VQA evaluation: greedy-decode an answer per question and score it against the gold answer."""
import argparse
import json
import os
import re
import string

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ChameleonProcessor, MMoEChameleonForConditionalGeneration

from peft import MMoELoraConfig, get_peft_model

from evaluation.eval_utils import (
    add_results_to_json,
    calculate_accuracy_and_stderr,
    log_samples,
    set_seeds,
)

ADAPTER_NAME = "MMOELORA"


# --- Scoring -----------------------------------------------------------------
# Most tasks are scored by exact string match. ScienceQA is multiple choice, so a
# response is accepted either as the option letter (in any of the usual surface
# forms) or as the text of the gold option -- "mushroom" and "B" are both correct
# when option B is "mushroom". Scoring it by exact match instead understates
# ScienceQA accuracy by roughly 28 points.

def _normalize(text):
    """Lowercase, strip surrounding punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = text.strip(string.punctuation + " ")
    return re.sub(r"\s+", " ", text)


# a leading option token: "b", "B.", "(B)", "[b] ", "B) ", "B : "
_LEADING_LETTER = re.compile(r'^\s*[\(\[\{"]?\s*([a-d])\b\s*[\)\]\}":.\-]*')
# an explicit declaration anywhere: "answer: B", "option b", "ans = c"
_DECLARED_LETTER = re.compile(r'(?:answer|ans|option|choice)\s*[:=]?\s*([a-d])\b')


def scienceqa_correct(question, answer, generated_text):
    """True if the response names the gold option, by letter or by its text."""
    options = re.findall(r'([A-D])\.\s*([^\n]+)', question)
    option_text = {letter.strip().lower(): _normalize(text) for letter, text in options}

    generated = _normalize(generated_text)
    gold_letter = _normalize(answer)
    gold_text = option_text.get(gold_letter, "")

    if not generated:
        return False

    # a bare option letter
    if generated in {"a", "b", "c", "d"} and generated == gold_letter:
        return True

    # a leading option letter, e.g. "B. Denver"
    match = _LEADING_LETTER.match(generated)
    if match and match.group(1) == gold_letter:
        return True

    # an explicitly declared letter, e.g. "answer: b"
    match = _DECLARED_LETTER.search(generated)
    if match and match.group(1) == gold_letter:
        return True

    # the text of the gold option, as a whole phrase so "bee" does not match "b"
    if gold_text:
        if generated == gold_text or generated.startswith(gold_text + " "):
            return True
        if re.search(r"\b" + re.escape(gold_text) + r"\b", generated):
            return True

    return False


def score_response(task_id, question, answer, generated_text):
    task = task_id.lower()
    if "scienceqa" in task:
        return scienceqa_correct(question, answer, generated_text)

    generated, gold = generated_text.strip().lower(), answer.strip().lower()
    if generated == gold:
        return True
    # ImageNet answers are class names, accepted anywhere in the response
    return "imagenet" in task and gold in generated


class CustomDataset(Dataset):
    def __init__(self, instructions, eval_data_dir):
        self.instructions = [inst for inst in instructions if "image" in inst]
        self.eval_data_dir = eval_data_dir

        # The TextVQA and VizWiz test splits are held out for leaderboard submission and
        # ship without answers, so accuracy cannot be computed against them. Use val.json
        # for those two tasks; ScienceQA, ImageNet and GQA are labelled in test.json.
        if self.instructions and "answer" not in self.instructions[0]:
            raise ValueError(
                "This instruction file has no 'answer' field, so accuracy cannot be "
                "computed. It is most likely an unlabelled test split (TextVQA and "
                "VizWiz); use the val.json split for those tasks instead."
            )

    def __getitem__(self, index):
        line = self.instructions[index]
        image_path = os.path.join(self.eval_data_dir, line["image"])
        return image_path, line["text"], line["answer"]

    def __len__(self):
        return len(self.instructions)


def eval_model(model, instruction_file, processor, save_dir, save_name, eval_data_dir, task_id,
               max_new_tokens=60):
    with open(os.path.expanduser(instruction_file), "r") as f:
        instructions = json.load(f)

    dataset = CustomDataset(instructions, eval_data_dir)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    is_imagenet = 'imagenet' in task_id.lower()
    logs = []
    scores = []

    for image_path, question, answer in tqdm(data_loader, total=len(dataset)):
        image_path, question, answer = image_path[0], question[0], answer[0]
        image = Image.open(image_path)

        text = question + '<image>' if is_imagenet else question
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device, torch.bfloat16)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        generated_text = processor.decode(new_tokens, skip_special_tokens=True)

        is_correct = score_response(task_id, question, answer, generated_text)
        scores.append(is_correct)
        logs.append({
            "question": question,
            "image_path": image_path,
            "answer": answer,
            "generated_text": generated_text,
            "is_correct": is_correct,
        })

    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)

    log_samples(os.path.join(save_dir, "logs", f"{save_name}.json"), task_id, logs)

    accuracy, std_err = calculate_accuracy_and_stderr(scores)
    metrics = {f"{task_id}": {"accuracy": accuracy, "std_err": std_err}}
    add_results_to_json(os.path.join(save_dir, "results", f"{save_name}.json"), metrics)
    print(f"{task_id}: accuracy={accuracy:.4f} std_err={std_err:.4f} (n={len(scores)})")


def load_model(args):
    model = MMoEChameleonForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
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
    model = load_model(args)
    model.eval()

    processor = ChameleonProcessor.from_pretrained(args.base_model)

    set_seeds(args.seed)

    eval_model(model, args.instruction_file, processor, args.save_dir, args.save_name,
               args.eval_data_dir, args.task_id)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the trained MoDE adapter")
    parser.add_argument("--base_model", default="leloy/Anole-7b-v0.1-hf", type=str,
                        help="Base model the MoDE experts are attached to")
    parser.add_argument("--instruction_file", required=True, type=str,
                        help="Instruction JSON for the task being evaluated")
    parser.add_argument("--eval_data_dir", required=True, type=str,
                        help="Root directory the 'image' paths in the instruction file are relative to")
    parser.add_argument("--save_dir", default="evaluation/outputs", type=str,
                        help="Directory for eval logs and results")
    parser.add_argument("--save_name", default="eval_results", type=str, help="Name of the saved file")
    parser.add_argument("--task_id", required=True, type=str, help="Name of the task being evaluated")
    parser.add_argument('--lora_num', type=int, default=4, help="number of LoRA experts")
    parser.add_argument('--lora_r', type=int, default=8, help="lora r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="lora dropout")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())

import json
import os
import random

import numpy as np
import torch
import transformers
from PIL import Image
from open_clip import create_model_from_pretrained


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def add_results_to_json(file_path, metrics):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    for key in metrics:
        data[key] = metrics[key]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        json.dump(data, f, indent=4)


def log_samples(file_path, task_id, samples):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data[task_id] = samples

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        json.dump(data, f, indent=4)


def calculate_accuracy_and_stderr(scores):
    scores = np.array(scores)
    accuracy = np.mean(scores)
    standard_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return accuracy, standard_error


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".JPG")


def list_images(image_dir):
    return sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.endswith(IMAGE_EXTENSIONS)
    )


class CLIPSimilarity:
    def __init__(self, model_name, pretrained=None, device=None):
        self.model_name = model_name
        self.device = (
            device if device else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model, self.preprocess = create_model_from_pretrained(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

    def _process_images(self, image_paths, batch_size=64):
        """L2-normalised image features for a list of paths, in batches."""
        features = []
        for start in range(0, len(image_paths), batch_size):
            batch = []
            for image_path in image_paths[start:start + batch_size]:
                with Image.open(image_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    batch.append(self.preprocess(img))
            images = torch.stack(batch).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
            features.append(image_features / image_features.norm(dim=-1, keepdim=True))
        return torch.cat(features)

    def calculate_image_alignment(self, image_paths, ref_image_paths, batch_size=64):
        """Cosine similarity of every generated image against every reference image.

        Returns one score per generated image (its mean similarity to the reference
        set); the mean over those scores is the CLIP image alignment of the run.
        """
        features = self._process_images(image_paths, batch_size)
        ref_features = self._process_images(ref_image_paths, batch_size)
        return (features @ ref_features.t()).mean(dim=1).tolist()


def get_clip_model():
    # OpenAI ViT-B/32, the backbone the concept101 image-alignment protocol uses.
    # The `-quickgelu` config is the one that matches the released OpenAI weights.
    return CLIPSimilarity(model_name="ViT-B-32-quickgelu", pretrained="openai")

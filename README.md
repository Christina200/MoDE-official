# MoDE
[NeurIPS 2025] Official implementation of "Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models" by Xiwen Wei, Mustafa Munir, Radu Marculescu. 

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.arxiv.org/abs/2512.03125)
[![dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/ChristinaW/MoDE-official)

## Update

**August 16, 2026:** The processed dataset has been temporarily made private after I identified unintended personal information in it. I am removing the affected content and verifying the dataset. Public access will be restored once this process is complete. 

## Abstract

Unified Multimodal Generative Models (UMGMs) unify visual understanding and image generation within a single autoregressive framework. However, their ability to continually learn new tasks is severely hindered by catastrophic forgetting, both within a modality (intra-modal) and across modalities (inter-modal). While intra-modal forgetting has been studied in prior continual learning (CL) work, inter-modal forgetting remains largely unexplored. In this paper, we identify and empirically validate this phenomenon in UMGMs and provide a theoretical explanation rooted in gradient conflict between modalities. To address both intra- and inter-modal forgetting, we propose Modality-Decoupled Experts (MoDE), a lightweight and scalable architecture that isolates modality-specific updates to mitigate the gradient conflict and leverages knowledge distillation to prevent catastrophic forgetting and preserve pre-trained capabilities. Unlike previous CL methods that remain modality-coupled and suffer from modality gradient conflict, MoDE explicitly decouples modalities to prevent interference. Experiments across diverse benchmarks demonstrate that MoDE significantly mitigates both inter- and intra-modal forgetting, outperforming prior CL baselines in unified multimodal generation settings.

## Method

MoDE attaches a set of LoRA experts to the MLP projections (`up_proj`, `down_proj`, `gate_proj`) of
[Anole-7b](https://huggingface.co/leloy/Anole-7b-v0.1-hf), with a router that dispatches each token
according to its modality. Text and image tokens therefore take separate expert paths, which is what
decouples their gradients.

During training the same model doubles as its own teacher: with the experts disabled it *is* the
frozen pre-trained model, so a KL term on the image-generation stream preserves the base model's
generation ability without holding a second 7B checkpoint in memory.

## Repository layout

```
training/          MoDE training (train.py, distillation trainer, data pipeline)
evaluation/        VQA and text-to-image evaluation
scripts/train/     The five-task continual sequence
scripts/eval/      Evaluation of the final model
third_party/       Forks of transformers and peft that implement MoDE
data/concept101/   Text prompts for the text-to-image evaluation
```

MoDE is implemented in two forked libraries, both vendored here:

- `third_party/transformers` — adds `MMoEChameleonForConditionalGeneration`, which routes tokens
  by modality and exposes the train/inference modes used for image generation.
- `third_party/peft` — adds the `MMoELora` tuner (`MMoELoraConfig`, `MMoELoraModel`,
  `MMoELoRALinear`) implementing the modality-routed experts and the teacher-mode switch.

Both forks are stripped to what MoDE uses: `transformers` keeps Chameleon/Anole, the Llama
tokenizers and the PyTorch core (no other models, no TensorFlow/Flax, no pipelines), and `peft`
keeps only the MMoELoRA tuner. See each directory's README/`pyproject.toml` for details.

## Environment setup

Requires a CUDA 12.1 toolchain and, for training, 8 GPUs with at least 40GB each.

```bash
conda create -n mode python=3.10 -y
conda activate mode

# PyTorch 2.1.1 / CUDA 12.1
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# The MoDE forks -- install these instead of the PyPI transformers/peft
pip install -e third_party/transformers
pip install -e third_party/peft
```

Verify that both forks are the ones being imported:

```bash
python -c "
import transformers, peft
from transformers import MMoEChameleonForConditionalGeneration
from peft import MMoELoraConfig
print(transformers.__file__); print(peft.__file__)"
```

Both paths must point inside `third_party/`.

## Data

The preprocessed data is on HuggingFace at
[`ChristinaW/MoDE-official`](https://huggingface.co/datasets/ChristinaW/MoDE-official):

```bash
pip install huggingface_hub
huggingface-cli download ChristinaW/MoDE-official --repo-type dataset --local-dir .
```

This provides:

| Path | Used by | Contents |
| --- | --- | --- |
| `data/<TASK>/train_data.jsonl` | training | Pre-tokenized samples for each task |
| `data/laion_data.jsonl` | training | Image-generation stream for the distillation loss |
| `instructions/<TASK>/<split>.json` | VQA eval | Questions, answers and relative image paths |

VQA evaluation uses the **test** split for ScienceQA, ImageNet and GQA, but the **val** split for
TextVQA and VizWiz: those two benchmarks hold their test answers out for leaderboard submission, so
their `test.json` has no `answer` field and accuracy cannot be computed against it. `scripts/eval/eval_vqa.sh`
already selects the right split per task, and the loader raises a clear error if pointed at an
unlabelled one.

Training reads only the `.jsonl` files, so **no raw images are needed to train**.

VQA evaluation does need the raw images, since the instruction files store paths relative to an
image root. Those come from the source benchmarks (ScienceQA, TextVQA, ImageNet, GQA, VizWiz) as
assembled by [CoIN](https://github.com/zackschen/CoIN); point `EVAL_DATA_DIR` at that root.

## Training

MoDE is trained over five tasks in sequence — ScienceQA → TextVQA → ImageNet → GQA → VizWiz — where
each task starts from the adapter produced by the previous one.

```bash
bash scripts/train/train_all.sh
```

Or one task at a time:

```bash
bash scripts/train/1_ScienceQA.sh
bash scripts/train/2_TextVQA.sh
bash scripts/train/3_ImageNet.sh
bash scripts/train/4_GQA.sh
bash scripts/train/5_VizWiz.sh
```

Each writes its adapter to `training/outputs/<TASK>/<RUN_NAME>/final_merged_model`. The final model
of the sequence — the one the evaluation scripts use by default — is the VizWiz output.

Settings are environment variables (see `scripts/train/common.sh`); the defaults reproduce the
paper's configuration:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NPROC` | `8` | GPUs for `torchrun` |
| `LORA_NUM` | `4` | Number of modality-decoupled experts |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | LoRA rank and alpha |
| `LAMBDA_DISTILL` | `0.3` | Weight of the distillation loss |
| `LR` | `1e-4` | Learning rate |
| `OUTPUT_ROOT` | `training/outputs` | Where adapters are written |
| `RUN_NAME` | `mode_3_4exp` | Names the run's output subdirectory |
| `WANDB_ENTITY` | *(empty)* | Set to enable Weights & Biases logging |

For example, on 4 GPUs with a different output location:

```bash
NPROC=4 OUTPUT_ROOT=/scratch/mode_runs bash scripts/train/train_all.sh
```

On a SLURM cluster, wrap the entrypoint in your own batch script:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:8
bash scripts/train/train_all.sh
```

## Evaluation

Both scripts evaluate the end of the continual sequence by default. Override `CKPT` to evaluate a
different adapter.

**Intra-modal (VQA).** Scores the final model on all five tasks, which is what exposes forgetting of
the earlier ones:

```bash
EVAL_DATA_DIR=/path/to/cl_datasets bash scripts/eval/eval_vqa.sh
```

**Inter-modal (text-to-image).** Generates images from the 101 concept101 prompts and reports CLIP
image alignment against the real concept101 target images, measuring how much image-generation
ability survived:

```bash
bash scripts/eval/eval_t2i.sh
```

This needs the concept101 reference images in `data/concept101/target_images` (one real image per
concept, from the [CustomConcept101](https://github.com/adobe-research/custom-diffusion/blob/main/customconcept101/README.md)
benchmark). Point `--target_path` elsewhere if you keep them outside the repo.

Results and per-sample logs are written under `evaluation/outputs/`.


## Citation

```bibtex
@article{wei2026mitigating,
  title={Mitigating intra-and inter-modal forgetting in continual learning of unified multimodal models},
  author={Wei, Xiwen and Munir, Mustafa and Marculescu, Radu},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={151991--152019},
  year={2026}
}
```

## Acknowledgements

MoDE builds on [Anole](https://huggingface.co/leloy/Anole-7b-v0.1-hf), [MedMax](https://github.com/Hritikbansal/medmax.git) and Meta's
[Chameleon](https://github.com/facebookresearch/chameleon), and extends
[transformers](https://github.com/huggingface/transformers) and
[peft](https://github.com/huggingface/peft) (both Apache-2.0; their licenses are kept in each
`third_party/` directory). The continual-learning task suite follows
[CoIN](https://github.com/zackschen/CoIN).

## License

MIT, see [LICENSE](LICENSE).

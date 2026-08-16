import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Continual training of MoDE on a single task.")

    parser.add_argument('--train_data', required=True, type=str, help="path to tokenized train data")
    parser.add_argument('--val_data', default="", type=str, help="path to tokenized val data")

    parser.add_argument('--ckpt', required=True, type=str,
                        help="adapter to resume from; use the base model id for the first task")
    parser.add_argument('--base_model', default='leloy/Anole-7b-v0.1-hf', type=str,
                        help="base model the MoDE experts are attached to")
    parser.add_argument('--ds', required=True, type=str, help="deepspeed config")
    parser.add_argument('--output_dir', required=True, type=str, help="output directory")
    parser.add_argument('--resume', default=False, action='store_true', help='resume from the last checkpoint')

    # Knowledge distillation
    parser.add_argument('--img_gen_data', required=True, type=str,
                        help="path to the image-generation data used for distillation")
    parser.add_argument('--lambda_distill', type=float, default=0.3, help="weight of the distillation loss")

    # Modality-decoupled experts
    parser.add_argument('--lora_num', type=int, default=4, help="number of LoRA experts")
    parser.add_argument('--lora_r', type=int, default=8, help="lora r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="lora dropout")
    parser.add_argument('--task_id', type=int, default=-1, help="ID of the task in the continual sequence")

    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epoch', type=float, default=1.0, help="num of epochs")
    parser.add_argument('--grad_acc', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--steps', type=int, default=-1, help="num of training steps")
    parser.add_argument('--bs', type=int, default=1, help="per device batch size")
    parser.add_argument('--save_strategy', default="no", type=str, choices=["no", "epoch", "steps"],
                        help="save strategy")
    parser.add_argument('--save_steps', type=float, default=0.1, help="save checkpoints every save_steps")
    parser.add_argument('--logging_steps', type=float, default=0.001, help="logging interval")
    parser.add_argument('--eval_steps', type=float, default=0.1, help="evaluation interval")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="warmup ratio")
    parser.add_argument('--lr_scheduler', default="cosine", type=str, help="lr scheduler")

    parser.add_argument('--bf16', default=False, action='store_true', help='bf16')
    parser.add_argument('--fp16', default=False, action='store_true', help='fp16')

    parser.add_argument('--wandb', default=False, action='store_true', help='log to wandb')
    parser.add_argument('--wandb_entity', default="", type=str, help='wandb entity')
    parser.add_argument('--wandb_project', default="mode", type=str, help='wandb project')
    parser.add_argument('--name', default="", type=str, help="wandb run name")

    return parser.parse_args()

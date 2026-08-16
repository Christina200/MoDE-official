import os
from functools import partial

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import save_file
from transformers import ChameleonForConditionalGeneration, TrainingArguments

from peft import MMoELoraConfig, get_peft_model

from .data import DistillDataset, distill_collate_fn
from .distillation_trainer import DistillationTrainer
from .parser import parse_args

ADAPTER_NAME = "MMOELORA"


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.wandb:
        if local_rank == 0:
            wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=args)
            if args.name != "":
                wandb.run.name = args.name
    else:
        os.environ['WANDB_DISABLED'] = 'true'

    os.makedirs(args.output_dir, exist_ok=True)

    # The experts are always attached to the pre-trained base model. For every task after
    # the first, `--ckpt` points at the adapter learned so far and is loaded on top.
    model = ChameleonForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    peft_config = MMoELoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_nums=args.lora_num,
        target_modules=["up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.ckpt != args.base_model:
        model.load_adapter(args.ckpt, adapter_name=ADAPTER_NAME)
        model.set_adapter(adapter_name=ADAPTER_NAME)

    train_dataset = DistillDataset(args.train_data, args.img_gen_data)
    val_dataset = DistillDataset(args.val_data, args.img_gen_data) if args.val_data != "" else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epoch,
        max_steps=args.steps,
        gradient_accumulation_steps=args.grad_acc,
        per_device_train_batch_size=args.bs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        eval_strategy="steps" if args.val_data != "" else "no",
        eval_steps=args.eval_steps,
        deepspeed=args.ds,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        max_grad_norm=10.0
    )

    trainer = DistillationTrainer(
        lambda_distill=args.lambda_distill,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(distill_collate_fn, task_id=args.task_id),
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if dist.is_initialized():
        dist.barrier()
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        if hasattr(model, "module"):
            model = model.module

        save_path = os.path.join(args.output_dir, "final_merged_model")
        print(f"Saving model to {save_path} (Hugging Face format)...")
        model.save_pretrained(save_path)

        # save_pretrained drops the per-expert weights, so write the adapter tensors explicitly
        adapter_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k}
        adapter_save_path = os.path.join(save_path, "adapter_model.safetensors")
        save_file(adapter_state_dict, adapter_save_path)
        print(f"Adapter weights saved to {adapter_save_path}")


if __name__ == "__main__":
    main()

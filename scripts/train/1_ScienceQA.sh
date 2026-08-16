#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

torchrun --nproc_per_node=${NPROC} -m training.train \
    --train_data "${DATA_ROOT}/ScienceQA/train_data.jsonl" \
    --ckpt "${BASE_MODEL}" \
    --base_model "${BASE_MODEL}" \
    --ds training/ds_config.json \
    --output_dir "${OUTPUT_ROOT}/ScienceQA/${RUN_NAME}" \
    --img_gen_data "${IMG_GEN_DATA}" \
    --lambda_distill ${LAMBDA_DISTILL} \
    --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_num ${LORA_NUM} \
    --epoch 1 --bs 1 --lr ${LR} --warmup_ratio 0.1 --bf16 \
    --save_strategy steps --save_steps 100 --eval_steps 100 \
    --name "ScienceQA-${RUN_NAME}" \
    $(wandb_args)

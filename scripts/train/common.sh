#!/bin/bash
# Shared settings for the MoDE continual-learning sequence.
# Override any of these from the environment, e.g.:
#   OUTPUT_ROOT=/scratch/mode_runs NPROC=4 bash scripts/train/train_all.sh

export BASE_MODEL="${BASE_MODEL:-leloy/Anole-7b-v0.1-hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-training/outputs}"
export RUN_NAME="${RUN_NAME:-mode_3_4exp}"
export DATA_ROOT="${DATA_ROOT:-data}"
export IMG_GEN_DATA="${IMG_GEN_DATA:-${DATA_ROOT}/laion_data.jsonl}"
export NPROC="${NPROC:-8}"

# MoDE hyper-parameters
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_NUM="${LORA_NUM:-4}"
export LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.3}"
export LR="${LR:-1e-4}"

# Set WANDB_ENTITY to enable logging; left empty, training runs with wandb disabled.
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# Emits the --wandb flags only when an entity is configured.
wandb_args() {
    if [ -n "$WANDB_ENTITY" ]; then
        echo "--wandb --wandb_entity ${WANDB_ENTITY}"
    fi
}

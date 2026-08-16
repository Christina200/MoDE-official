#!/bin/bash
# Shared settings for MoDE evaluation.

export BASE_MODEL="${BASE_MODEL:-leloy/Anole-7b-v0.1-hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-training/outputs}"
export RUN_NAME="${RUN_NAME:-mode_3_4exp}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-evaluation/outputs}"

# The adapter to evaluate. By default this is the end of the continual sequence,
# i.e. the model after the final task (VizWiz).
export CKPT="${CKPT:-${OUTPUT_ROOT}/VizWiz/${RUN_NAME}/final_merged_model}"

# Instruction JSONs (from the HuggingFace dataset) and the image root they refer to.
export INSTRUCTIONS_ROOT="${INSTRUCTIONS_ROOT:-instructions}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-cl_datasets}"

export LORA_R="${LORA_R:-8}"
export LORA_NUM="${LORA_NUM:-4}"

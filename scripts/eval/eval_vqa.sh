#!/bin/bash
# VQA evaluation of the final model on every task in the continual sequence.
# Running all five shows how much of each earlier task survives later training.
set -e
source "$(dirname "$0")/common.sh"

# task:instruction-file pairs. ScienceQA, ImageNet and GQA are evaluated on the
# test split; TextVQA and VizWiz on the val split, matching the reported results.
TASKS=(
    "ScienceQA:test.json"
    "TextVQA:val.json"
    "ImageNet:test.json"
    "GQA:test.json"
    "VizWiz:val.json"
)

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    split="${entry##*:}"

    python -m evaluation.eval_vqa_pretrained \
        --ckpt "${CKPT}" \
        --base_model "${BASE_MODEL}" \
        --instruction_file "${INSTRUCTIONS_ROOT}/${task}/${split}" \
        --eval_data_dir "${EVAL_DATA_DIR}" \
        --save_dir "${EVAL_OUTPUT_ROOT}/vqa/${RUN_NAME}" \
        --save_name "${RUN_NAME}-vqa" \
        --task_id "${task}" \
        --lora_num ${LORA_NUM} --lora_r ${LORA_R}
done

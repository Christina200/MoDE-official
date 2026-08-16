#!/bin/bash
# Text-to-image evaluation: CLIP image alignment between images generated from the
# concept101 prompts and the real concept101 target images. This measures how much
# image-generation ability is retained after the continual sequence.
set -e
source "$(dirname "$0")/common.sh"

python -m evaluation.t2i_pretrained \
    --ckpt "${CKPT}" \
    --base_model "${BASE_MODEL}" \
    --prompt_file "data/concept101/text_input.json" \
    --target_path "data/concept101/target_images" \
    --save_dir "${EVAL_OUTPUT_ROOT}/t2i/${RUN_NAME}" \
    --save_name "${RUN_NAME}-t2i" \
    --lora_num ${LORA_NUM} --lora_r ${LORA_R}

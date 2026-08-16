#!/bin/bash
# Trains MoDE across the full continual sequence. Each task starts from the
# adapter produced by the previous one.
set -e

HERE="$(dirname "$0")"

bash "${HERE}/1_ScienceQA.sh"
bash "${HERE}/2_TextVQA.sh"
bash "${HERE}/3_ImageNet.sh"
bash "${HERE}/4_GQA.sh"
bash "${HERE}/5_VizWiz.sh"

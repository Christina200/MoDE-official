# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]
SEQ_CLS_HEAD_NAMES = ["score", "classifier"]
INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
DUMMY_TARGET_MODULES = "dummy-target-modules"
DUMMY_MODEL_CONFIG = {"model_type": "custom"}

# If users specify more than this number of target modules, we apply an optimization to try to reduce the target modules
# to a minimal set of suffixes, which makes loading faster. We only apply this when exceeding a certain size since
# otherwise there is no point in optimizing and there is a small chance of bugs in the optimization algorithm, so no
# point in taking unnecessary risks. See #2045 for more context.
MIN_TARGET_MODULES_FOR_OPTIMIZATION = 20

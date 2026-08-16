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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch

from .config import PeftConfig
from .peft_model import PeftModel, PeftModelForCausalLM
from .tuners import MMoELoraConfig, MMoELoraModel
from .tuners.tuners_utils import BaseTuner


if TYPE_CHECKING:
    from transformers import PreTrainedModel


MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[PeftModel]] = {
    "CAUSAL_LM": PeftModelForCausalLM,
}

PEFT_TYPE_TO_CONFIG_MAPPING: dict[str, type[PeftConfig]] = {
    "MMOELORA": MMoELoraConfig,
}

PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[BaseTuner]] = {
    "MMOELORA": MMoELoraModel,
}


def get_peft_config(config_dict: dict[str, Any]) -> PeftConfig:
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
) -> PeftModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
    """
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING:
        return PeftModel(model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)

    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    )

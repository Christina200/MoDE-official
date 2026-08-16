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

import re
import warnings
from dataclasses import asdict
from enum import Enum

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import _get_submodules

from .config import MMoELoraConfig
from .layer import MMoELoraLayer, MMoELoRALinear


class MMoELoraModel(BaseTuner):

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, teacher_mode: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        self.teacher_mode = teacher_mode

    def _check_new_adapter_config(self, config: MMoELoraConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )
    
    @staticmethod
    def _check_target_module_exists(mmoe_lora_config, key):
        return check_target_module_exists(mmoe_lora_config, key)
    
    def _create_and_replace(
        self,
        mmoe_lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config[adapter_name].r,
            "lora_alpha": self.peft_config[adapter_name].lora_alpha,
            "lora_dropout": self.peft_config[adapter_name].lora_dropout,
            "lora_nums": self.peft_config[adapter_name].lora_nums,
            "blc_alpha": self.peft_config[adapter_name].blc_alpha,
            "blc_weight": self.peft_config[adapter_name].blc_weight,
            "fan_in_fan_out": self.peft_config[adapter_name].fan_in_fan_out,
            "merge_weights": (self.peft_config[adapter_name].merge_weights or self.peft_config[adapter_name].inference_mode) and not is_hf_device_map_available,
        }

        if isinstance(self.peft_config[adapter_name].target_modules, str):
            target_module_found = re.fullmatch(self.peft_config[adapter_name].target_modules, current_key)
        else:
            target_module_found = any(current_key.endswith(target_key) for target_key in self.peft_config[adapter_name].target_modules)
        if target_module_found: # here
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, current_key)
            bias = target.bias is not None

            if isinstance(target, torch.nn.Linear) and self.peft_config[adapter_name].enable_lora is None:
                new_module = MMoELoRALinear(target.in_features, target.out_features, bias=bias, teacher_mode=self.teacher_mode, **kwargs)

            self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config[adapter_name].target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )


    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  
        except AttributeError:
            return getattr(self.model, name)


    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, MMoELoraLayer):
                module.disable_adapters = False if enabled else True

    def _set_teacher_mode(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, MMoELoRALinear):
                module.teacher_mode = enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def _mark_only_adapters_as_trainable(self, model: nn.Module, bias: str = "none") -> None:
        for n, p in model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        if bias == "none":
            return
        elif bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in model.modules():
                if isinstance(m, MMoELoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError
        
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config
    
    @staticmethod
    def _check_target_module_exists(mmoe_lora_config, key):
        return check_target_module_exists(mmoe_lora_config, key)
    
    def set_adapter(self, adapter_name: str | list[str]) -> None:
        for module in self.model.modules():
            if isinstance(module, MMoELoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name


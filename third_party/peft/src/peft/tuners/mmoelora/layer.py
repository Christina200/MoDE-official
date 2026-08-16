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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.other import transpose


class MMoELoraLayer:
    adapter_layer_names = ("lora_A_text0", "lora_B_text0", "lora_route_text", "lora_A_image0", "lora_B_image0")
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        teacher_mode: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
        self.teacher_mode = teacher_mode

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            module_dict.requires_grad_(True)
        self._active_adapter = adapter_names

class MMoELoRALinear(nn.Linear, MMoELoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        teacher_mode: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MMoELoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, teacher_mode=teacher_mode)

        self.lora_num = lora_nums
        
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0:
            self.lora_route_text = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A_text{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B_text{i}", nn.Linear(r, out_features, bias=False))
            for i in range(1):
                setattr(self, f"lora_A_image{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B_image{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A_text0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A_text{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B_text{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route_text.weight, a=math.sqrt(5))
        
        if hasattr(self, "lora_A_image0"):
            for i in range(1):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A_image{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B_image{i}").weight)


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route_text.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A_text{i}").train(mode)
            getattr(self, f"lora_B_text{i}").train(mode)
        getattr(self, f"lora_A_image0").train(mode)
        getattr(self, f"lora_B_image0").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route_text.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A_text{i}").eval()
            getattr(self, f"lora_B_text{i}").eval()
        getattr(self, f"lora_A_image0").eval()
        getattr(self, f"lora_B_image0").eval()

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor = None, return_aux: bool = False, task_types=None):

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(") 
        elif self.teacher_mode:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            if modality_ids is not None:
                orig_shape = x.shape[:-1]
                x = x.view(-1, x.shape[-1])
                modality_ids_flat = modality_ids.view(-1)
                vl_out = torch.zeros(x.size(0), self.out_features, dtype=x.dtype, device=x.device)
                text_mask = modality_ids_flat == 0
                if text_mask.any():
                    x_text = x[text_mask]
                    x_text_float = x_text.to(torch.float32)
                    lora_route_text_weight = self.lora_route_text.weight.to(torch.float32)
                    lora_route_text_bias = self.lora_route_text.bias.to(torch.float32) if self.lora_route_text.bias is not None else None
                    text_route_weight = torch.nn.functional.linear(x_text_float, lora_route_text_weight, lora_route_text_bias)
                    text_route_weight = nn.functional.softmax(text_route_weight, dim=-1)
                    text_route_weight = text_route_weight.to(x.dtype)
                    dropped_x_text = self.lora_dropout(x_text).to(x.dtype)
                    for i in range(self.lora_num):
                        expert_weight = torch.unsqueeze(text_route_weight[:,i], -1)
                        expert_out = getattr(self, f"lora_A_text{i}")(dropped_x_text)
                        expert_out = getattr(self, f"lora_B_text{i}")(expert_out)

                        vl_out[text_mask] += expert_weight * expert_out * self.scaling 

                image_mask = modality_ids_flat == 1
                if image_mask.any():
                    x_image = x[image_mask]
                    dropped_x_image = self.lora_dropout(x_image).to(x.dtype)
                    for i in range(1):
                        expert_out = getattr(self, f"lora_A_image{i}")(dropped_x_image)
                        expert_out = getattr(self, f"lora_B_image{i}")(expert_out)
                        vl_out[image_mask] += expert_out * self.scaling 

                result = result + vl_out.view(*orig_shape, self.out_features) 
            else:
                raise ValueError("modality_ids must be provided for modality-specific routing.")

        return result
    


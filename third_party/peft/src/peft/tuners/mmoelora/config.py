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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MMoELoraConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": ""
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": ""})
    lora_nums: int = field(default=None, metadata={"help": ""})
    blc_alpha: int = field(default=None, metadata={"help": ""})
    blc_weight: int = field(default=None, metadata={"help": ""})
    lora_dropout: float = field(default=None, metadata={"help": ""})
    merge_weights: bool = field(
        default=False, metadata={"help": ""}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": ""},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": ""})
    bias: str = field(default="none", metadata={"help": ""})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": ""
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )


    def __post_init__(self):
        self.peft_type = PeftType.MMOELORA


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional, Dict

from torch import Tensor, nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import loralibs


class LoRAMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_cfg: Dict = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.lora_type = lora_cfg['lora_type']
        self.lora_r = lora_cfg['lora_r']
        pos = lora_cfg['lora_pos']
        self.fc1 = getattr(loralibs, '{}Linear'.format(self.lora_type))(in_features, hidden_features, r=self.lora_r, bias=bias) if pos['up'] else nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = getattr(loralibs, '{}Linear'.format(self.lora_type))(hidden_features, out_features, r=self.lora_r, bias=bias) if pos['down'] else nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
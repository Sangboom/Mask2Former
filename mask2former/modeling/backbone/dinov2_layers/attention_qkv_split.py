# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
from typing import Dict

from torch import Tensor
from torch import nn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import loralibs

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class LoRAQKVSplitAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lora_cfg: Dict = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.lora_type = lora_cfg['lora_type']
        self.lora_r = lora_cfg['lora_r']
        pos = lora_cfg['lora_pos']

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = getattr(loralibs, '{}Linear'.format(self.lora_type))(dim, dim, r=self.lora_r, bias=qkv_bias) if pos['q'] else nn.Linear(dim, dim, bias=qkv_bias)
        self.k = getattr(loralibs, '{}Linear'.format(self.lora_type))(dim, dim, r=self.lora_r, bias=qkv_bias) if pos['k'] else nn.Linear(dim, dim, bias=qkv_bias)
        self.v = getattr(loralibs, '{}Linear'.format(self.lora_type))(dim, dim, r=self.lora_r, bias=qkv_bias) if pos['v'] else nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = getattr(loralibs, '{}Linear'.format(self.lora_type))(dim, dim, r=self.lora_r, bias=proj_bias) if pos['o'] else nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.scale # added
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # added
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # added

        # (B, N, 3, H, C // H) -> (3, B, H, N, C // H)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffLoRAQKVSplitAttention(LoRAQKVSplitAttention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads) # added
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads) # added
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads) # added

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
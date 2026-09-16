from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import QuantConfig
from .base import QuantOpBase


class QuantConv2d(QuantOpBase):
    op_kind = "conv"

    def __init__(self, float_op: nn.Conv2d, cfg: QuantConfig, adaround: bool = False):
        super().__init__(
            cfg,
            quantize_weight=cfg.quantize_weight,
            quantize_input=cfg.quantize_input,
            quantize_output=cfg.quantize_output,
            weight_ch_axis=0,
            adaround=adaround,
        )
        self.float_op = float_op

    @classmethod
    def from_float(cls, module: nn.Conv2d, cfg: QuantConfig, adaround: bool = False):
        return cls(module, cfg, adaround=adaround)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.float_op
        x = self._qin(x)
        w = self._qw(conv.weight)
        y = F.conv2d(x, w, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return self._qout(y)

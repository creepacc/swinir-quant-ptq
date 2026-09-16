from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import QuantConfig
from .base import QuantOpBase


class QuantLinear(QuantOpBase):
    op_kind = "linear"

    def __init__(self, float_op: nn.Linear, cfg: QuantConfig, adaround: bool = False):
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
    def from_float(cls, module: nn.Linear, cfg: QuantConfig, adaround: bool = False):
        return cls(module, cfg, adaround=adaround)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._qin(x)
        w = self._qw(self.float_op.weight)
        y = F.linear(x, w, self.float_op.bias)
        return self._qout(y)


class QuantMatMul(QuantOpBase):
    op_kind = "matmul"

    def __init__(
        self,
        cfg: QuantConfig,
        quantize_a: bool | None = None,
        quantize_b: bool | None = None,
        quantize_output: bool | None = None,
    ):
        qa = cfg.quantize_input if quantize_a is None else quantize_a
        qb = cfg.quantize_input if quantize_b is None else quantize_b
        qout = (True if "matmul" not in cfg.fp32_ops else False) if quantize_output is None else quantize_output
        super().__init__(
            cfg,
            quantize_weight=False,
            quantize_input=qa or qb,
            quantize_output=qout,
        )
        self.quantize_a = qa
        self.quantize_b = qb

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.quantize_a:
            a = self._qin(a)
        if self.quantize_b:
            b = self._qin(b)
        return self._qout(a @ b)

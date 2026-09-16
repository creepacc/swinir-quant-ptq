from __future__ import annotations

import torch

from ..config import QuantConfig
from .base import QuantOpBase


class QuantElementwise(QuantOpBase):
    op_kind = "ewise"

    def __init__(self, cfg: QuantConfig, quantize_output: bool = True):
        super().__init__(
            cfg,
            quantize_weight=False,
            quantize_input=False,
            quantize_output=quantize_output,
        )


class QuantAdd(QuantElementwise):
    op_kind = "add"

    def __init__(self, cfg: QuantConfig, quantize_output: bool = True, residual: bool = False):
        super().__init__(cfg, quantize_output=quantize_output)
        self.is_residual = bool(residual)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._qout(a + b)


class QuantMul(QuantElementwise):
    op_kind = "mul"

    def forward(self, a: torch.Tensor, b) -> torch.Tensor:
        return self._qout(a * b)


class QuantSub(QuantElementwise):
    op_kind = "sub"

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._qout(a - b)


class QuantDiv(QuantElementwise):
    op_kind = "div"

    def forward(self, a: torch.Tensor, b) -> torch.Tensor:
        return self._qout(a / b)

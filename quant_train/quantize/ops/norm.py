from __future__ import annotations

import torch
import torch.nn as nn

from ..config import QuantConfig
from .base import QuantOpBase


class QuantLayerNorm(QuantOpBase):
    """Interface only; default is FP32 (layer_norm in fp32_ops)."""

    op_kind = "layer_norm"

    def __init__(self, float_op: nn.LayerNorm, cfg: QuantConfig):
        enable = "layer_norm" not in cfg.fp32_ops
        super().__init__(
            cfg,
            quantize_weight=False,
            quantize_input=False,
            quantize_output=enable and cfg.quantize_output,
        )
        self.float_op = float_op

    @classmethod
    def from_float(cls, module: nn.LayerNorm, cfg: QuantConfig):
        return cls(module, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = getattr(self, "channel_restore", None)
        q = getattr(self, "stream_Q", None)
        if s is not None:
            x = x * s.to(device=x.device, dtype=x.dtype)
        if q is not None:
            qt = q.to(device=x.device, dtype=x.dtype)
            x = x @ qt.transpose(0, 1)
        y = self.float_op(x)
        if q is not None and getattr(self, "stream_Q_reapply", False):
            y = y @ q.to(device=y.device, dtype=y.dtype)
        if s is not None and getattr(self, "channel_restore_reapply", False):
            y = y / s.to(device=y.device, dtype=y.dtype).clamp(min=1e-8)
        return self._qout(y)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import QuantConfig
from ..fake_quantize import UniformFakeQuantize
from .base import QuantOpBase


def _act_output_on(cfg: QuantConfig, kind: str) -> bool:
    if kind in cfg.fp32_ops:
        return False
    return cfg.act_backend in ("fakequant_out", "lut")


def softmax_exp_lut(xq: torch.Tensor, fq: UniformFakeQuantize) -> torch.Tensor:
    """256-entry exp LUT on the fake-quant grid of xq (nearest index)."""
    lo_b, hi_b = fq._clip_pair()
    if lo_b is None or hi_b is None:
        lo_b, hi_b = fq._bounds_from_qparams()
    lo = lo_b.to(device=xq.device, dtype=xq.dtype).reshape(1)
    hi = lo + (hi_b.to(device=xq.device, dtype=xq.dtype).reshape(1) - lo).clamp(min=fq.min_scale)
    n = int(fq.qmax - fq.qmin)
    scale = (hi - lo) / float(n)
    idx = ((xq.detach() - lo) / scale.clamp(min=fq.min_scale)).round().clamp(0, n).long()
    table = torch.exp(lo + torch.arange(n + 1, device=xq.device, dtype=xq.dtype) * scale)
    e = table[idx]
    return e + (torch.exp(xq) - e).detach()


class QuantActivation(QuantOpBase):
    op_kind = "activation"
    backend: str = "fp32"

    def __init__(self, cfg: QuantConfig, kind: str):
        self.backend = "fp32" if kind in cfg.fp32_ops else cfg.act_backend
        super().__init__(
            cfg,
            quantize_weight=False,
            quantize_input=False,
            quantize_output=_act_output_on(cfg, kind),
        )
        self.kind = kind

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._qout(self._apply_act(x))


class QuantGELU(QuantActivation):
    op_kind = "gelu"

    def __init__(self, cfg: QuantConfig, float_op: nn.GELU | None = None):
        super().__init__(cfg, "gelu")
        self.float_op = float_op or nn.GELU()

    @classmethod
    def from_float(cls, module: nn.GELU, cfg: QuantConfig):
        return cls(cfg, module)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        return self.float_op(x)


class QuantSoftmax(QuantActivation):
    op_kind = "softmax"

    def __init__(self, cfg: QuantConfig, float_op: nn.Softmax | None = None):
        mode = getattr(cfg, "softmax_mode", "default")
        if mode == "lut":
            self.backend = "lut"
            self.kind = "softmax"
            QuantOpBase.__init__(
                self,
                cfg,
                quantize_weight=False,
                quantize_input=True,
                quantize_output=True,
            )
        else:
            super().__init__(cfg, "softmax")
        self.float_op = float_op or nn.Softmax(dim=-1)

    @classmethod
    def from_float(cls, module: nn.Softmax, cfg: QuantConfig):
        return cls(cfg, module)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        return self.float_op(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend != "lut":
            return self._qout(self._apply_act(x))
        z = x - x.amax(dim=-1, keepdim=True)
        zq = self._qin(z)
        e = softmax_exp_lut(zq, self.input_fq)
        y = e / e.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return self._qout(y)


class QuantLeakyReLU(QuantActivation):
    op_kind = "leaky_relu"

    def __init__(self, cfg: QuantConfig, float_op: nn.LeakyReLU | None = None):
        super().__init__(cfg, "leaky_relu")
        self.float_op = float_op or nn.LeakyReLU()

    @classmethod
    def from_float(cls, module: nn.LeakyReLU, cfg: QuantConfig):
        return cls(cfg, module)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        return self.float_op(x)


class QuantReLU(QuantActivation):
    op_kind = "relu"

    def __init__(self, cfg: QuantConfig, float_op: nn.ReLU | None = None):
        super().__init__(cfg, "relu")
        self.float_op = float_op or nn.ReLU()

    @classmethod
    def from_float(cls, module: nn.ReLU, cfg: QuantConfig):
        return cls(cfg, module)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        return self.float_op(x)


class QuantSiLU(QuantActivation):
    op_kind = "silu"

    def __init__(self, cfg: QuantConfig, float_op: nn.SiLU | None = None):
        super().__init__(cfg, "silu")
        self.float_op = float_op or nn.SiLU()

    @classmethod
    def from_float(cls, module: nn.SiLU, cfg: QuantConfig):
        return cls(cfg, module)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) if self.float_op is None else self.float_op(x)

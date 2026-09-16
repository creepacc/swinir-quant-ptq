from __future__ import annotations

import torch
import torch.nn as nn

from .observer import Observer, build_observer

# AdaRound rectified sigmoid (Nagel et al. ICML 2020, eq. 23)
ADAROUND_GAMMA = -0.1
ADAROUND_ZETA = 1.1


def adaround_h(alpha: torch.Tensor) -> torch.Tensor:
    """h(V) = clip(sigmoid(V) * (ζ − γ) + γ, 0, 1)."""
    return torch.clamp(
        torch.sigmoid(alpha) * (ADAROUND_ZETA - ADAROUND_GAMMA) + ADAROUND_GAMMA,
        0.0,
        1.0,
    )


def invert_adaround_h(rest: torch.Tensor) -> torch.Tensor:
    """V such that h(V) = fractional part (nearest-round init)."""
    rest = rest.clamp(1e-6, 1.0 - 1e-6)
    inner = ((rest - ADAROUND_GAMMA) / (ADAROUND_ZETA - ADAROUND_GAMMA)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(inner / (1.0 - inner))


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class FakeQuantizeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_quant_enabled = False
        self.observer_enabled = False

    def enable_observer(self, flag: bool = True):
        self.observer_enabled = flag

    def enable_fakequant(self, flag: bool = True):
        self.fake_quant_enabled = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformFakeQuantize(FakeQuantizeBase):
    """Uniform affine fake-quant with optional AdaRound on weights."""

    def __init__(
        self,
        observer: Observer,
        qmin: int = -128,
        qmax: int = 127,
        ch_axis: int = -1,
        min_scale: float = 1e-9,
        adaround: bool = False,
    ):
        super().__init__()
        self.observer = observer
        self.qmin = int(qmin)
        self.qmax = int(qmax)
        self.ch_axis = ch_axis
        self.min_scale = min_scale
        self.adaround = adaround
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        self.alpha = None  # created in init_alpha

    def extra_repr(self) -> str:
        return f"qmin={self.qmin}, qmax={self.qmax}, adaround={self.adaround}"

    def calculate_qparams(self):
        scale, zp = self.observer.calculate_qparams(self.qmin, self.qmax)
        self.scale = scale.detach()
        self.zero_point = zp.detach()

    def set_clip_bounds(self, lo: torch.Tensor, hi: torch.Tensor):
        lo = lo.detach().reshape(1).to(device=self.scale.device, dtype=self.scale.dtype)
        hi = hi.detach().reshape(1).to(device=self.scale.device, dtype=self.scale.dtype)
        self._drop_clip()
        self.register_buffer("clip_min", lo)
        self.register_buffer("clip_max", hi)
        self._sync_scale_zp()

    def _drop_clip(self):
        for name in ("clip_min", "clip_max"):
            if name in self._buffers:
                del self._buffers[name]
            if name in self._parameters:
                del self._parameters[name]
            if hasattr(self, name):
                delattr(self, name)

    def _bounds_from_qparams(self):
        scale = self.scale.detach().clamp(min=self.min_scale)
        zp = self.zero_point.detach()
        lo = (self.qmin - zp) * scale
        hi = (self.qmax - zp) * scale
        return lo.reshape(1), hi.reshape(1)

    def _clip_pair(self):
        lo = getattr(self, "clip_min", None)
        hi = getattr(self, "clip_max", None)
        return lo, hi

    def _sync_scale_zp(self):
        lo, hi = self._clip_pair()
        if lo is None or hi is None:
            return
        lo = lo.detach()
        hi = lo + (hi.detach() - lo).clamp(min=self.min_scale)
        scale = (hi - lo) / float(self.qmax - self.qmin)
        zp = self.qmin - lo / scale.clamp(min=self.min_scale)
        self.scale = scale.detach().reshape_as(self.scale)
        self.zero_point = zp.round().detach().reshape_as(self.zero_point)

    def enable_learn_bounds(self):
        cur_lo, cur_hi = self._clip_pair()
        if cur_lo is None or cur_hi is None:
            lo, hi = self._bounds_from_qparams()
        else:
            lo = cur_lo.detach().reshape(1)
            hi = cur_hi.detach().reshape(1)
        self._drop_clip()
        self.clip_min = nn.Parameter(lo.clone())
        self.clip_max = nn.Parameter(hi.clone())
        return self.clip_min, self.clip_max

    def freeze_bounds(self):
        lo, hi = self._clip_pair()
        if lo is None or hi is None:
            return
        self.set_clip_bounds(lo.detach().reshape(1).clone(), hi.detach().reshape(1).clone())

    def observe_tensor(self, x: torch.Tensor):
        self.observer.observe(x)

    def init_alpha(self, weight: torch.Tensor):
        if not self.adaround:
            return
        scale = self._reshape_qparam(self.scale, weight)
        w_scaled = weight.detach() / scale.clamp(min=self.min_scale)
        rest = w_scaled - torch.floor(w_scaled)
        self.alpha = nn.Parameter(invert_adaround_h(rest))

    def _reshape_qparam(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if p.numel() == 1 or self.ch_axis < 0:
            return p.view(1).to(device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[self.ch_axis] = -1
        return p.to(device=x.device, dtype=x.dtype).reshape(shape)

    def _fake_quant_uniform(self, x: torch.Tensor) -> torch.Tensor:
        lo_b, hi_b = self._clip_pair()
        if lo_b is not None and hi_b is not None:
            lo = lo_b.to(device=x.device, dtype=x.dtype)
            hi = lo + (hi_b.to(device=x.device, dtype=x.dtype) - lo).clamp(min=self.min_scale)
            xc = torch.minimum(torch.maximum(x, lo), hi)
            scale = (hi - lo) / float(self.qmax - self.qmin)
            q = RoundSTE.apply((xc - lo) / scale.clamp(min=self.min_scale))
            q = torch.clamp(q, 0, self.qmax - self.qmin)
            return q * scale + lo
        scale = self._reshape_qparam(self.scale, x).clamp(min=self.min_scale)
        zp = self._reshape_qparam(self.zero_point, x)
        x_int = RoundSTE.apply(x / scale) + zp
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        return (x_int - zp) * scale

    def _fake_quant_adaround(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._reshape_qparam(self.scale, x).clamp(min=self.min_scale)
        h = adaround_h(self.alpha)
        x_scaled = x / scale
        x_q = torch.clamp(torch.floor(x_scaled) + h, self.qmin, self.qmax)
        return x_q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self.observer.observe(x)
            buf = getattr(self, "_act_buf", None)
            if buf is not None:
                buf.add(self, x)
        if not self.fake_quant_enabled:
            return x
        if self.adaround and self.alpha is not None:
            return self._fake_quant_adaround(x)
        return self._fake_quant_uniform(x)

    def hard_round_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Commit AdaRound (or uniform FQ) to a hard-rounded dequant weight."""
        scale = self._reshape_qparam(self.scale, w).clamp(min=self.min_scale)
        zp = self._reshape_qparam(self.zero_point, w)
        if self.adaround and self.alpha is not None:
            h = adaround_h(self.alpha.detach())
            w_int = torch.floor(w.detach() / scale) + (h >= 0.5).to(w.dtype)
        else:
            w_int = torch.round(w.detach() / scale) + zp
        w_int = torch.clamp(w_int, self.qmin, self.qmax)
        return (w_int - zp) * scale

    def drop_alpha(self):
        if isinstance(self.alpha, nn.Parameter) and "alpha" in self._parameters:
            del self._parameters["alpha"]
        self.alpha = None
        self.adaround = False


class LUTFakeQuantize(FakeQuantizeBase):
    """Placeholder for later LUT activation quantization."""

    def __init__(self, observer: Observer, qmin: int = -128, qmax: int = 127, **_):
        super().__init__()
        self.observer = observer
        self.qmin = qmin
        self.qmax = qmax
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        self.lut = None

    def calculate_qparams(self):
        scale, zp = self.observer.calculate_qparams(self.qmin, self.qmax)
        self.scale = scale.detach()
        self.zero_point = zp.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self.observer.observe(x)
        if not self.fake_quant_enabled or self.lut is None:
            return x
        raise NotImplementedError("LUT table is not generated yet")


def make_fakequant(
    cfg_quant,
    ch_axis: int = -1,
    adaround: bool = False,
    bits: int | None = None,
    kind: str = "act",
) -> UniformFakeQuantize:
    if kind == "weight":
        bits = bits if bits is not None else cfg_quant.w_bits
        symmetric = cfg_quant.weight_symmetric
    else:
        bits = bits if bits is not None else cfg_quant.a_bits
        symmetric = cfg_quant.act_symmetric
    qmin, qmax = cfg_quant.qmin, cfg_quant.qmax
    if bits != 8:
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
    obs = build_observer(
        cfg_quant.observer,
        ch_axis=ch_axis,
        symmetric=symmetric,
        min_scale=cfg_quant.min_scale,
    )
    return UniformFakeQuantize(
        observer=obs,
        qmin=qmin,
        qmax=qmax,
        ch_axis=ch_axis,
        min_scale=cfg_quant.min_scale,
        adaround=adaround,
    )

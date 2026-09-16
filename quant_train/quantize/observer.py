from __future__ import annotations

import torch
import torch.nn as nn


class Observer(nn.Module):
    def __init__(self, ch_axis: int = -1, symmetric: bool = True, min_scale: float = 1e-9):
        super().__init__()
        self.ch_axis = ch_axis
        self.symmetric = symmetric
        self.min_scale = min_scale
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.int64))

    def _reduce(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ch_axis < 0:
            return x.min(), x.max()
        dims = [d for d in range(x.ndim) if d != self.ch_axis]
        return x.amin(dim=dims), x.amax(dim=dims)

    def observe(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def calculate_qparams(self, qmin: int, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
        mn = self.min_val.float()
        mx = self.max_val.float()
        if not torch.isfinite(mn).any() or not torch.isfinite(mx).any():
            scale = torch.ones_like(mn) if mn.ndim > 0 else torch.ones(1)
            zp = torch.zeros_like(scale)
            return scale, zp
        if self.symmetric:
            amax = torch.maximum(mn.abs(), mx.abs())
            scale = amax / max(abs(qmin), abs(qmax))
            zp = torch.zeros_like(scale)
        else:
            scale = (mx - mn) / float(qmax - qmin)
            zp = torch.round(qmin - mn / scale.clamp(min=self.min_scale))
        scale = scale.clamp(min=self.min_scale)
        return scale.detach(), zp.detach()


class MinMaxObserver(Observer):
    def observe(self, x: torch.Tensor) -> None:
        mn, mx = self._reduce(x.detach())
        if self.min_val.numel() != mn.numel():
            self.min_val = mn.clone()
            self.max_val = mx.clone()
        else:
            self.min_val = torch.minimum(self.min_val.to(mn.device), mn)
            self.max_val = torch.maximum(self.max_val.to(mx.device), mx)
        self.num_batches += 1


class MovingAverageMinMaxObserver(Observer):
    def __init__(self, momentum: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def observe(self, x: torch.Tensor) -> None:
        mn, mx = self._reduce(x.detach())
        if self.num_batches.item() == 0 or self.min_val.numel() != mn.numel():
            self.min_val = mn.clone()
            self.max_val = mx.clone()
        else:
            m = self.momentum
            self.min_val = (1 - m) * self.min_val.to(mn.device) + m * mn
            self.max_val = (1 - m) * self.max_val.to(mx.device) + m * mx
        self.num_batches += 1


def build_observer(name: str, **kwargs) -> Observer:
    name = (name or "minmax").lower()
    if name in ("ema", "moving_average", "movingaverageminmax"):
        return MovingAverageMinMaxObserver(**kwargs)
    return MinMaxObserver(**kwargs)

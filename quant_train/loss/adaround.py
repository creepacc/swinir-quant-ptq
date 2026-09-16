from __future__ import annotations

from typing import Iterable, List

import torch

from ..quantize.fake_quantize import adaround_h
from .base import PTQLossBase


def _as_list(alphas: Iterable[torch.Tensor] | None) -> List[torch.Tensor]:
    if alphas is None:
        return []
    if not isinstance(alphas, (list, tuple)):
        return [alphas]
    return list(alphas)


def adaround_softness(alphas: Iterable[torch.Tensor] | None) -> float:
    """mean(1 - |2h(V)-1|): 1 = all at 0.5, 0 = all binary. Paper: rise then fall."""
    total = 0.0
    n = 0
    for a in _as_list(alphas):
        h = adaround_h(a.detach())
        total += float((1.0 - (2.0 * h - 1.0).abs()).sum().item())
        n += h.numel()
    return total / max(n, 1)


class AdaRoundRegLoss(PTQLossBase):
    """Paper eq. 24: Σ (1 − |2 h(V) − 1|^β), h = rectified sigmoid."""

    def forward(self, alphas: Iterable[torch.Tensor], beta: float = 2.0, extra=None):
        total = None
        for a in _as_list(alphas):
            h = adaround_h(a)
            term = (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.tensor(0.0)
        return total

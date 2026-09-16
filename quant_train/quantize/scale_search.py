"""Activation clip search: 2DQuant DOBI, with percentile fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .config import SearchConfig
from .fake_quantize import UniformFakeQuantize


class ActSampleBuf:
    def __init__(self, max_samples: int = 50000):
        self.max_samples = int(max_samples)
        self._chunks: Dict[int, List[torch.Tensor]] = {}

    def add(self, fq: Optional[UniformFakeQuantize], x: torch.Tensor):
        if fq is None or not torch.is_tensor(x):
            return
        flat = x.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        if flat.numel() > self.max_samples:
            idx = torch.randint(0, flat.numel(), (self.max_samples,), device=flat.device)
            flat = flat[idx]
        chunks = self._chunks.setdefault(id(fq), [])
        chunks.append(flat.cpu())
        n = sum(c.numel() for c in chunks)
        if n > self.max_samples * 2:
            merged = torch.cat(chunks, dim=0)
            idx = torch.randint(0, merged.numel(), (self.max_samples,))
            self._chunks[id(fq)] = [merged[idx]]

    def get(self, fq: UniformFakeQuantize) -> Optional[torch.Tensor]:
        chunks = self._chunks.get(id(fq))
        if not chunks:
            return None
        x = torch.cat(chunks, dim=0)
        if x.numel() > self.max_samples:
            idx = torch.randint(0, x.numel(), (self.max_samples,))
            x = x[idx]
        return x


def _qdq(x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    scale = scale.clamp(min=1e-12)
    x_int = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    return (x_int - zp) * scale


def search_act_scale(
    x: torch.Tensor,
    percentiles,
    qmin: int = -128,
    qmax: int = 127,
    min_scale: float = 1e-9,
):
    x = x.reshape(-1).float()
    best_mse = torch.tensor(float("inf"))
    best_scale = torch.ones(1)
    best_zp = torch.zeros(1)
    best_p = 100.0
    levels = float(qmax - qmin)
    for p in percentiles:
        p = float(p)
        if p >= 100.0:
            lo, hi = x.min(), x.max()
        else:
            alpha = (100.0 - p) / 200.0
            lo = torch.quantile(x, alpha)
            hi = torch.quantile(x, 1.0 - alpha)
        if hi <= lo:
            hi = lo + min_scale
        scale = ((hi - lo) / levels).clamp(min=min_scale)
        zp = torch.round(qmin - lo / scale)
        xh = _qdq(x, scale, zp, qmin, qmax)
        mse = torch.mean((x - xh) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = scale.detach().reshape(1)
            best_zp = zp.detach().reshape(1)
            best_p = p
    return best_scale, best_zp, best_p, float(best_mse)


def classify_dist(x: torch.Tensor) -> str:
    """Bell-shaped vs exponential (2DQuant §3.2). One-sided / skewed → asymmetric."""
    lo = float(x.min())
    hi = float(x.max())
    if lo >= -1e-6 or hi <= 1e-6:
        return "asymmetric"
    frac_pos = float((x >= 0).float().mean())
    if frac_pos > 0.9 or frac_pos < 0.1:
        return "asymmetric"
    rng = max(hi - lo, 1e-12)
    mid = 0.5 * (lo + hi)
    if abs(float(x.mean()) - mid) / rng > 0.2:
        return "asymmetric"
    return "symmetric"


def search_act_dobi(
    x: torch.Tensor,
    k: int = 100,
    qmin: int = -128,
    qmax: int = 127,
    min_scale: float = 1e-9,
):
    """2DQuant Algorithm 1: inward MSE search; pin min for exponential tensors.
    Paper typesets u ← u + iΔu; that expands the grid — we narrow: u − iΔu.
    """
    x = x.reshape(-1).float()
    vmin, vmax = x.min(), x.max()
    if vmax <= vmin:
        vmax = vmin + min_scale
    rng = vmax - vmin
    kind = classify_dist(x)
    k = max(int(k), 1)
    delta_l = rng / (2.0 * k) if kind == "symmetric" else x.new_zeros(())
    delta_u = rng / (2.0 * k)
    levels = float(qmax - qmin)
    best_mse = torch.tensor(float("inf"))
    best = (vmin, vmax, torch.ones(1), torch.zeros(1))
    for i in range(k + 1):
        lo = vmin + i * delta_l
        hi = vmax - i * delta_u
        if hi <= lo:
            continue
        scale = ((hi - lo) / levels).clamp(min=min_scale)
        zp = torch.round(qmin - lo / scale)
        xh = _qdq(x, scale, zp, qmin, qmax)
        mse = torch.mean((x - xh) ** 2)
        if mse < best_mse:
            best_mse = mse
            best = (lo.detach(), hi.detach(), scale.detach().reshape(1), zp.detach().reshape(1))
    lo, hi, scale, zp = best
    return scale, zp, lo, hi, kind, float(best_mse)


def apply_act_scale_search(fq: UniformFakeQuantize, act_buf: ActSampleBuf, cfg: SearchConfig, *, method: str | None = None, percentiles=None) -> bool:
    samples = act_buf.get(fq)
    if samples is None:
        return False
    method = (method or getattr(cfg, "method", "dobi") or "dobi").lower()
    pcts = percentiles if percentiles else cfg.percentiles
    if method == "percentile":
        scale, zp, p, mse = search_act_scale(
            samples,
            pcts,
            qmin=fq.qmin,
            qmax=fq.qmax,
            min_scale=fq.min_scale,
        )
        fq.scale.copy_(scale.to(device=fq.scale.device, dtype=fq.scale.dtype).reshape_as(fq.scale))
        fq.zero_point.copy_(zp.to(device=fq.zero_point.device, dtype=fq.zero_point.dtype).reshape_as(fq.zero_point))
        lo = (fq.qmin - zp) * scale
        hi = (fq.qmax - zp) * scale
        fq.set_clip_bounds(lo, hi)
        fq._searched_percentile = p
        fq._dobi_kind = "percentile"
        fq._searched_mse = mse
        return True
    k = int(getattr(cfg, "dobi_k", 100))
    scale, zp, lo, hi, kind, mse = search_act_dobi(
        samples,
        k=k,
        qmin=fq.qmin,
        qmax=fq.qmax,
        min_scale=fq.min_scale,
    )
    fq.set_clip_bounds(lo, hi)
    fq._dobi_kind = kind
    fq._searched_mse = mse
    fq._searched_percentile = None
    return True

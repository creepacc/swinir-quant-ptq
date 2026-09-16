"""HarmoQ Structural Residual Calibration (SRC).

Closed-form weight correction so W absorbs high-frequency activation
quant error. Inference is always (W+δW) x, so the regressor is the same x:

    δW* = −W E[δx_s xᵀ] (E[xxᵀ] + λI)⁻¹

δx_s is the Laplacian residual H(xq)−H(x) when enabled, otherwise xq−x.
H is not applied to the Gram of x (that would solve δW (Hx) while the
layer still multiplies x). Applied to Linear / grouped-1 Conv.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantize.ops.base import iter_quant_ops, set_fakequant
from ..quantize.ops.conv import QuantConv2d
from ..quantize.ops.linear import QuantLinear
from ..utils.progress import progress


_LAP_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _laplacian_nchw(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    k = _LAP_KERNEL.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=c)


def _structuralize_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, L, C] → [B*L, C], Laplacian on square spatial grid when possible."""
    if x.ndim != 3:
        return x.reshape(-1, x.shape[-1])
    b, length, c = x.shape
    side = int(length ** 0.5)
    if side * side != length or side < 3:
        return x.reshape(-1, c)
    spatial = x.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
    return _laplacian_nchw(spatial).permute(0, 2, 3, 1).reshape(-1, c)


def _moments_linear(x: torch.Tensor, xq: torch.Tensor, use_lap: bool):
    xs = x.reshape(-1, x.shape[-1])
    if use_lap:
        dx = _structuralize_tokens(xq) - _structuralize_tokens(x)
    else:
        dx = xq.reshape(-1, xq.shape[-1]) - xs
    n = xs.shape[0]
    return xs.T @ xs, dx.T @ xs, n


def _moments_conv(x: torch.Tensor, xq: torch.Tensor, conv: nn.Conv2d, use_lap: bool):
    cols = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
    xs = cols.transpose(1, 2).reshape(-1, cols.shape[1])
    if use_lap and min(x.shape[-2], x.shape[-1]) >= 3:
        dmap = _laplacian_nchw(xq) - _laplacian_nchw(x)
        dcols = F.unfold(dmap, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
        dx = dcols.transpose(1, 2).reshape(-1, dcols.shape[1])
    else:
        cols_q = F.unfold(xq, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
        dx = cols_q.transpose(1, 2).reshape(-1, cols_q.shape[1]) - xs
    n = xs.shape[0]
    return xs.T @ xs, dx.T @ xs, n


class _MomAcc:
    def __init__(self, dim: int, device, dtype):
        self.xxt = torch.zeros(dim, dim, device=device, dtype=torch.float64)
        self.dxx = torch.zeros(dim, dim, device=device, dtype=torch.float64)
        self.n = 0

    def add(self, xxt, dxx, n):
        self.xxt += xxt.double()
        self.dxx += dxx.double()
        self.n += int(n)


def _src_targets(model: nn.Module):
    targets = []
    for m in iter_quant_ops(model):
        if isinstance(m, QuantLinear):
            targets.append(m)
        elif isinstance(m, QuantConv2d) and m.float_op.groups == 1:
            targets.append(m)
    return targets


def _collect_moments(model, loader, device, targets, use_laplacian: bool) -> Dict[int, _MomAcc]:
    store: Dict[int, torch.Tensor] = {}
    handles = [m.register_forward_pre_hook(lambda mod, inp, s=store: s.__setitem__(id(mod), inp[0].detach()))
               for m in targets]
    acc: Dict[int, _MomAcc] = {}
    try:
        for batch in loader:
            store.clear()
            model(batch["lr"].to(device))
            for m in targets:
                x = store.get(id(m))
                if x is None:
                    continue
                if m.input_fq is None or not m.quantize_input or not m.input_fq.fake_quant_enabled:
                    continue
                xq = m.input_fq(x)
                if isinstance(m, QuantLinear):
                    xxt, dxx, n = _moments_linear(x, xq, use_laplacian)
                else:
                    xxt, dxx, n = _moments_conv(x, xq, m.float_op, use_laplacian)
                slot = acc.get(id(m))
                if slot is None:
                    slot = _MomAcc(xxt.shape[0], xxt.device, xxt.dtype)
                    acc[id(m)] = slot
                slot.add(xxt, dxx, n)
    finally:
        for h in handles:
            h.remove()
    return acc


def _apply_delta(m, slot: _MomAcc, reg: float) -> float | None:
    if slot.n < dim_safe(slot):
        return None
    xxt = slot.xxt / max(slot.n, 1)
    dxx = slot.dxx / max(slot.n, 1)
    ridge = reg * float(xxt.diag().mean().clamp(min=1e-12))
    eye = torch.eye(xxt.shape[0], device=xxt.device, dtype=xxt.dtype)
    try:
        inv = torch.linalg.inv(xxt + ridge * eye)
    except RuntimeError:
        inv = torch.linalg.pinv(xxt + ridge * eye)
    w = m.float_op.weight.detach()
    if isinstance(m, QuantLinear):
        dw = -(w.double() @ dxx @ inv).to(dtype=w.dtype)
        if dw.shape != w.shape:
            return None
        m.float_op.weight.copy_(w + dw)
    else:
        w_flat = w.reshape(w.shape[0], -1)
        dw = -(w_flat.double() @ dxx @ inv).to(dtype=w.dtype).reshape_as(w)
        m.float_op.weight.copy_(w + dw)
    _refresh_weight_scales([m])
    return float(dw.float().norm().item())


@torch.no_grad()
def apply_src(
    model: nn.Module,
    loader,
    device: str,
    reg: float = 1e-3,
    use_laplacian: bool = True,
    grain: str = "layer",
) -> Dict[str, float]:
    """Rewrite float_op.weight. grain=layer updates one Linear/Conv then re-forwards."""
    model.eval()
    set_fakequant(model, True)
    targets = _src_targets(model)
    if not targets:
        print("[src] no Linear/Conv targets, skip")
        return {}
    grain = (grain or "layer").lower()
    stats: Dict[str, float] = {}
    print(f"[src] grain={grain} targets={len(targets)} λ={reg} laplacian={use_laplacian}")

    if grain != "layer":
        acc = _collect_moments(model, progress(loader, desc="src collect", total=len(loader)),
                               device, targets, use_laplacian)
        for m in targets:
            slot = acc.get(id(m))
            if slot is None:
                continue
            nrm = _apply_delta(m, slot, reg)
            if nrm is None:
                continue
            stats[_module_name(model, m)] = nrm
    else:
        for i, m in enumerate(progress(targets, desc="src layer", total=len(targets))):
            acc = _collect_moments(model, loader, device, [m], use_laplacian)
            slot = acc.get(id(m))
            if slot is None:
                continue
            nrm = _apply_delta(m, slot, reg)
            if nrm is None:
                continue
            stats[_module_name(model, m)] = nrm

    print(f"[src] calibrated {len(stats)}/{len(targets)} layers, grain={grain}")
    if stats:
        norms = list(stats.values())
        print(f"[src] ||δW||_F mean={sum(norms)/len(norms):.4g} max={max(norms):.4g}")
    return stats


def _refresh_weight_scales(targets) -> int:
    n = 0
    for m in targets:
        fq = getattr(m, "weight_fq", None)
        if fq is None or not hasattr(m, "float_op"):
            continue
        obs = fq.observer
        obs.min_val = torch.full_like(obs.min_val, float("inf"))
        obs.max_val = torch.full_like(obs.max_val, float("-inf"))
        obs.num_batches.zero_()
        fq.observe_tensor(m.float_op.weight.detach())
        fq.calculate_qparams()
        n += 1
    return n


def dim_safe(slot: _MomAcc) -> int:
    return slot.xxt.shape[0]


def _module_name(model: nn.Module, target: nn.Module) -> str:
    for n, m in model.named_modules():
        if m is target:
            return n
    return type(target).__name__

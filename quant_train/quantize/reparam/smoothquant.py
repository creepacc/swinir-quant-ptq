"""SmoothQuant + OS+ on the residual stream (diagonal s, still per-tensor INT8)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..ops.base import set_fakequant, set_observer
from ..ops.elementwise import QuantAdd
from .ops import (
    conv_op,
    div_linear_out,
    embed_dim,
    linear_op,
    ln_op,
    mul_linear_in,
    scale_conv_in,
    scale_conv_out,
    scale_conv_sandwich,
    set_ln_restore,
    skip_inner,
)


def _ch_absmax(x: torch.Tensor, ch: int) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] == ch:
        return x.detach().float().abs().amax(dim=(0, 2, 3))
    if x.shape[-1] == ch:
        dims = tuple(range(x.ndim - 1))
        return x.detach().float().abs().amax(dim=dims)
    return x.detach().float().abs().reshape(-1, ch).amax(dim=0)


@torch.no_grad()
def collect_channel_stats(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, torch.Tensor]:
    c = embed_dim(model)
    set_fakequant(model, False)
    set_observer(model, False)
    acc: Dict[str, torch.Tensor] = {}

    def _acc(key: str, vec: torch.Tensor):
        vec = vec.detach().float().cpu()
        if key not in acc:
            acc[key] = vec
        else:
            acc[key] = torch.maximum(acc[key], vec)

    handles = []
    for name, m in model.named_modules():
        if skip_inner(name):
            continue
        if isinstance(m, QuantAdd) and getattr(m, "is_residual", False):

            def on_add(_mod, _inp, out, n=name):
                _acc(n, _ch_absmax(out, c))

            handles.append(m.register_forward_hook(on_add))
            continue
        lin = linear_op(m)
        if lin is not None and lin.in_features == c:

            def on_lin(_mod, inp, _out, n=name):
                if inp and torch.is_tensor(inp[0]):
                    _acc("in:" + n, _ch_absmax(inp[0], c))

            handles.append(m.register_forward_hook(on_lin))
            continue
        conv = conv_op(m)
        if conv is not None and conv.in_channels == c:

            def on_conv(_mod, inp, _out, n=name):
                if inp and torch.is_tensor(inp[0]):
                    _acc("in:" + n, _ch_absmax(inp[0], c))

            handles.append(m.register_forward_hook(on_conv))

    model.eval()
    n = 0
    for batch in loader:
        model(batch["lr"].to(device))
        n += 1
    for h in handles:
        h.remove()
    print(f"[reparam] collected channel stats from {n} batches, {len(acc)} tensors")
    return acc


def _residual_s(stats: Dict[str, torch.Tensor], c: int, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    vecs = [v for k, v in stats.items() if not k.startswith("in:")]
    if not vecs:
        vecs = [v for k, v in stats.items() if k.startswith("in:")]
    xmax = torch.stack(vecs).amax(dim=0).clamp(min=eps)
    s = xmax.pow(alpha)
    s = s / s.mean().clamp(min=eps)
    s = s.clamp(min=eps)
    if s.numel() != c:
        raise RuntimeError(f"smooth s dim {s.numel()} != embed {c}")
    return s


@torch.no_grad()
def fuse_smoothquant(model: nn.Module, stats: Dict[str, torch.Tensor], alpha: float = 0.5) -> torch.Tensor:
    c = embed_dim(model)
    s = _residual_s(stats, c, alpha)
    n_lin_out = n_conv = n_ln = 0
    for name, m in model.named_modules():
        if skip_inner(name):
            continue
        lin = linear_op(m)
        if lin is not None:
            if lin.out_features == c:
                div_linear_out(lin, s)
                n_lin_out += 1
            continue
        conv = conv_op(m)
        if conv is not None:
            cin, co = conv.in_channels, conv.out_channels
            if cin == c and co == c:
                scale_conv_sandwich(conv, s)
                n_conv += 1
            elif cin != c and co == c:
                scale_conv_out(conv, s)
                n_conv += 1
            elif cin == c and co != c:
                scale_conv_in(conv, s)
                n_conv += 1
            continue
        ln = ln_op(m)
        if ln is not None:
            set_ln_restore(ln, s)
            reapply = name == "norm" or name.endswith("patch_embed.norm") or name.endswith("patch_unembed.norm")
            ln.channel_restore_reapply = reapply
            n_ln += 1
    print(f"[reparam] smoothquant alpha={alpha:.2f} lin_out={n_lin_out} conv={n_conv} ln={n_ln}")
    return s


@torch.no_grad()
def fuse_osplus_post_ln(model: nn.Module, stats: Dict[str, torch.Tensor], alpha: float = 0.5, eps: float = 1e-6):
    """Migrate post-LN outliers into qkv / mlp.fc1 (OS+ into LN affine)."""
    n = 0
    named = dict(model.named_modules())
    for name, m in named.items():
        if skip_inner(name):
            continue
        lin = linear_op(m)
        if lin is None:
            continue
        if not (name.endswith(".attn.qkv") or name.endswith(".mlp.fc1")):
            continue
        key = "in:" + name
        if key not in stats:
            continue
        xmax = stats[key].to(device=lin.weight.device, dtype=lin.weight.dtype).clamp(min=eps)
        w = lin.weight.detach().float().abs().amax(dim=0).clamp(min=eps)
        s = xmax.pow(alpha) / w.pow(1.0 - alpha)
        s = (s / s.mean().clamp(min=eps)).clamp(min=eps)
        mul_linear_in(lin, s)
        ln_name = name[: -len(".attn.qkv")] + ".norm1" if name.endswith(".attn.qkv") else name[: -len(".mlp.fc1")] + ".norm2"
        ln_m = named.get(ln_name)
        ln = ln_op(ln_m) if ln_m is not None else None
        if ln is not None:
            core = ln.float_op if hasattr(ln, "float_op") else ln
            ss = s.to(device=core.weight.device, dtype=core.weight.dtype)
            core.weight.data.div_(ss)
            if core.bias is not None:
                core.bias.data.div_(ss)
        n += 1
    print(f"[reparam] os+ folded into {n} post-LN linears")

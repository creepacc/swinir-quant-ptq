"""Fuse an orthogonal residual-stream rotation (QuaRot/SpinQuant R1, no Cayley train)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import (
    conv_op,
    embed_dim,
    linear_op,
    ln_op,
    rotate_conv_in,
    rotate_conv_out,
    rotate_conv_sandwich,
    rotate_linear_out,
    set_ln_Q,
    skip_inner,
)


def make_orthogonal(dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    a = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    # polar cleanup
    u, _, vh = torch.linalg.svd(q, full_matrices=False)
    q = u @ vh
    return q.to(device=device, dtype=dtype)


@torch.no_grad()
def fuse_rotation(model: nn.Module, seed: int = 0) -> torch.Tensor:
    c = embed_dim(model)
    ref = next(model.parameters())
    q = make_orthogonal(c, seed, ref.device, ref.dtype)
    n_lin = n_conv = n_ln = 0
    for name, m in model.named_modules():
        if skip_inner(name):
            continue
        lin = linear_op(m)
        if lin is not None:
            if lin.out_features == c and lin.in_features != c:
                rotate_linear_out(lin, q)
                n_lin += 1
            elif lin.in_features == c and lin.out_features == c and name.endswith(".proj"):
                rotate_linear_out(lin, q)
                n_lin += 1
            continue
        conv = conv_op(m)
        if conv is not None:
            cin, co = conv.in_channels, conv.out_channels
            if cin == c and co == c:
                rotate_conv_sandwich(conv, q)
                n_conv += 1
            elif cin != c and co == c:
                rotate_conv_out(conv, q)
                n_conv += 1
            elif cin == c and co != c:
                rotate_conv_in(conv, q)
                n_conv += 1
            continue
        ln = ln_op(m)
        if ln is not None:
            set_ln_Q(ln, q)
            reapply = name == "norm" or name.endswith("patch_embed.norm") or name.endswith("patch_unembed.norm")
            if isinstance(ln, nn.Module):
                ln.stream_Q_reapply = reapply
            n_ln += 1
    print(f"[reparam] quarot fused Q={c}x{c} seed={seed} linear={n_lin} conv={n_conv} ln={n_ln}")
    return q

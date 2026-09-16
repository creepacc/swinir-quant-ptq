from __future__ import annotations

import torch
import torch.nn as nn

from ..ops.conv import QuantConv2d
from ..ops.linear import QuantLinear
from ..ops.norm import QuantLayerNorm


def linear_op(m: nn.Module) -> nn.Linear | None:
    if isinstance(m, QuantLinear):
        return m.float_op
    if isinstance(m, nn.Linear):
        return m
    return None


def conv_op(m: nn.Module) -> nn.Conv2d | None:
    if isinstance(m, QuantConv2d):
        return m.float_op
    if isinstance(m, nn.Conv2d):
        return m
    return None


def ln_op(m: nn.Module) -> nn.Module | None:
    if isinstance(m, QuantLayerNorm):
        return m
    if isinstance(m, nn.LayerNorm):
        return m
    return None


def embed_dim(model: nn.Module) -> int:
    return int(getattr(model, "embed_dim", 180))


def skip_inner(name: str) -> bool:
    return name.endswith("float_op")


@torch.no_grad()
def mul_linear_in(lin: nn.Linear, s: torch.Tensor):
    s = s.to(device=lin.weight.device, dtype=lin.weight.dtype).reshape(1, -1)
    lin.weight.data.mul_(s)


@torch.no_grad()
def div_linear_out(lin: nn.Linear, s: torch.Tensor):
    s = s.to(device=lin.weight.device, dtype=lin.weight.dtype).reshape(-1, 1)
    lin.weight.data.div_(s)
    if lin.bias is not None:
        lin.bias.data.div_(s.reshape(-1))


@torch.no_grad()
def rotate_linear_in(lin: nn.Linear, q: torch.Tensor):
    q = q.to(device=lin.weight.device, dtype=lin.weight.dtype)
    lin.weight.data.copy_(lin.weight.data @ q)


@torch.no_grad()
def rotate_linear_out(lin: nn.Linear, q: torch.Tensor):
    q = q.to(device=lin.weight.device, dtype=lin.weight.dtype)
    lin.weight.data.copy_(q.transpose(0, 1) @ lin.weight.data)
    if lin.bias is not None:
        lin.bias.data.copy_(q.transpose(0, 1) @ lin.bias.data)


@torch.no_grad()
def rotate_conv_out(conv: nn.Conv2d, q: torch.Tensor):
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    w = conv.weight.data
    co = w.shape[0]
    conv.weight.data.copy_((q.transpose(0, 1) @ w.reshape(co, -1)).reshape_as(w))
    if conv.bias is not None:
        conv.bias.data.copy_(q.transpose(0, 1) @ conv.bias.data)


@torch.no_grad()
def rotate_conv_in(conv: nn.Conv2d, q: torch.Tensor):
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    w = conv.weight.data.permute(0, 2, 3, 1) @ q
    conv.weight.data.copy_(w.permute(0, 3, 1, 2).contiguous())


@torch.no_grad()
def rotate_conv_sandwich(conv: nn.Conv2d, q: torch.Tensor):
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    w = conv.weight.data
    co, cin, kh, kw = w.shape
    w = w.permute(2, 3, 0, 1).reshape(-1, co, cin)
    w = torch.einsum("ij,njk,kl->nil", q.transpose(0, 1), w, q)
    conv.weight.data.copy_(w.reshape(kh, kw, co, cin).permute(2, 3, 0, 1).contiguous())
    if conv.bias is not None:
        conv.bias.data.copy_(q.transpose(0, 1) @ conv.bias.data)


@torch.no_grad()
def scale_conv_sandwich(conv: nn.Conv2d, s: torch.Tensor):
    s = s.to(device=conv.weight.device, dtype=conv.weight.dtype).reshape(-1)
    w = conv.weight.data
    w.mul_(s.reshape(1, -1, 1, 1))
    w.div_(s.reshape(-1, 1, 1, 1))
    if conv.bias is not None:
        conv.bias.data.div_(s)


@torch.no_grad()
def scale_conv_out(conv: nn.Conv2d, s: torch.Tensor):
    s = s.to(device=conv.weight.device, dtype=conv.weight.dtype).reshape(-1, 1, 1, 1)
    conv.weight.data.div_(s)
    if conv.bias is not None:
        conv.bias.data.div_(s.reshape(-1))


@torch.no_grad()
def scale_conv_in(conv: nn.Conv2d, s: torch.Tensor):
    s = s.to(device=conv.weight.device, dtype=conv.weight.dtype).reshape(1, -1, 1, 1)
    conv.weight.data.mul_(s)


def set_ln_restore(m: nn.Module, s: torch.Tensor):
    s = s.detach().reshape(-1)
    if isinstance(m, QuantLayerNorm):
        m.register_buffer("channel_restore", s.to(device=m.float_op.weight.device, dtype=m.float_op.weight.dtype).clone())
        return
    if isinstance(m, nn.LayerNorm) and m.weight is not None:
        m.register_buffer("channel_restore", s.to(device=m.weight.device, dtype=m.weight.dtype).clone())


def set_ln_Q(m: nn.Module, q: torch.Tensor):
    q = q.detach()
    if isinstance(m, QuantLayerNorm):
        m.register_buffer("stream_Q", q.to(device=m.float_op.weight.device, dtype=m.float_op.weight.dtype).clone())
        return
    if isinstance(m, nn.LayerNorm):
        m.register_buffer("stream_Q", q.to(device=m.weight.device, dtype=m.weight.dtype).clone())

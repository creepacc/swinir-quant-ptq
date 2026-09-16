"""Mixins that override + / * / @. Combined with original SwinIR classes at convert time."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import QuantConfig
from ..ops.elementwise import QuantAdd, QuantMul
from ..ops.linear import QuantMatMul


def adopt(module: nn.Module, cls) -> nn.Module:
    module.__class__ = cls
    return module


class QuantWindowAttentionMixin:
    quant_block_kind = "window_attn"

    def _init_quant_ops(self, cfg: QuantConfig):
        skip_attn = bool(getattr(cfg, "skip_attn_act", False))
        skip_sm_io = skip_attn or getattr(cfg, "softmax_mode", "default") in ("fp32", "lut")
        self.q_scale_mul = QuantMul(cfg, quantize_output=not skip_attn)
        self.qk_matmul = QuantMatMul(
            cfg,
            quantize_a=not skip_attn,
            quantize_b=not skip_attn,
            quantize_output=not skip_attn,
        )
        self.bias_add = QuantAdd(cfg, quantize_output=not skip_sm_io)
        self.mask_add = QuantAdd(cfg, quantize_output=not skip_sm_io)
        self.av_matmul = QuantMatMul(
            cfg,
            quantize_a=not skip_sm_io,
            quantize_b=not skip_attn,
            quantize_output=not skip_attn,
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_scale_mul(q, self.scale)
        attn = self.qk_matmul(q, k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = self.bias_add(attn, relative_position_bias.unsqueeze(0))
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = self.mask_add(attn, mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = self.av_matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantSwinTransformerBlockMixin:
    quant_block_kind = "swin_block"

    def _init_quant_ops(self, cfg: QuantConfig):
        qout = not bool(getattr(cfg, "skip_residual_act", False))
        self.attn_res_add = QuantAdd(cfg, quantize_output=qout, residual=True)
        self.mlp_res_add = QuantAdd(cfg, quantize_output=qout, residual=True)

    def forward(self, x, x_size):
        from swinir.network_swinir import window_partition, window_reverse

        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = self.attn_res_add(shortcut, self.drop_path(x))
        x = self.mlp_res_add(x, self.drop_path(self.mlp(self.norm2(x))))
        return x


class QuantRSTBMixin:
    quant_block_kind = "rstb"

    def _init_quant_ops(self, cfg: QuantConfig):
        qout = not bool(getattr(cfg, "skip_residual_act", False))
        self.body_res_add = QuantAdd(cfg, quantize_output=qout, residual=True)

    def forward(self, x, x_size):
        y = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size)))
        return self.body_res_add(y, x)


class QuantSwinIRMixin:
    quant_block_kind = "network"

    def _init_quant_ops(self, cfg: QuantConfig):
        qout = not bool(getattr(cfg, "skip_residual_act", False))
        self.skip_add = QuantAdd(cfg, quantize_output=qout, residual=True)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.skip_add(self.conv_after_body(self.forward_features(x)), x)
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.skip_add(self.conv_after_body(self.forward_features(x)), x)
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.skip_add(self.conv_after_body(self.forward_features(x)), x)
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.skip_add(self.conv_after_body(self.forward_features(x_first)), x_first)
            x = x + self.conv_last(res)
        x = x / self.img_range + mean
        return x[:, :, : H * self.upscale, : W * self.upscale]

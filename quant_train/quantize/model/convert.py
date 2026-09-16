from __future__ import annotations

import torch.nn as nn

from ..config import AppConfig, QuantConfig
from ..ops.activation import QuantGELU, QuantLeakyReLU, QuantReLU, QuantSiLU, QuantSoftmax
from ..ops.base import QuantOpBase
from ..ops.conv import QuantConv2d
from ..ops.linear import QuantLinear
from ..ops.norm import QuantLayerNorm
from .quant_blocks import (
    QuantRSTBMixin,
    QuantSwinIRMixin,
    QuantSwinTransformerBlockMixin,
    QuantWindowAttentionMixin,
    adopt,
)


def _leaf_map():
    return {
        nn.Linear: QuantLinear,
        nn.Conv2d: QuantConv2d,
        nn.GELU: QuantGELU,
        nn.Softmax: QuantSoftmax,
        nn.LeakyReLU: QuantLeakyReLU,
        nn.ReLU: QuantReLU,
        nn.SiLU: QuantSiLU,
        nn.LayerNorm: QuantLayerNorm,
    }


def _replace_leaves(module: nn.Module, cfg: QuantConfig, adaround: bool):
    mapping = _leaf_map()
    for name, child in list(module.named_children()):
        if isinstance(child, QuantOpBase):
            continue
        cls = mapping.get(type(child))
        if cls is not None:
            if cls in (QuantLinear, QuantConv2d):
                setattr(module, name, cls.from_float(child, cfg, adaround=adaround))
            else:
                setattr(module, name, cls.from_float(child, cfg))
        else:
            _replace_leaves(child, cfg, adaround)


def _quant_cls(src, mixin):
    return type("Quant" + src.__name__, (mixin, src), {})


def _patch_convdts(module: nn.Module):
    import torch.nn.functional as F
    from swinir.network_swinir import ConvDTS

    for m in module.modules():
        if isinstance(m, ConvDTS):
            m.forward = lambda x, self=m: F.pixel_shuffle(self.conv(x), self.scale)


def _wrap_composites(module: nn.Module, cfg: QuantConfig):
    from swinir.network_swinir import RSTB, SwinIR, SwinTransformerBlock, WindowAttention

    pairs = (
        (WindowAttention, QuantWindowAttentionMixin),
        (SwinTransformerBlock, QuantSwinTransformerBlockMixin),
        (RSTB, QuantRSTBMixin),
        (SwinIR, QuantSwinIRMixin),
    )
    for name, child in list(module.named_children()):
        _wrap_composites(child, cfg)
        for src, mixin in pairs:
            if type(child) is src:
                adopt(child, _quant_cls(src, mixin))
                if hasattr(child, "_init_quant_ops"):
                    child._init_quant_ops(cfg)
                setattr(module, name, child)
                break


def summarize_fakequants(model: nn.Module) -> dict:
    """Count inserted fake-quant tensors. Linear/Conv output FQ is off by default."""
    n_w = n_in = n_out = 0
    by_kind = {}
    for m in model.modules():
        if not isinstance(m, QuantOpBase):
            continue
        kind = getattr(m, "op_kind", type(m).__name__)
        bucket = by_kind.setdefault(kind, {"weight": 0, "input": 0, "output": 0})
        if m.weight_fq is not None:
            n_w += 1
            bucket["weight"] += 1
        if m.input_fq is not None:
            n_in += 1
            bucket["input"] += 1
        if m.output_fq is not None:
            n_out += 1
            bucket["output"] += 1
    print(f"[prepare] fakequant tensors: weight={n_w} input={n_in} output={n_out}")
    for kind in sorted(by_kind):
        c = by_kind[kind]
        if c["weight"] or c["input"] or c["output"]:
            print(f"  {kind:12s}  W={c['weight']:3d}  Ain={c['input']:3d}  Aout={c['output']:3d}")
    return {"weight": n_w, "input": n_in, "output": n_out, "by_kind": by_kind}


def prepare_swinir(model: nn.Module, cfg: AppConfig) -> nn.Module:
    """In-place wrap: leaves first, then composite classes."""
    qcfg, adaround = cfg.quant, cfg.train.adaround
    _replace_leaves(model, qcfg, adaround)
    _patch_convdts(model)
    _wrap_composites(model, qcfg)
    from swinir.network_swinir import SwinIR

    if type(model) is SwinIR:
        adopt(model, _quant_cls(SwinIR, QuantSwinIRMixin))
        model._init_quant_ops(qcfg)
    summarize_fakequants(model)
    return model

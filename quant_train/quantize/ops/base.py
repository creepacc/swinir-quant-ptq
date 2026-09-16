from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn

from ..config import QuantConfig
from ..fake_quantize import UniformFakeQuantize, make_fakequant


class QuantOpBase(nn.Module):
    """Common switches + optional input / weight / output fake-quant."""

    op_kind: str = "generic"

    def __init__(
        self,
        cfg: QuantConfig,
        quantize_weight: bool = False,
        quantize_input: bool = False,
        quantize_output: bool = False,
        weight_ch_axis: int = 0,
        adaround: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.quantize_weight = quantize_weight
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.input_fq: Optional[UniformFakeQuantize] = None
        self.weight_fq: Optional[UniformFakeQuantize] = None
        self.output_fq: Optional[UniformFakeQuantize] = None
        if quantize_input:
            self.input_fq = make_fakequant(cfg, ch_axis=-1, adaround=False, kind="act")
        if quantize_weight:
            ch = weight_ch_axis if cfg.per_channel_weight else -1
            self.weight_fq = make_fakequant(cfg, ch_axis=ch, adaround=adaround, kind="weight")
        if quantize_output:
            self.output_fq = make_fakequant(cfg, ch_axis=-1, adaround=False, kind="act")

    def _qin(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_input and self.input_fq is not None:
            return self.input_fq(x)
        return x

    def _qw(self, w: torch.Tensor) -> torch.Tensor:
        if self.quantize_weight and self.weight_fq is not None:
            return self.weight_fq(w)
        return w

    def _qout(self, y: torch.Tensor) -> torch.Tensor:
        if self.quantize_output and self.output_fq is not None:
            return self.output_fq(y)
        return y

    def iter_fqs(self) -> Iterator[UniformFakeQuantize]:
        for fq in (self.input_fq, self.weight_fq, self.output_fq):
            if fq is not None:
                yield fq


def iter_quant_ops(model: nn.Module) -> Iterator[QuantOpBase]:
    for m in model.modules():
        if isinstance(m, QuantOpBase):
            yield m


def set_observer(model: nn.Module, enabled: bool):
    for m in iter_quant_ops(model):
        for fq in m.iter_fqs():
            fq.enable_observer(enabled)


def set_fakequant(model: nn.Module, enabled: bool):
    for m in iter_quant_ops(model):
        for fq in m.iter_fqs():
            fq.enable_fakequant(enabled)


def freeze_all(
    model: nn.Module,
    adaround: bool = False,
    search_cfg=None,
    act_buf=None,
    init_adaround: bool = False,
):
    from collections import Counter

    from ..scale_search import apply_act_scale_search

    searched = 0
    picked = Counter()
    for m in iter_quant_ops(model):
        if m.weight_fq is not None:
            w = None
            if hasattr(m, "float_op") and hasattr(m.float_op, "weight"):
                w = m.float_op.weight
            if w is not None:
                m.weight_fq.observe_tensor(w)
            m.weight_fq.calculate_qparams()
            if init_adaround and adaround and w is not None:
                m.weight_fq.init_alpha(w)
        for fq in (m.input_fq, m.output_fq):
            if fq is None:
                continue
            if search_cfg is not None and getattr(search_cfg, "enable", False) and act_buf is not None:
                method = None
                pcts = None
                if getattr(m, "is_residual", False):
                    rp = tuple(getattr(search_cfg, "residual_percentiles", ()) or ())
                    rm = (getattr(search_cfg, "residual_method", "") or "").lower()
                    if rp:
                        pcts = rp
                        method = rm or "percentile"
                    elif rm:
                        method = rm
                if apply_act_scale_search(fq, act_buf, search_cfg, method=method, percentiles=pcts):
                    searched += 1
                    picked[getattr(fq, "_dobi_kind", getattr(fq, "_searched_percentile", None))] += 1
                    continue
            fq.calculate_qparams()
            fq.set_clip_bounds(*fq._bounds_from_qparams())
        for fq in m.iter_fqs():
            fq.enable_observer(False)
            fq.enable_fakequant(True)
    if searched:
        hist = ", ".join(f"{k}:{n}" for k, n in sorted(picked.items(), key=lambda x: (x[0] is None, str(x[0]))))
        method = getattr(search_cfg, "method", "dobi") if search_cfg is not None else "minmax"
        print(f"[search] activation {method} applied on {searched} tensors ({hist})")


def freeze_act_bounds(model: nn.Module) -> int:
    n = 0
    for m in iter_quant_ops(model):
        for fq in (m.input_fq, m.output_fq):
            if fq is None:
                continue
            fq.freeze_bounds()
            n += 1
    return n


def init_adaround_all(model: nn.Module) -> int:
    n = 0
    for m in iter_quant_ops(model):
        if m.weight_fq is None:
            continue
        w = None
        if hasattr(m, "float_op") and hasattr(m.float_op, "weight"):
            w = m.float_op.weight
        if w is None:
            continue
        m.weight_fq.adaround = True
        m.weight_fq.init_alpha(w)
        n += 1
    print(f"[adaround] initialized alpha on {n} weight tensors")
    return n

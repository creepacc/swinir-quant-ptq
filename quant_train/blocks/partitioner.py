from __future__ import annotations

from typing import List

import torch.nn as nn

from ..quantize.ops.conv import QuantConv2d
from ..quantize.ops.linear import QuantLinear
from .unit import Part, ReconstructionUnit


class ModelPartitioner:
    def split(self, model: nn.Module) -> List[Part]:
        raise NotImplementedError


class SwinIRPartitioner(ModelPartitioner):
    def __init__(self, grain: str = "block", block_type: str = "swin_block"):
        self.grain = ReconstructionUnit(grain)
        self.block_type = block_type

    def split(self, model: nn.Module) -> List[Part]:
        if self.grain == ReconstructionUnit.NETWORK:
            return [Part(name="", kind="network", module=model)]
        if self.grain == ReconstructionUnit.LAYER:
            return self._split_layers(model)
        return self._split_blocks(model)

    def _split_layers(self, model: nn.Module) -> List[Part]:
        parts = []
        for name, m in model.named_modules():
            if isinstance(m, (QuantLinear, QuantConv2d)):
                parts.append(Part(name=name, kind="layer", module=m))
        return parts

    def _head_tail(self, model: nn.Module) -> tuple[List[Part], List[Part]]:
        head, tail = [], []
        if hasattr(model, "conv_first"):
            head.append(Part("conv_first", "head", model.conv_first))
        extras = []
        if hasattr(model, "conv_after_body"):
            extras.append(("conv_after_body", model.conv_after_body))
        if hasattr(model, "conv_before_upsample"):
            extras.append(("conv_before_upsample", model.conv_before_upsample))
        if hasattr(model, "upsample"):
            extras.append(("upsample", model.upsample))
        if hasattr(model, "conv_last"):
            extras.append(("conv_last", model.conv_last))
        if extras:
            # one tail part that still hooks the last conv; extras stored for completeness
            name, mod = extras[-1]
            extra = {n: m for n, m in extras[:-1]}
            tail.append(Part(name=name, kind="tail", module=mod, extra_modules=extra))
        return head, tail

    def _split_blocks(self, model: nn.Module) -> List[Part]:
        want = "swin_block" if self.block_type == "swin_block" else "rstb"
        body = []
        for name, m in model.named_modules():
            if getattr(m, "quant_block_kind", None) == want:
                body.append(Part(name=name, kind="block", module=m))
        head, tail = self._head_tail(model)
        return head + body + tail

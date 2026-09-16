from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import torch.nn as nn

from ..quantize.ops.base import QuantOpBase
from ..quantize.ops.conv import QuantConv2d
from ..quantize.ops.linear import QuantLinear


class ReconstructionUnit(str, Enum):
    LAYER = "layer"
    BLOCK = "block"
    NETWORK = "network"


@dataclass
class Part:
    name: str
    kind: str
    module: nn.Module
    extra_modules: Dict[str, nn.Module] = field(default_factory=dict)

    def named_quant_ops(self) -> List[tuple[str, QuantOpBase]]:
        found = []
        for n, m in self.module.named_modules():
            if isinstance(m, QuantOpBase):
                found.append((f"{self.name}.{n}" if n else self.name, m))
        for n, m in self.extra_modules.items():
            for cn, cm in m.named_modules():
                if isinstance(cm, QuantOpBase):
                    found.append((f"{n}.{cn}" if cn else n, cm))
        return found

    def named_weight_ops(self) -> List[tuple[str, nn.Module]]:
        ops = []
        for n, m in self.module.named_modules():
            if isinstance(m, (QuantLinear, QuantConv2d)):
                ops.append((n, m))
        return ops

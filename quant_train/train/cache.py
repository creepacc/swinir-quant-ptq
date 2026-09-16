from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class CalibCache:
    parts: Dict[str, List[dict]] = field(default_factory=dict)
    images: List[dict] = field(default_factory=list)

    def to(self, device: str):
        return self

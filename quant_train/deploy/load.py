from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ..quantize.ops.base import QuantOpBase


def _load_fq(fq, blob: dict):
    if fq is None or not blob:
        return
    scale = torch.tensor(blob["scale"], dtype=torch.float32)
    zp = torch.tensor(blob["zero_point"], dtype=torch.float32)
    fq.scale = scale.to(fq.scale.device)
    fq.zero_point = zp.to(fq.zero_point.device)
    if blob.get("adaround_alpha") is not None:
        alpha = torch.tensor(blob["adaround_alpha"], dtype=torch.float32)
        fq.alpha = nn.Parameter(alpha.to(fq.scale.device))
        fq.adaround = True
    fq.enable_observer(False)
    fq.enable_fakequant(True)


def load_quant_params(model: nn.Module, path: str) -> dict:
    path = Path(path)
    blob = json.loads(path.read_text(encoding="utf-8"))
    layers = blob.get("layers", {})
    named = {n: m for n, m in model.named_modules() if isinstance(m, QuantOpBase)}
    loaded, missing = 0, []
    for name, m in named.items():
        rec = layers.get(name)
        if rec is None:
            missing.append(name)
            continue
        _load_fq(m.weight_fq, rec.get("weight") or {})
        _load_fq(m.input_fq, rec.get("input") or {})
        _load_fq(m.output_fq, rec.get("output") or {})
        loaded += 1
    if missing:
        print(f"[deploy] warning: {len(missing)} modules have no saved params (first: {missing[:3]})")
    print(f"[deploy] loaded {loaded}/{len(named)} quant modules from {path}")
    return blob.get("meta", {})

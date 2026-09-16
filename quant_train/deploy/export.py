from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from ..quantize.ops.base import QuantOpBase


def _to_list(t: torch.Tensor):
    return t.detach().float().cpu().flatten().tolist()


def _dump_fq(fq) -> Dict[str, Any]:
    if fq is None:
        return {}
    out = {
        "scale": _to_list(fq.scale),
        "zero_point": _to_list(fq.zero_point),
        "qmin": fq.qmin,
        "qmax": fq.qmax,
        "ch_axis": fq.ch_axis,
        "symmetric": bool(getattr(fq.observer, "symmetric", True)),
    }
    if hasattr(fq, "_searched_percentile"):
        out["searched_percentile"] = fq._searched_percentile
        out["searched_mse"] = getattr(fq, "_searched_mse", None)
    return out


def export_quant_params(model: nn.Module, path: str, meta: Dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = {}
    for name, m in model.named_modules():
        if not isinstance(m, QuantOpBase):
            continue
        entry = {"op_kind": m.op_kind}
        if m.weight_fq is not None:
            entry["weight"] = _dump_fq(m.weight_fq)
        if m.input_fq is not None:
            entry["input"] = _dump_fq(m.input_fq)
        if m.output_fq is not None:
            entry["output"] = _dump_fq(m.output_fq)
        layers[name] = entry
    blob = {"meta": meta or {}, "layers": layers}
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"[deploy] exported {len(layers)} layers -> {path}")
    return path

"""Per-layer FP32 vs fake-quant stats on one chosen image."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantize.ops.base import QuantOpBase, set_fakequant


def _unravel(index: int, shape: torch.Size) -> str:
    if not shape:
        return "0"
    coords = []
    rem = int(index)
    for dim in reversed(shape):
        coords.append(str(rem % int(dim)))
        rem //= int(dim)
    return ",".join(reversed(coords))


def _pair_stats(fp: torch.Tensor, q: torch.Tensor) -> Dict[str, Any]:
    fp = fp.detach().float()
    q = q.detach().float()
    if fp.shape != q.shape:
        n = min(fp.numel(), q.numel())
        fp = fp.reshape(-1)[:n]
        q = q.reshape(-1)[:n]
    diff = (fp - q).abs()
    if diff.numel() == 0:
        return {
            "shape": "0",
            "max_abs_diff": 0.0,
            "max_diff_index": "",
            "fp_min": 0.0,
            "fp_max": 0.0,
            "q_min": 0.0,
            "q_max": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "cos_sim": 1.0,
        }
    flat_i = int(diff.reshape(-1).argmax().item())
    a = fp.reshape(1, -1)
    b = q.reshape(1, -1)
    cos = float(F.cosine_similarity(a, b, dim=1, eps=1e-8).item())
    return {
        "shape": "x".join(str(s) for s in fp.shape),
        "max_abs_diff": float(diff.max().item()),
        "max_diff_index": _unravel(flat_i, fp.shape),
        "fp_min": float(fp.min().item()),
        "fp_max": float(fp.max().item()),
        "q_min": float(q.min().item()),
        "q_max": float(q.max().item()),
        "mae": float(diff.mean().item()),
        "mse": float((diff ** 2).mean().item()),
        "cos_sim": cos,
    }


def _collect_named(model: nn.Module, x: torch.Tensor, fq_on: bool) -> Dict[str, torch.Tensor]:
    set_fakequant(model, fq_on)
    bags: Dict[str, torch.Tensor] = {}
    handles = []

    for name, m in model.named_modules():
        if not isinstance(m, QuantOpBase):
            continue

        def on_mod(mod, inp, out, n=name, op=m):
            if torch.is_tensor(out):
                bags[f"{n}.output"] = out.detach()
            if op.input_fq is None and inp and torch.is_tensor(inp[0]):
                bags[f"{n}.input"] = inp[0].detach()

        handles.append(m.register_forward_hook(on_mod))
        if m.input_fq is not None:

            def on_in(mod, inp, out, n=name):
                if torch.is_tensor(out):
                    bags[f"{n}.input"] = out.detach()

            handles.append(m.input_fq.register_forward_hook(on_in))

    model.eval()
    with torch.no_grad():
        model(x)
        for name, m in model.named_modules():
            if not isinstance(m, QuantOpBase) or m.weight_fq is None:
                continue
            if not hasattr(m, "float_op") or not hasattr(m.float_op, "weight"):
                continue
            w = m.float_op.weight
            bags[f"{name}.weight"] = m.weight_fq(w).detach() if fq_on else w.detach()

    for h in handles:
        h.remove()
    return bags


@torch.no_grad()
def collect_layer_diff_rows(qmodel: nn.Module, lr: torch.Tensor) -> List[Dict[str, Any]]:
    """Compare each QuantOp tensor with fakequant off vs on. Returns CSV rows."""
    was_fq = True
    fp_bags = _collect_named(qmodel, lr, False)
    q_bags = _collect_named(qmodel, lr, True)
    set_fakequant(qmodel, was_fq)
    keys = sorted(set(fp_bags) & set(q_bags))
    rows = []
    for key in keys:
        if key.endswith(".weight"):
            kind = "weight"
            layer = key[: -len(".weight")]
        elif key.endswith(".input"):
            kind = "input"
            layer = key[: -len(".input")]
        elif key.endswith(".output"):
            kind = "output"
            layer = key[: -len(".output")]
        else:
            kind = "tensor"
            layer = key
        st = _pair_stats(fp_bags[key], q_bags[key])
        rows.append({"layer": layer, "tensor": kind, **st})
    return rows


def write_layer_stats_csv(rows: List[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "tensor",
        "shape",
        "max_abs_diff",
        "max_diff_index",
        "fp_min",
        "fp_max",
        "q_min",
        "q_max",
        "mae",
        "mse",
        "cos_sim",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"[stats] layer diffs -> {path} ({len(rows)} rows)")
    return path


def export_layer_stats_csv(qmodel: nn.Module, lr: torch.Tensor, path: str | Path) -> Path:
    return write_layer_stats_csv(collect_layer_diff_rows(qmodel, lr), path)

"""Per-tensor histograms: filename = data name (e.g. conv_first.weight.png)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..quantize.ops.base import QuantOpBase, set_fakequant
from ..utils.hooks import FeatureHook

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_ERR = exc
else:
    _MPL_ERR = None


def _to_np(t: torch.Tensor, max_n: int = 250_000) -> np.ndarray:
    x = t.detach().float().reshape(-1)
    if x.numel() > max_n:
        idx = torch.randint(0, x.numel(), (max_n,), device=x.device)
        x = x[idx]
    return x.cpu().numpy()


def _stats(arr: np.ndarray) -> dict:
    return {
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "count": int(arr.size),
    }


def _save_one_hist(data: Optional[np.ndarray], path: Path, title: str, bins: int, color: str):
    if plt is None:
        raise RuntimeError(f"matplotlib is required to draw histograms: {_MPL_ERR}")
    if data is None or data.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(data, bins=bins, color=color, alpha=0.9)
    st = _stats(data)
    ax.set_title(f"{title}\nmin={st['min']:.4g}  max={st['max']:.4g}")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def snapshot_weights(model: nn.Module) -> Dict[str, np.ndarray]:
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, QuantOpBase):
            continue
        if hasattr(m, "float_op") and hasattr(m.float_op, "weight"):
            out[f"{name}.weight"] = _to_np(m.float_op.weight)
    return out


def _collect_acts(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    named = {n: m for n, m in model.named_modules() if isinstance(m, QuantOpBase)}
    hook = FeatureHook(named)
    model.eval()
    with hook, torch.no_grad():
        for batch in loader:
            model(batch["lr"].to(device))
            break
    dumped: Dict[str, np.ndarray] = {}
    for name, recs in hook.records.items():
        if not recs:
            continue
        rec = recs[0]
        ins = rec["input"]
        if ins and torch.is_tensor(ins[0]):
            dumped[f"{name}.input"] = _to_np(ins[0])
        if len(ins) > 1 and torch.is_tensor(ins[1]):
            dumped[f"{name}.input1"] = _to_np(ins[1])
        out = rec["output"]
        if torch.is_tensor(out):
            dumped[f"{name}.output"] = _to_np(out)
    hook.close()
    return dumped


@torch.no_grad()
def export_histograms(
    model: nn.Module,
    loader: DataLoader,
    out_dir: str,
    device: str,
    weight_fp32: Dict[str, np.ndarray],
    bins: int = 64,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_q = snapshot_weights(model)
    set_fakequant(model, False)
    act_fp32 = _collect_acts(model, loader, device)
    set_fakequant(model, True)
    act_q = _collect_acts(model, loader, device)

    names = sorted(set(weight_fp32) | set(weight_q) | set(act_fp32) | set(act_q))
    n = 0
    for name in names:
        fp = weight_fp32.get(name) if name.endswith(".weight") else act_fp32.get(name)
        qv = weight_q.get(name) if name.endswith(".weight") else act_q.get(name)
        if _save_one_hist(fp, out_dir / f"{name}.fp32.png", f"{name} fp32", bins, "steelblue"):
            n += 1
        if _save_one_hist(qv, out_dir / f"{name}.quant.png", f"{name} quant", bins, "coral"):
            n += 1
    print(f"[ranges] wrote {n} histogram images -> {out_dir}")
    set_fakequant(model, True)
    return n

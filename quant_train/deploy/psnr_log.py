"""Callable PSNR/SSIM dump: per-image rows + mean row."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..quantize.ops.base import set_fakequant
from ..utils.metrics import psnr_quant_vs_fp32, psnr_vs_hr, ssim_quant_vs_fp32, ssim_vs_hr
from ..utils.progress import progress

_FIELDS = [
    "name",
    "q_vs_fp32",
    "q_vs_hr",
    "fp32_vs_hr",
    "q_vs_fp32_ssim",
    "q_vs_hr_ssim",
    "fp32_vs_hr_ssim",
]


@torch.no_grad()
def collect_psnr_rows(
    qmodel: nn.Module,
    fp32_model: nn.Module,
    loader: DataLoader,
    device: str,
) -> List[dict]:
    qmodel.eval()
    fp32_model.eval()
    set_fakequant(qmodel, True)
    rows = []
    for batch in progress(loader, desc="stats psnr/ssim", total=len(loader)):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        q_sr = qmodel(lr)
        fp_sr = fp32_model(lr)
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else batch["name"]
        rows.append(
            {
                "name": str(name),
                "q_vs_fp32": psnr_quant_vs_fp32(q_sr, fp_sr),
                "q_vs_hr": psnr_vs_hr(q_sr, hr),
                "fp32_vs_hr": psnr_vs_hr(fp_sr, hr),
                "q_vs_fp32_ssim": ssim_quant_vs_fp32(q_sr, fp_sr),
                "q_vs_hr_ssim": ssim_vs_hr(q_sr, hr),
                "fp32_vs_hr_ssim": ssim_vs_hr(fp_sr, hr),
            }
        )
    return rows


def write_psnr_csv(rows: List[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.6f}" if k != "name" else row[k]) for k in _FIELDS})
        if rows:
            keys = [k for k in _FIELDS if k != "name"]
            mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
            w.writerow({"name": "mean", **{k: f"{mean[k]:.6f}" for k in keys}})
    print(f"[stats] psnr/ssim -> {path} ({len(rows)} images)")
    return path


def export_psnr_csv(
    qmodel: nn.Module,
    fp32_model: nn.Module,
    loader: DataLoader,
    device: str,
    path: str | Path,
) -> Path:
    rows = collect_psnr_rows(qmodel, fp32_model, loader, device)
    return write_psnr_csv(rows, path)

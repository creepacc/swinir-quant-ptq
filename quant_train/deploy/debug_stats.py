"""Dump layer-diff CSV + {step}_psnr.csv at a training checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .layer_stats import export_layer_stats_csv
from .psnr_log import export_psnr_csv


def dump_step_stats(
    step: str,
    qmodel: nn.Module,
    fp32_model: nn.Module,
    lr: torch.Tensor,
    image_name: str,
    loader,
    device: str,
    stat_dir: str | Path,
) -> None:
    out = Path(stat_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[stats] checkpoint={step} image={image_name}")
    export_layer_stats_csv(qmodel, lr, out / f"{step}_layers.csv")
    export_psnr_csv(qmodel, fp32_model, loader, device, out / f"{step}_psnr.csv")

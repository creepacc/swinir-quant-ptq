from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader

from .rotate import fuse_rotation
from .smoothquant import collect_channel_stats, fuse_osplus_post_ln, fuse_smoothquant


def apply_residual_fix(model: nn.Module, loader: DataLoader, cfg, device: str) -> None:
    mode = (getattr(cfg.quant, "residual_fix", "none") or "none").strip().lower()
    if mode in ("", "none"):
        print("[reparam] residual_fix=none")
        return
    alpha = float(getattr(cfg.quant, "smooth_alpha", 0.5))
    seed = int(getattr(cfg.quant, "rotate_seed", 0))
    do_rot = mode in ("quarot", "smooth_rotate")
    do_smooth = mode in ("smoothquant", "osplus", "smooth_rotate")
    if do_rot:
        fuse_rotation(model, seed=seed)
    if do_smooth:
        stats = collect_channel_stats(model, loader, device)
        fuse_smoothquant(model, stats, alpha=alpha)
        if mode in ("osplus", "smoothquant", "smooth_rotate"):
            fuse_osplus_post_ln(model, stats, alpha=alpha)
    print(f"[reparam] residual_fix={mode} done")

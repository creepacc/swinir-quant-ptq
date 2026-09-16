from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import PTQLossBase


def _align_fisher(fisher: torch.Tensor, ref: torch.Tensor) -> torch.Tensor | None:
    fisher = fisher.to(device=ref.device, dtype=ref.dtype)
    if fisher.shape == ref.shape:
        return fisher
    try:
        return torch.broadcast_to(fisher, ref.shape)
    except RuntimeError:
        return None


def _fisher_weight(fisher, ref: torch.Tensor) -> torch.Tensor | None:
    extra = fisher
    if extra is None or not torch.is_tensor(extra):
        return None
    g = _align_fisher(extra, ref)
    if g is None:
        return None
    w = g.pow(2)
    return w / w.mean().clamp(min=1e-12)


class ReconstructionMSELoss(PTQLossBase):
    def forward(self, q_out, fp_out, extra=None):
        extra = extra or {}
        tgt = fp_out.detach()
        diff2 = (q_out - tgt).pow(2)
        if extra.get("use_fisher"):
            w = _fisher_weight(extra.get("fisher"), diff2)
            if w is not None:
                diff2 = diff2 * w
        # AdaRound paper uses ||·||_F^2 (sum); Phase A keeps mean
        if extra.get("recon_reduction") == "sum":
            return diff2.sum()
        return diff2.mean()


class ReconstructionL1Loss(PTQLossBase):
    def forward(self, q_out, fp_out, extra=None):
        extra = extra or {}
        tgt = fp_out.detach()
        adiff = (q_out - tgt).abs()
        if extra.get("use_fisher"):
            w = _fisher_weight(extra.get("fisher"), adiff)
            if w is not None:
                adiff = adiff * w
        if extra.get("recon_reduction") == "sum":
            return adiff.sum()
        return adiff.mean()


def build_recon(name: str) -> PTQLossBase:
    return ReconstructionL1Loss() if name == "l1" else ReconstructionMSELoss()

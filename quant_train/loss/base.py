from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantize.config import LossConfig
from .feature import feature_distill_loss


class PTQLossBase(nn.Module):
    def forward(self, q_out, fp_out, extra=None):
        raise NotImplementedError


class CombinedPTQLoss(PTQLossBase):
    def __init__(self, cfg: LossConfig, recon: PTQLossBase, image: Optional[PTQLossBase], reg: PTQLossBase):
        super().__init__()
        self.cfg = cfg
        self.recon = recon
        self.image = image
        self.reg = reg

    def decompose(self, q_out, fp_out, extra=None):
        extra = extra or {}
        if extra.get("dqc"):
            recon_w = F.l1_loss(q_out, fp_out.detach())
            img_w = q_out.new_zeros(())
            q_feats, fp_feats = extra.get("q_feats"), extra.get("fp_feats")
            if q_feats and fp_feats:
                img_w = self.cfg.lambda_feat * feature_distill_loss(q_feats, fp_feats)
            return recon_w + img_w, recon_w, q_out.new_zeros(()), img_w
        recon = self.recon(q_out, fp_out, extra)
        recon_w = self.cfg.lambda_recon * recon
        total = recon_w
        img_w = q_out.new_zeros(())
        if self.image is not None and extra.get("q_img") is not None and extra.get("fp_img") is not None:
            img_w = self.cfg.lambda_img * self.image(extra["q_img"], extra["fp_img"])
            total = total + img_w
        reg_w = q_out.new_zeros(())
        if extra.get("alphas") and extra.get("apply_reg", True):
            reg_w = self.cfg.lambda_reg * self.reg(extra["alphas"], extra.get("beta", 2.0))
            total = total + reg_w
        return total, recon_w, reg_w, img_w

    def forward(self, q_out, fp_out, extra=None):
        total, _, _, _ = self.decompose(q_out, fp_out, extra)
        return total

from __future__ import annotations

import torch.nn.functional as F

from .base import PTQLossBase


class ImageMSELoss(PTQLossBase):
    def forward(self, q_out, fp_out, extra=None):
        return F.mse_loss(q_out, fp_out.detach())


class ImageL1Loss(PTQLossBase):
    def forward(self, q_out, fp_out, extra=None):
        return F.l1_loss(q_out, fp_out.detach())


def build_image(name: str) -> PTQLossBase:
    return ImageL1Loss() if name == "l1" else ImageMSELoss()

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _align(pred: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = pred.detach().float()
    b = ref.detach().float()
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])
    return a[..., :h, :w], b[..., :h, :w]


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """BT.601 luma, img in [0, 1], shape [B,C,H,W] or [C,H,W]."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.shape[1] == 1:
        return img
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr(pred: torch.Tensor, ref: torch.Tensor, max_val: float = 1.0) -> float:
    a, b = _align(pred, ref)
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def psnr_quant_vs_fp32(q_img: torch.Tensor, fp_img: torch.Tensor) -> float:
    return psnr(q_img, fp_img)


def psnr_vs_hr(sr_img: torch.Tensor, hr_img: torch.Tensor) -> float:
    return psnr(sr_img, hr_img)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).reshape(1, 1, window_size, window_size)


def ssim(pred: torch.Tensor, ref: torch.Tensor, max_val: float = 1.0) -> float:
    """SSIM on Y channel (HarmoQ / SR convention)."""
    a, b = _align(pred, ref)
    if a.ndim == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    a = rgb_to_y(a)
    b = rgb_to_y(b)
    window = _gaussian_window(11, 1.5, device=a.device, dtype=a.dtype)
    mu1 = F.conv2d(a, window, padding=5)
    mu2 = F.conv2d(b, window, padding=5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = F.conv2d(a * a, window, padding=5) - mu1_sq
    sig2 = F.conv2d(b * b, window, padding=5) - mu2_sq
    sig12 = F.conv2d(a * b, window, padding=5) - mu12
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu12 + c1) * (2 * sig12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2)
    return float((num / den.clamp(min=1e-12)).mean().item())


def ssim_quant_vs_fp32(q_img: torch.Tensor, fp_img: torch.Tensor) -> float:
    return ssim(q_img, fp_img)


def ssim_vs_hr(sr_img: torch.Tensor, hr_img: torch.Tensor) -> float:
    return ssim(sr_img, hr_img)

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] or [3,H,W] float in [0,1] -> uint8 HWC RGB."""
    x = t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def save_image(t: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor_to_image(t)
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image

        Image.fromarray(img).save(path)

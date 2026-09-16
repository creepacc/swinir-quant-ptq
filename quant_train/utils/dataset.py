from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


class Flickr2KSRDataset(Dataset):
    def __init__(
        self,
        root: str,
        scale: int = 4,
        patch_size: Optional[int] = 64,
        crop: str = "random",
        max_items: Optional[int] = None,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.scale = int(scale)
        self.patch_size = patch_size
        self.crop = crop
        self.hr_dir = self.root / "Flickr2K_HR"
        self.lr_dir = self.root / "Flickr2K_LR_bicubic" / f"X{self.scale}"
        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"LR dir not found: {self.lr_dir}")
        ids = sorted(p.stem.replace(f"x{self.scale}", "") for p in self.lr_dir.glob("*.png"))
        rng = np.random.RandomState(seed)
        if max_items is not None and max_items < len(ids):
            pick = rng.choice(len(ids), size=max_items, replace=False)
            ids = [ids[i] for i in sorted(pick.tolist())]
        self.ids: List[str] = ids

    def __len__(self) -> int:
        return len(self.ids)

    def _crop(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return lr, hr
        ph, pw = self.patch_size, self.patch_size
        h, w = lr.shape[:2]
        if h < ph or w < pw:
            return lr, hr
        if self.crop == "center":
            top = (h - ph) // 2
            left = (w - pw) // 2
        else:
            top = int(np.random.randint(0, h - ph + 1))
            left = int(np.random.randint(0, w - pw + 1))
        lr = lr[top : top + ph, left : left + pw]
        hr = hr[top * self.scale : (top + ph) * self.scale, left * self.scale : (left + pw) * self.scale]
        return lr, hr

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        lr = _read_rgb(self.lr_dir / f"{stem}x{self.scale}.png")
        hr = _read_rgb(self.hr_dir / f"{stem}.png")
        lr, hr = self._crop(lr, hr)
        return {
            "lr": _to_tensor(lr),
            "hr": _to_tensor(hr),
            "name": stem,
        }

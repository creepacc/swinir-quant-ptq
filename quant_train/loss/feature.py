from __future__ import annotations

import torch


def feature_distill_loss(q_feats: dict, fp_feats: dict) -> torch.Tensor:
    """2DQuant eq.5: MSE of L2-normalized teacher/student features."""
    total = None
    n = 0
    for name, q in q_feats.items():
        fp = fp_feats.get(name)
        if fp is None or not torch.is_tensor(q) or not torch.is_tensor(fp):
            continue
        fp = fp.detach().to(device=q.device, dtype=q.dtype)
        if fp.shape != q.shape:
            continue
        qn = q / q.norm().clamp(min=1e-12)
        fn = fp / fp.norm().clamp(min=1e-12)
        term = (qn - fn).pow(2).mean()
        total = term if total is None else total + term
        n += 1
    if total is None:
        ref = next((v for v in q_feats.values() if torch.is_tensor(v)), None)
        return torch.zeros((), device=ref.device if ref is not None else "cpu")
    return total / max(n, 1)

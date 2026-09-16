from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..quantize.ops.base import QuantOpBase, iter_quant_ops

_SKIP_SUBSTR = (
    ".input_fq.",
    ".weight_fq.",
    ".output_fq.",
    ".observer.",
    ".q_scale_mul.",
    ".qk_matmul.",
    ".bias_add.",
    ".mask_add.",
    ".av_matmul.",
    ".attn_res_add.",
    ".mlp_res_add.",
    ".body_res_add.",
    ".skip_add.",
)


@torch.no_grad()
def fold_adaround_weights(model: nn.Module) -> int:
    """Hard-round AdaRound into float_op.weight and drop alpha."""
    n = 0
    for m in iter_quant_ops(model):
        fq = m.weight_fq
        if fq is None or not hasattr(m, "float_op") or not hasattr(m.float_op, "weight"):
            continue
        w = m.float_op.weight
        folded = fq.hard_round_weight(w)
        w.copy_(folded.to(dtype=w.dtype))
        fq.drop_alpha()
        n += 1
    print(f"[fold] hard-rounded {n} weight tensors, alpha dropped")
    return n


def export_folded_state_dict(model: nn.Module, path: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folded = {}
    for k, v in model.state_dict().items():
        if k.endswith(".alpha") or any(s in k for s in _SKIP_SUBSTR):
            continue
        nk = k.replace(".float_op.", ".")
        folded[nk] = v.detach().cpu()
    torch.save({"params": folded}, path)
    print(f"[deploy] folded weights {len(folded)} tensors -> {path}")
    return path

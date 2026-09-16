"""Check whether residual-add outlier channels are stable across images/crops.

    python -m quant_train.tools.probe_outlier_channels
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..pipeline import steps
from ..quantize.ops.base import set_fakequant, set_observer
from ..quantize.ops.elementwise import QuantAdd
from ..utils.dataset import Flickr2KSRDataset

INI = "quant_train/configs/residual_study.ini"
TARGET = "layers.5.body_res_add"
K = 10


def _ch_absmax(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim == 4:
        return x.abs().amax(dim=(0, 2, 3)).cpu()
    return x.abs().amax(dim=tuple(range(x.ndim - 1))).cpu()


@torch.no_grad()
def collect_run(model: nn.Module, loader: DataLoader, device: str, names: tuple[str, ...]):
    bags: dict[str, dict] = {}

    def make(name):
        def hook(_m, inp, out):
            a = inp[0] if inp and torch.is_tensor(inp[0]) else None
            b = inp[1] if len(inp) > 1 and torch.is_tensor(inp[1]) else None
            rec = bags.setdefault(name, {"y": [], "a": [], "b": [], "img": []})
            rec["y"].append(_ch_absmax(out))
            if a is not None:
                rec["a"].append(_ch_absmax(a))
            if b is not None:
                rec["b"].append(_ch_absmax(b))

        return hook

    handles = []
    wanted = set(names)
    for n, m in model.named_modules():
        if n in wanted and isinstance(m, QuantAdd):
            handles.append(m.register_forward_hook(make(n)))
    imgs = []
    for batch in loader:
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else batch["name"]
        imgs.append(str(name))
        model(batch["lr"].to(device))
    for h in handles:
        h.remove()
    return bags, imgs


def _stack(xs):
    return torch.stack(xs, dim=0)  # [N, C]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_tensor(tag: str, mats: torch.Tensor, img_names: list[str], out_dir: Path):
    """mats: [N, C] per-image channel absmax."""
    n, c = mats.shape
    arr = mats.numpy()
    ranks = np.argsort(-arr, axis=1)
    topk = ranks[:, :K]
    mean_ch = arr.mean(axis=0)
    max_ch = arr.max(axis=0)
    mean_rank = np.argsort(-mean_ch)

    pair_s = []
    pair_j = []
    for i in range(n):
        si = set(topk[i].tolist())
        for j in range(i + 1, n):
            pair_s.append(spearman(arr[i], arr[j]))
            sj = set(topk[j].tolist())
            pair_j.append(len(si & sj) / float(K))

    freq = Counter()
    for row in topk:
        freq.update(int(x) for x in row.tolist())
    always = [ch for ch, cnt in freq.items() if cnt == n]
    often = [(ch, cnt) for ch, cnt in freq.most_common(15)]

    print(f"\n======== {tag}  N={n} C={c} top{K} ========")
    print(f"  pairwise Spearman(channel absmax): mean={np.mean(pair_s):.3f}  min={np.min(pair_s):.3f}")
    print(f"  pairwise top{K} Jaccard:           mean={np.mean(pair_j):.3f}  min={np.min(pair_j):.3f}")
    print(f"  channels in EVERY image's top{K}:  {sorted(always)}  ({len(always)})")
    print(f"  most frequent top{K} channels:     {often[:10]}")
    print(f"  mean-absmax top{K} channels:       {mean_rank[:K].tolist()}")
    print(f"  mean top1 / mean median channel:   {mean_ch[mean_rank[0]]:.1f} / {np.median(mean_ch):.1f}")

    csv_path = out_dir / f"{tag.replace('.', '_')}_channel_absmax.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image"] + [f"ch{i}" for i in range(c)] + [f"top{k}" for k in range(K)])
        for i, name in enumerate(img_names):
            w.writerow([name] + [f"{x:.6g}" for x in arr[i]] + [int(x) for x in topk[i]])
    freq_path = out_dir / f"{tag.replace('.', '_')}_topk_freq.csv"
    with freq_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "times_in_topK", "mean_absmax", "max_absmax"])
        for ch in range(c):
            w.writerow([ch, freq.get(ch, 0), f"{mean_ch[ch]:.6g}", f"{max_ch[ch]:.6g}"])
    print(f"  wrote {csv_path.name}, {freq_path.name}")
    return {
        "tag": tag,
        "spearman": float(np.mean(pair_s)),
        "jaccard": float(np.mean(pair_j)),
        "always_k": len(always),
        "always": sorted(always),
        "mean_top": mean_rank[:K].tolist(),
    }


def make_loader(cfg, n, crop, seed):
    ds = Flickr2KSRDataset(
        root=cfg.paths.data_root,
        scale=cfg.model.scale,
        patch_size=cfg.data.eval_patch_size,
        crop=crop,
        max_items=n,
        seed=seed,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), list(ds.ids)


def main():
    root = Path("/mnt/e/tensorrt/tensorRT_10/swinir-trt")
    cfg = steps.load_config(str(root / INI))
    device = steps._device(cfg)
    fp32 = steps.build_fp32_swinir(cfg)
    qmodel = steps.prepare_quant_model(fp32, cfg)
    set_fakequant(qmodel, False)
    set_observer(qmodel, False)
    qmodel.eval()
    out = Path(cfg.paths.output_dir) / "outlier_channels"
    out.mkdir(parents=True, exist_ok=True)

    body_names = tuple(f"layers.{i}.body_res_add" for i in range(6)) + (TARGET,)
    body_names = tuple(dict.fromkeys(body_names))

    runs = [
        ("center_s0_n24", 24, "center", 0),
        ("center_s1_n24", 24, "center", 1),
        ("random_s0_n24", 24, "random", 0),
    ]
    recs = []
    for tag, n, crop, seed in runs:
        loader, ids = make_loader(cfg, n, crop, seed)
        print(f"\n[run] {tag} crop={crop} seed={seed} ids[:8]={ids[:8]}")
        bags, imgs = collect_run(qmodel, loader, device, body_names)
        y = _stack(bags[TARGET]["y"])
        a = _stack(bags[TARGET]["a"])
        recs.append(summarize_tensor(f"{tag}__{TARGET}__y", y, imgs, out))
        recs.append(summarize_tensor(f"{tag}__{TARGET}__conv_branch", a, imgs, out))

    # same 24 center s0 images: body add of every RSTB
    loader, ids = make_loader(cfg, 24, "center", 0)
    bags, imgs = collect_run(qmodel, loader, device, body_names)
    print("\n======== same 24 center images, body_res_add at each RSTB ========")
    mean_tops = {}
    for i in range(6):
        name = f"layers.{i}.body_res_add"
        rec = summarize_tensor(f"center_s0_n24__{name}__y", _stack(bags[name]["y"]), imgs, out)
        mean_tops[i] = rec["mean_top"]
        recs.append(rec)
    print("\nmean-absmax top10 channels vs RSTB depth:")
    for i, t in mean_tops.items():
        print(f"  RSTB{i}: {t}")

    # overlap of mean ranking RSTB5 vs earlier
    t5 = set(mean_tops[5][:5])
    print("\ntop5(RSTB5) overlap with earlier RSTB mean-top10:")
    for i in range(5):
        print(f"  vs RSTB{i}: {sorted(t5 & set(mean_tops[i][:10]))}")

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "spearman", "jaccard", "always_k", "always", "mean_top"])
        w.writeheader()
        for r in recs:
            w.writerow({**r, "always": " ".join(map(str, r["always"])), "mean_top": " ".join(map(str, r["mean_top"]))})
    print(f"\n[done] {summary}")


if __name__ == "__main__":
    main()

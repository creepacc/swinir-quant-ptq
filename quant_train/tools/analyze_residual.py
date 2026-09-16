"""Calibrate W8A8, eval residual-act ON vs OFF, dump residual I/O histograms.

Usage:
    python -m quant_train.tools.analyze_residual \\
        --config quant_train/configs/residual_study.ini
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pipeline import steps
from ..quantize.ops.base import QuantOpBase, set_fakequant
from ..quantize.ops.elementwise import QuantAdd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_ERR = exc
else:
    _MPL_ERR = None


RESIDUAL_KEYS = ("attn_res_add", "mlp_res_add", "body_res_add", "skip_add")
PLOT_LAYERS = (
    "skip_add",
    "layers.0.residual_group.blocks.0.attn_res_add",
    "layers.0.residual_group.blocks.0.mlp_res_add",
    "layers.0.body_res_add",
    "layers.2.body_res_add",
    "layers.5.residual_group.blocks.0.attn_res_add",
    "layers.5.residual_group.blocks.5.mlp_res_add",
    "layers.5.body_res_add",
)


def _kind(name: str) -> str | None:
    for key in RESIDUAL_KEYS:
        if name.endswith(key) or name == key:
            return key
    return None


def iter_residual_adds(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, QuantAdd) and _kind(name):
            yield name, m


def set_residual_fakequant(model: nn.Module, enabled: bool) -> int:
    n = 0
    for _, m in iter_residual_adds(model):
        if m.output_fq is not None:
            m.output_fq.enable_fakequant(enabled)
            n += 1
    return n


def set_residual_consumer_input_fq(model: nn.Module, enabled: bool) -> int:
    """Linear/Conv that read the residual stream: qkv, mlp.fc1, RSTB conv, conv_after_body."""
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, QuantOpBase) or m.input_fq is None:
            continue
        hit = (
            name.endswith(".attn.qkv")
            or name.endswith(".mlp.fc1")
            or name.endswith(".conv")
            or name == "conv_after_body"
        )
        if hit:
            m.input_fq.enable_fakequant(enabled)
            n += 1
    return n


def _pct(x: torch.Tensor, p: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x.detach().float().reshape(-1), p).item())


def _sqnr(fp: torch.Tensor, q: torch.Tensor) -> float:
    fp = fp.float()
    err = (fp - q.float()).pow(2).mean()
    sig = fp.pow(2).mean()
    if float(err) <= 0:
        return 99.0
    return float((10.0 * torch.log10(sig.clamp(min=1e-12) / err.clamp(min=1e-12))).item())


def _channel_axis(x: torch.Tensor) -> int:
    # tokens: (B, L, C) or NCHW: (B, C, H, W)
    if x.ndim == 4:
        return 1
    return -1


def _channel_maxabs(x: torch.Tensor) -> torch.Tensor:
    ax = _channel_axis(x)
    dims = [i for i in range(x.ndim) if i != ax % x.ndim and i != ax]
    return x.detach().float().abs().amax(dim=dims)


@torch.no_grad()
def collect_residual_bags(model: nn.Module, x: torch.Tensor, fq_on: bool):
    set_fakequant(model, fq_on)
    if fq_on:
        set_residual_fakequant(model, True)
    bags = {}

    def make(name, op):
        def hook(_m, inp, out):
            a = inp[0].detach() if inp and torch.is_tensor(inp[0]) else None
            b = inp[1].detach() if len(inp) > 1 and torch.is_tensor(inp[1]) else None
            y = out.detach() if torch.is_tensor(out) else None
            bags[name] = {"a": a, "b": b, "y": y, "op": op}

        return hook

    handles = []
    for name, m in iter_residual_adds(model):
        handles.append(m.register_forward_hook(make(name, m)))
    model.eval()
    model(x)
    for h in handles:
        h.remove()
    return bags


def _clip_bounds(op: QuantOpBase):
    fq = op.output_fq
    if fq is None:
        return None, None
    lo, hi = fq._clip_pair()
    if lo is None or hi is None:
        lo, hi = fq._bounds_from_qparams()
    return float(lo.reshape(-1)[0]), float(hi.reshape(-1)[0])


def _apply_local_fq(op: QuantOpBase, y: torch.Tensor) -> torch.Tensor:
    fq = op.output_fq
    if fq is None:
        return y
    was = fq.fake_quant_enabled
    fq.enable_fakequant(True)
    q = fq(y)
    fq.enable_fakequant(was)
    return q


def analyze_one(name: str, fp: dict, qnet: dict) -> dict:
    a = fp["a"]
    b = fp["b"]
    y = fp["y"]
    yq_local = _apply_local_fq(fp["op"], y)
    yq_acc = qnet["y"]
    a_acc = qnet["a"]
    b_acc = qnet["b"]
    lo, hi = _clip_bounds(fp["op"])
    y_flat = y.float().reshape(-1)
    sat = 0.0
    if lo is not None:
        sat = float(((y_flat < lo) | (y_flat > hi)).float().mean().item())
    ch = _channel_maxabs(y)
    g = float(ch.max().clamp(min=1e-12))
    waste = float(torch.log2((g / ch.clamp(min=1e-12)).mean()).item())
    row = {
        "layer": name,
        "kind": _kind(name),
        "shape": "x".join(str(s) for s in y.shape),
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "a_mean": float(a.float().mean()),
        "a_std": float(a.float().std()),
        "a_p99": _pct(a.abs(), 0.99),
        "a_p999": _pct(a.abs(), 0.999),
        "b_min": float(b.min()),
        "b_max": float(b.max()),
        "b_mean": float(b.float().mean()),
        "b_std": float(b.float().std()),
        "b_p99": _pct(b.abs(), 0.99),
        "b_p999": _pct(b.abs(), 0.999),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_mean": float(y.float().mean()),
        "y_std": float(y.float().std()),
        "y_p99": _pct(y.abs(), 0.99),
        "y_p999": _pct(y.abs(), 0.999),
        "clip_lo": lo if lo is not None else 0.0,
        "clip_hi": hi if hi is not None else 0.0,
        "sat_ratio": sat,
        "local_mae": float((y - yq_local).abs().mean()),
        "local_mse": float((y - yq_local).pow(2).mean()),
        "local_cos": float(F.cosine_similarity(y.reshape(1, -1).float(), yq_local.reshape(1, -1).float()).item()),
        "local_sqnr": _sqnr(y, yq_local),
        "acc_mae": float((y - yq_acc).abs().mean()),
        "acc_mse": float((y - yq_acc).pow(2).mean()),
        "acc_cos": float(F.cosine_similarity(y.reshape(1, -1).float(), yq_acc.reshape(1, -1).float()).item()),
        "acc_sqnr": _sqnr(y, yq_acc),
        "a_acc_mae": float((a - a_acc).abs().mean()),
        "b_acc_mae": float((b - b_acc).abs().mean()),
        "ch_maxabs_mean": float(ch.mean()),
        "ch_maxabs_max": float(ch.max()),
        "ch_maxabs_min": float(ch.min()),
        "per_ch_bit_waste": waste,
        "scale": float(fp["op"].output_fq.scale.reshape(-1)[0]) if fp["op"].output_fq is not None else 0.0,
    }
    return row, {
        "a": a,
        "b": b,
        "y": y,
        "yq_local": yq_local,
        "yq_acc": yq_acc,
        "lo": lo,
        "hi": hi,
    }


def _to_np(t: torch.Tensor, max_n: int = 200_000) -> np.ndarray:
    x = t.detach().float().reshape(-1)
    if x.numel() > max_n:
        idx = torch.randint(0, x.numel(), (max_n,), device=x.device)
        x = x[idx]
    return x.cpu().numpy()


def _hist_with_clip(ax, data, bins, color, label, lo=None, hi=None):
    ax.hist(data, bins=bins, color=color, alpha=0.75, label=label, density=True)
    if lo is not None:
        ax.axvline(lo, color="k", ls="--", lw=1)
        ax.axvline(hi, color="k", ls="--", lw=1)


def save_layer_plot(path: Path, name: str, rec: dict, bins: int = 128):
    if plt is None:
        raise RuntimeError(f"matplotlib required: {_MPL_ERR}")
    a = _to_np(rec["a"])
    b = _to_np(rec["b"])
    y = _to_np(rec["y"])
    yq = _to_np(rec["yq_local"])
    err = _to_np(rec["y"] - rec["yq_local"])
    lo, hi = rec["lo"], rec["hi"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    _hist_with_clip(axes[0, 0], a, bins, "#4C78A8", "input0 (shortcut)", lo, hi)
    axes[0, 0].set_title(f"{name}\ninput0  [{a.min():.3g}, {a.max():.3g}]")
    _hist_with_clip(axes[0, 1], b, bins, "#F58518", "input1 (branch)", lo, hi)
    axes[0, 1].set_title(f"input1  [{b.min():.3g}, {b.max():.3g}]")
    axes[1, 0].hist(y, bins=bins, color="#54A24B", alpha=0.7, label="FP32 out", density=True)
    axes[1, 0].hist(yq, bins=bins, color="#E45756", alpha=0.55, label="INT8 dequant", density=True)
    if lo is not None:
        axes[1, 0].axvline(lo, color="k", ls="--", lw=1, label="clip")
        axes[1, 0].axvline(hi, color="k", ls="--", lw=1)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title(f"output FP vs Q  clip=[{lo:.3g}, {hi:.3g}]")
    axes[1, 1].hist(err, bins=bins, color="#B279A2", alpha=0.85, density=True)
    axes[1, 1].set_title(f"local quant error  mae={np.abs(err).mean():.3g}")
    for ax in axes.ravel():
        ax.set_xlabel("value")
        ax.set_ylabel("density")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_depth_plot(path: Path, rows: list[dict]):
    if plt is None:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for kind, rs in by.items():
        xs = list(range(len(rs)))
        axes[0].plot(xs, [r["y_max"] for r in rs], marker="o", ms=3, label=kind)
        axes[1].plot(xs, [r["local_sqnr"] for r in rs], marker="o", ms=3, label=kind)
        axes[2].plot(xs, [r["sat_ratio"] * 100 for r in rs], marker="o", ms=3, label=kind)
    axes[0].set_title("FP32 |out| max vs depth")
    axes[1].set_title("local SQNR (dB)")
    axes[2].set_title("saturation %")
    for ax in axes:
        ax.set_xlabel("occurrence order")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]):
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    print("\n[residual] family summary (local FQ on FP32 sum, one eval image)")
    print(f"{'kind':16s} n  y_max_mean  sat%   SQNR  acc_SQNR  acc_mae  bit_waste")
    for kind in RESIDUAL_KEYS:
        rs = by.get(kind, [])
        if not rs:
            continue
        n = len(rs)
        ymax = sum(r["y_max"] for r in rs) / n
        sat = 100 * sum(r["sat_ratio"] for r in rs) / n
        sq = sum(r["local_sqnr"] for r in rs) / n
        asq = sum(r["acc_sqnr"] for r in rs) / n
        amae = sum(r["acc_mae"] for r in rs) / n
        waste = sum(r["per_ch_bit_waste"] for r in rs) / n
        print(f"{kind:16s} {n:2d}  {ymax:9.3f}  {sat:5.2f}  {sq:6.2f}  {asq:7.2f}  {amae:7.3f}  {waste:6.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = steps.load_config(args.config)
    out = Path(cfg.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[residual-study] skip_residual_act={cfg.quant.skip_residual_act} "
          f"(analysis always keeps residual FQ, then disables it for OFF eval)")

    fp32 = steps.build_fp32_swinir(cfg)
    qmodel = steps.prepare_quant_model(fp32, cfg)
    n_res = sum(1 for _ in iter_residual_adds(qmodel))
    print(f"[residual-study] residual QuantAdd count={n_res}")
    parts, trainer = steps.make_partitioner_and_trainer(qmodel, cfg)

    print("[1] calibrate + freeze (residual FQ ON)")
    steps.calibrate_observers(qmodel, cfg, trainer)
    steps.freeze_qparams(qmodel, trainer)

    stat_lr, _, stat_name = steps.resolve_stat_image(cfg)
    device = steps._device(cfg)
    lr = stat_lr.to(device)
    print(f"[2] residual I/O dump on {stat_name}")
    bags_fp = collect_residual_bags(qmodel, lr, fq_on=False)
    bags_q = collect_residual_bags(qmodel, lr, fq_on=True)

    rows = []
    plots = {}
    for name in bags_fp:
        row, rec = analyze_one(name, bags_fp[name], bags_q[name])
        rows.append(row)
        plots[name] = rec
    rows.sort(key=lambda r: r["layer"])
    write_csv(out / "stats" / "residual_io.csv", rows)
    summarize(rows)

    hist_dir = out / "hists"
    plotted = 0
    wanted = set(PLOT_LAYERS)
    # always plot skip / first / last of each kind
    for name, rec in plots.items():
        is_rstb = name.endswith("body_res_add") and name.startswith("layers.")
        if name in wanted or is_rstb:
            save_layer_plot(hist_dir / f"{name.replace('.', '_')}.png", name, rec, bins=cfg.deploy.hist_bins)
            plotted += 1
    save_depth_plot(hist_dir / "residual_depth.png", rows)
    print(f"[2] wrote {len(rows)} rows, {plotted} layer hists -> {hist_dir}")

    def _eval(tag: str):
        scores = steps.evaluate_psnr(
            qmodel, fp32, cfg, enable_all_fq=False, eval_subdir=f"eval_{tag}"
        )
        return np.mean(np.array(scores), axis=0) if scores else None

    print("[3] eval residual Add FQ ON (calib only, no AdaRound/SRC)")
    set_fakequant(qmodel, True)
    set_residual_fakequant(qmodel, True)
    on_mean = _eval("residual_on")

    print("[4] eval residual Add FQ OFF (next Linear/Conv input FQ still on)")
    n_off = set_residual_fakequant(qmodel, False)
    print(f"[4] disabled residual Add output FQ: {n_off}")
    off_mean = _eval("residual_off")

    print("[5] eval residual STREAM FP32 (Add out FQ + consumer input FQ off)")
    n_in = set_residual_consumer_input_fq(qmodel, False)
    print(f"[5] disabled residual-consumer input FQ: {n_in}")
    stream_mean = _eval("residual_stream_off")

    summary_path = out / "stats" / "psnr_residual_on_vs_off.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "q_vs_fp32", "q_vs_hr", "fp32_vs_hr", "q_vs_fp32_ssim", "q_vs_hr_ssim", "fp32_vs_hr_ssim"])
        for tag, mean in (
            ("residual_on", on_mean),
            ("residual_add_off", off_mean),
            ("residual_stream_off", stream_mean),
        ):
            if mean is not None:
                w.writerow([tag, *[f"{x:.6f}" for x in mean]])
    print("\n======== residual ON vs OFF ========")
    for label, mean in (
        ("Add FQ ON            ", on_mean),
        ("Add FQ OFF           ", off_mean),
        ("residual stream FP32 ", stream_mean),
    ):
        if mean is not None:
            print(f"  {label} q_vs_fp32={mean[0]:.3f}  q_vs_hr={mean[1]:.3f}  fp32_vs_hr={mean[2]:.3f}")
    print(f"[done] csv={out / 'stats' / 'residual_io.csv'}")
    print(f"[done] psnr={summary_path}")


if __name__ == "__main__":
    main()

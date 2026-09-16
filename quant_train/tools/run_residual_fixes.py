"""Run residual INT8 fix ablations (M1/M2/M12/M3) and dump PSNR + residual SQNR.

    python -m quant_train.tools.run_residual_fixes
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ..pipeline import steps
from ..quantize.ops.base import set_fakequant
from .analyze_residual import (
    analyze_one,
    collect_residual_bags,
    summarize,
    write_csv,
)

CONFIGS = [
    ("none", "quant_train/configs/residual_study.ini"),
    ("M1_smoothquant", "quant_train/configs/residual_m1_smooth.ini"),
    ("M2_quarot", "quant_train/configs/residual_m2_rotate.ini"),
    ("M12_smooth_rotate", "quant_train/configs/residual_m12_smooth_rotate.ini"),
    ("M3_residual_clip", "quant_train/configs/residual_m3_clip.ini"),
]


@torch.no_grad()
def _fp32_match(qmodel, fp32, lr) -> float:
    set_fakequant(qmodel, False)
    a = qmodel(lr)
    b = fp32(lr)
    return float((a - b).abs().max().item())


def run_one(tag: str, ini: str) -> dict:
    cfg = steps.load_config(ini)
    cfg.eval.save_images = False
    print(f"\n======== {tag}  residual_fix={cfg.quant.residual_fix} ========")
    fp32 = steps.build_fp32_swinir(cfg)
    qmodel = steps.prepare_quant_model(fp32, cfg)
    parts, trainer = steps.make_partitioner_and_trainer(qmodel, cfg)
    steps.apply_stream_reparam(qmodel, cfg)
    device = steps._device(cfg)
    lr, _, stat_name = steps.resolve_stat_image(cfg)
    lr = lr.to(device)
    drift = _fp32_match(qmodel, fp32, lr)
    print(f"[{tag}] FP32 max-abs drift after reparam (FQ off) = {drift:.4g}  image={stat_name}")

    steps.calibrate_observers(qmodel, cfg, trainer)
    steps.freeze_qparams(qmodel, trainer)

    bags_fp = collect_residual_bags(qmodel, lr, fq_on=False)
    bags_q = collect_residual_bags(qmodel, lr, fq_on=True)
    rows = []
    for name in bags_fp:
        row, _ = analyze_one(name, bags_fp[name], bags_q[name])
        rows.append(row)
    rows.sort(key=lambda r: r["layer"])
    out = Path(cfg.paths.output_dir)
    write_csv(out / "stats" / "residual_io.csv", rows)
    summarize(rows)
    body5 = next((r for r in rows if r["layer"] == "layers.5.body_res_add"), None)

    set_fakequant(qmodel, True)
    scores = steps.evaluate_psnr(qmodel, fp32, cfg, enable_all_fq=False, eval_subdir="eval")
    mean = np.mean(np.array(scores), axis=0)
    rec = {
        "tag": tag,
        "residual_fix": cfg.quant.residual_fix,
        "fp32_drift": drift,
        "q_vs_fp32": float(mean[0]),
        "q_vs_hr": float(mean[1]),
        "fp32_vs_hr": float(mean[2]),
        "q_vs_fp32_ssim": float(mean[3]),
        "body5_scale": float(body5["scale"]) if body5 else 0.0,
        "body5_local_sqnr": float(body5["local_sqnr"]) if body5 else 0.0,
        "body5_bit_waste": float(body5["per_ch_bit_waste"]) if body5 else 0.0,
        "body5_sat": float(body5["sat_ratio"]) if body5 else 0.0,
        "body5_clip_hi": float(body5["clip_hi"]) if body5 else 0.0,
    }
    print(
        f"[{tag}] PSNR q_vs_fp32={rec['q_vs_fp32']:.3f} q_vs_hr={rec['q_vs_hr']:.3f} "
        f"body5_scale={rec['body5_scale']:.4g} sqnr={rec['body5_local_sqnr']:.2f} "
        f"sat={100 * rec['body5_sat']:.2f}%"
    )
    return rec


def main():
    root = Path("/mnt/e/tensorrt/tensorRT_10/swinir-trt")
    recs = []
    for tag, ini in CONFIGS:
        recs.append(run_one(tag, str(root / ini)))
    summary = root / "quant_train/outputs_residual_fixes_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    keys = list(recs[0].keys())
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    print(f"\n[summary] {summary}")
    for r in recs:
        print(
            f"  {r['tag']:22s}  q/fp32={r['q_vs_fp32']:.3f}  q/hr={r['q_vs_hr']:.3f}  "
            f"drift={r['fp32_drift']:.3g}  body5_scale={r['body5_scale']:.4g}"
        )


if __name__ == "__main__":
    main()

"""Measure residual-Add input range mismatch vs output FQ step."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_train.pipeline import steps
from quant_train.quantize.ops.base import set_fakequant, set_observer
from quant_train.quantize.ops.elementwise import QuantAdd


RESIDUAL_NAMES = ("attn_res_add", "mlp_res_add", "body_res_add", "skip_add")


def _kind(name: str) -> str:
    for k in RESIDUAL_NAMES:
        if name.endswith(k) or f".{k}" in name:
            return k
    return "other_add"


def main():
    cfg = steps.load_config(str(ROOT / "quant_train/configs/first_run.ini"))
    cfg.train.adaround = False
    cfg.train.src = False
    cfg.quant.skip_attn_act = False
    cfg.quant.skip_residual_act = False
    device = steps._device(cfg)

    fp32 = steps.build_fp32_swinir(cfg)
    qmodel = steps.prepare_quant_model(fp32, cfg)
    qmodel.eval()
    set_observer(qmodel, False)
    set_fakequant(qmodel, False)

    recs = defaultdict(list)

    def make_hook(name: str):
        def hook(mod, inp, _out):
            a, b = inp[0].detach().float(), inp[1].detach().float()
            recs[name].append((a.cpu(), b.cpu()))

        return hook

    n_hook = 0
    for name, m in qmodel.named_modules():
        if isinstance(m, QuantAdd) and _kind(name) != "other_add":
            m.register_forward_hook(make_hook(name))
            n_hook += 1
    print(f"[diag] hooked {n_hook} residual Adds, fakequant off")

    loader = steps.make_loader(cfg, cfg.data.eval_num, cfg.data.eval_patch_size, "center")
    with torch.no_grad():
        for batch in loader:
            qmodel(batch["lr"].to(device))

    rows = []
    by_kind = defaultdict(list)
    for name, pairs in recs.items():
        a = torch.cat([p[0].reshape(-1) for p in pairs])
        b = torch.cat([p[1].reshape(-1) for p in pairs])
        s = a + b
        rms_a = float(a.pow(2).mean().sqrt())
        rms_b = float(b.pow(2).mean().sqrt())
        max_a = float(a.abs().max())
        max_b = float(b.abs().max())
        rng_s = float(s.max() - s.min())
        lsb = rng_s / 255.0 if rng_s > 0 else 0.0
        frac_lt_lsb = float((b.abs() < 0.5 * lsb).float().mean()) if lsb > 0 else 0.0
        ratio = rms_a / max(rms_b, 1e-12)
        snr = 20.0 * torch.log10(torch.tensor(rms_b / max(lsb / (12 ** 0.5), 1e-12))).item()
        row = {
            "name": name,
            "kind": _kind(name),
            "rms_a": rms_a,
            "rms_b": rms_b,
            "ratio": ratio,
            "max_a": max_a,
            "max_b": max_b,
            "lsb": lsb,
            "frac_b_lt_half_lsb": frac_lt_lsb,
            "snr_b_vs_sum_q": snr,
        }
        rows.append(row)
        by_kind[row["kind"]].append(row)

    rows.sort(key=lambda r: r["ratio"], reverse=True)
    print("\n# top-12 residual Adds by RMS(shortcut)/RMS(branch)")
    print(f"{'name':<55} {'kind':<14} {'rms_a':>8} {'rms_b':>8} {'a/b':>7} {'lsb':>8} {'%|b|<0.5lsb':>12} {'SNR_dB':>8}")
    for r in rows[:12]:
        print(
            f"{r['name']:<55} {r['kind']:<14} {r['rms_a']:8.3f} {r['rms_b']:8.3f} "
            f"{r['ratio']:7.2f} {r['lsb']:8.4f} {100 * r['frac_b_lt_half_lsb']:11.1f}% {r['snr_b_vs_sum_q']:8.1f}"
        )

    print("\n# per-kind mean")
    print(f"{'kind':<14} {'n':>3} {'rms_a':>8} {'rms_b':>8} {'a/b':>7} {'%|b|<0.5lsb':>12} {'SNR_dB':>8}")
    for kind in RESIDUAL_NAMES:
        xs = by_kind.get(kind, [])
        if not xs:
            continue
        n = len(xs)
        mean = lambda k: sum(x[k] for x in xs) / n
        print(
            f"{kind:<14} {n:3d} {mean('rms_a'):8.3f} {mean('rms_b'):8.3f} "
            f"{mean('ratio'):7.2f} {100 * mean('frac_b_lt_half_lsb'):11.1f}% {mean('snr_b_vs_sum_q'):8.1f}"
        )

    worst = rows[0]
    best = min(rows, key=lambda r: r["ratio"])
    print(f"\n# worst a/b: {worst['name']}  {worst['ratio']:.2f}")
    print(f"# best  a/b: {best['name']}  {best['ratio']:.2f}")


if __name__ == "__main__":
    main()

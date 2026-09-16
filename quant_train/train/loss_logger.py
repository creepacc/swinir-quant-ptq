"""Record per-iter recon / reg / total and draw curves."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_ERR = exc
else:
    _MPL_ERR = None


class LossLogger:
    def __init__(self):
        self.rows: List[dict] = []

    def add(
        self,
        phase: str,
        part: str,
        step: int,
        total: float,
        recon: float,
        reg: float,
        image: float = 0.0,
        h_soft: float = 0.0,
        beta: float = 0.0,
    ):
        self.rows.append(
            {
                "phase": phase,
                "part": part,
                "step": int(step),
                "total": float(total),
                "recon": float(recon),
                "reg": float(reg),
                "image": float(image),
                "h_soft": float(h_soft),
                "beta": float(beta),
            }
        )

    def _phase_rows(self, phase: str) -> List[dict]:
        return [r for r in self.rows if r["phase"] == phase]

    def save(self, out_dir: str | Path, phase: str | None = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        phases = [phase] if phase else sorted({r["phase"] for r in self.rows})
        for ph in phases:
            recs = self._phase_rows(ph)
            if not recs:
                continue
            csv_path = out / f"loss_{ph}.csv"
            fields = ["phase", "part", "step", "total", "recon", "reg", "image", "h_soft", "beta"]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(recs)
            _plot_phase(recs, out / f"loss_{ph}.png", ph)
            print(f"[stats] loss curve -> {out / f'loss_{ph}.png'} ({len(recs)} iters)")


def _plot_phase(recs: List[dict], path: Path, phase: str):
    if plt is None:
        print(f"[warn] skip loss plot, matplotlib missing: {_MPL_ERR}")
        return
    xs = list(range(len(recs)))
    n_rows = 3 if phase == "adaround" else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10.0, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    axes[0].plot(xs, [r["total"] for r in recs], label="total", linewidth=1.2)
    axes[0].plot(xs, [r["recon"] for r in recs], label="recon", linewidth=1.0, alpha=0.85)
    axes[0].plot(xs, [r["reg"] for r in recs], label="reg", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{phase} losses")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    by_part = defaultdict(list)
    for i, r in enumerate(recs):
        by_part[r["part"]].append((i, r["total"]))
    for part, pts in by_part.items():
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.8, alpha=0.7, label=part)
    axes[1].set_ylabel("total")
    if len(by_part) <= 8:
        axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    if phase == "adaround":
        axes[2].plot(xs, [r.get("h_soft", 0.0) for r in recs], color="C3", linewidth=1.2, label="mean(1-|2h-1|)")
        axb = axes[2].twinx()
        axb.plot(xs, [r.get("beta", 0.0) for r in recs], color="C4", linewidth=1.0, alpha=0.7, label="β")
        axes[2].set_ylabel("h_soft")
        axb.set_ylabel("β")
        axes[2].set_xlabel("iter (parts concatenated)")
        h1, l1 = axes[2].get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        axes[2].legend(h1 + h2, l1 + l2, loc="upper right")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title("h(V) softness (should rise in warmup, then fall as β anneals)")
    else:
        axes[1].set_xlabel("iter (parts concatenated)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

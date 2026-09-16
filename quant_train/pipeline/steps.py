"""Thin wrappers for each PTQ step. Read main.py for the order."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..blocks.partitioner import SwinIRPartitioner
from ..deploy.debug_stats import dump_step_stats
from ..deploy.export import export_quant_params
from ..deploy.fold import export_folded_state_dict, fold_adaround_weights
from ..deploy.load import load_quant_params as _load_quant_params
from ..deploy.ranges import export_histograms, snapshot_weights
from ..quantize.config import AppConfig, load_ini
from ..quantize.model.convert import prepare_swinir
from ..quantize.ops.base import set_fakequant, set_observer
from ..quantize.reparam import apply_residual_fix
from ..train.trainer import build_trainer
from ..utils.dataset import Flickr2KSRDataset
from ..utils.image import save_image
from ..train.src import apply_src
from ..utils.metrics import psnr_quant_vs_fp32, psnr_vs_hr, ssim_quant_vs_fp32, ssim_vs_hr
from ..utils.progress import progress


def load_config(ini_path: str) -> AppConfig:
    cfg = load_ini(ini_path)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if cfg.paths.sr_root not in sys.path:
        sys.path.insert(0, cfg.paths.sr_root)
    return cfg


def _device(cfg: AppConfig) -> str:
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to cpu")
        return "cpu"
    return cfg.train.device


def build_fp32_swinir(cfg: AppConfig):
    from swinir.network_swinir import SwinIR, load_pretrained_weights

    m = cfg.model
    model = SwinIR(
        upscale=m.scale,
        in_chans=3,
        img_size=m.img_size,
        window_size=m.window_size,
        img_range=m.img_range,
        depths=list(m.depths),
        embed_dim=m.embed_dim,
        num_heads=list(m.num_heads),
        mlp_ratio=m.mlp_ratio,
        upsampler=m.upsampler,
        resi_connection=m.resi_connection,
    )
    if cfg.paths.weights:
        load_pretrained_weights(model, cfg.paths.weights, device="cpu")
    model.eval()
    return model


def prepare_quant_model(fp32_model, cfg: AppConfig):
    qmodel = prepare_swinir(copy.deepcopy(fp32_model), cfg)
    device = _device(cfg)
    fp32_model.to(device)
    qmodel.to(device)
    print("[prepare] wrapped Linear/Conv/ewise/activations "
          f"(skip_attn_act={cfg.quant.skip_attn_act} skip_residual_act={cfg.quant.skip_residual_act})")
    return qmodel


def make_loader(cfg: AppConfig, max_items: int, patch_size: int, crop: str) -> DataLoader:
    ds = Flickr2KSRDataset(
        root=cfg.paths.data_root,
        scale=cfg.model.scale,
        patch_size=patch_size,
        crop=crop,
        max_items=max_items,
        seed=cfg.train.seed,
    )
    return DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )


def load_quant_params(qmodel, cfg: AppConfig):
    _load_quant_params(qmodel, cfg.paths.quant_params)
    set_observer(qmodel, False)
    set_fakequant(qmodel, True)


def apply_stream_reparam(qmodel, cfg: AppConfig):
    mode = (getattr(cfg.quant, "residual_fix", "none") or "none").strip().lower()
    if mode in ("", "none"):
        return
    device = _device(cfg)
    loader = make_loader(cfg, cfg.data.calib_num, cfg.data.patch_size, cfg.data.crop)
    set_fakequant(qmodel, False)
    set_observer(qmodel, False)
    apply_residual_fix(qmodel, loader, cfg, device)


def calibrate_observers(qmodel, cfg: AppConfig, trainer):
    device = _device(cfg)
    loader = make_loader(cfg, cfg.data.calib_num, cfg.data.patch_size, cfg.data.crop)
    trainer.calibrate(qmodel, loader, device)


def freeze_qparams(qmodel, trainer):
    trainer.freeze(qmodel)


def cache_fp32_parts(qmodel, cfg: AppConfig, trainer):
    device = _device(cfg)
    loader = make_loader(cfg, cfg.data.calib_num, cfg.data.patch_size, cfg.data.crop)
    return trainer.cache_fp32(qmodel, loader, device)


def resolve_stat_image(cfg: AppConfig) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Pick the debug image: INI `stat_image` is a stem or index; empty = first sorted id."""
    ds = Flickr2KSRDataset(
        root=cfg.paths.data_root,
        scale=cfg.model.scale,
        patch_size=cfg.data.eval_patch_size,
        crop="center",
        max_items=None,
        seed=cfg.train.seed,
    )
    key = (cfg.deploy.stat_image or "").strip()
    if not key:
        idx = 0
    elif key in ds.ids:
        idx = ds.ids.index(key)
    elif key.isdigit():
        idx = max(0, min(int(key), len(ds) - 1))
    else:
        hits = [i for i, s in enumerate(ds.ids) if s.startswith(key)]
        idx = hits[0] if hits else 0
        if not hits:
            print(f"[stats] stat_image={key!r} not found, fallback to {ds.ids[0]}")
    item = ds[idx]
    lr = item["lr"]
    hr = item["hr"]
    if lr.ndim == 3:
        lr = lr.unsqueeze(0)
        hr = hr.unsqueeze(0)
    print(f"[stats] using image {item['name']} (index {idx}/{len(ds)})")
    return lr, hr, item["name"]


def dump_debug_stats(step: str, qmodel, fp32_model, cfg: AppConfig, lr: torch.Tensor, image_name: str):
    if not getattr(cfg.deploy, "export_stats", True):
        return
    loader = make_loader(cfg, cfg.data.eval_num, cfg.data.eval_patch_size, "center")
    dump_step_stats(
        step,
        qmodel,
        fp32_model,
        lr.to(_device(cfg)),
        image_name,
        loader,
        _device(cfg),
        cfg.deploy.stat_dir or str(Path(cfg.paths.output_dir) / "stats"),
    )


def reconstruct_act_scales(qmodel, cache, cfg: AppConfig, trainer):
    trainer.reconstruct_act_scales(qmodel, cache, _device(cfg))


def run_src(qmodel, cfg: AppConfig):
    if not getattr(cfg.train, "src", True):
        print("[src] skip: src=false")
        return
    loader = make_loader(cfg, cfg.data.calib_num, cfg.data.patch_size, cfg.data.crop)
    apply_src(
        qmodel,
        loader,
        _device(cfg),
        reg=float(getattr(cfg.train, "src_reg", 1e-3)),
        use_laplacian=bool(getattr(cfg.train, "src_laplacian", True)),
        grain=str(getattr(cfg.train, "src_grain", "layer")),
    )


def init_adaround(qmodel, trainer):
    trainer.init_adaround(qmodel)


def reconstruct_adaround(qmodel, cache, cfg: AppConfig, trainer):
    trainer.reconstruct_adaround(qmodel, cache, _device(cfg))


def reconstruct_parts(qmodel, cache, cfg: AppConfig, trainer):
    trainer.reconstruct(qmodel, cache, _device(cfg))


def fold_weights(qmodel, cfg: AppConfig):
    w_fp32 = snapshot_weights(qmodel)
    fold_adaround_weights(qmodel)
    return w_fp32


def export_params(qmodel, cfg: AppConfig):
    meta = {
        "w_bits": cfg.quant.w_bits,
        "a_bits": cfg.quant.a_bits,
        "weight_symmetric": cfg.quant.weight_symmetric,
        "act_symmetric": cfg.quant.act_symmetric,
        "act_search": cfg.search.enable,
        "search_percentiles": list(cfg.search.percentiles),
        "act_scale_iters": cfg.train.act_scale_iters,
        "fisher": cfg.loss.fisher,
        "fisher_target": cfg.loss.fisher_target,
        "scale": cfg.model.scale,
        "grain": cfg.partition.grain,
        "block_type": cfg.partition.block_type,
        "folded": True,
    }
    export_quant_params(qmodel, cfg.deploy.save_path, meta)
    if cfg.deploy.weights_path:
        export_folded_state_dict(qmodel, cfg.deploy.weights_path)


def export_layer_histograms(qmodel, cfg: AppConfig, weight_fp32: dict):
    if not cfg.deploy.export_ranges:
        return
    device = _device(cfg)
    loader = make_loader(cfg, max(cfg.data.eval_num, 1), cfg.data.eval_patch_size, "center")
    export_histograms(
        qmodel,
        loader,
        cfg.deploy.range_dir,
        device,
        weight_fp32,
        bins=cfg.deploy.hist_bins,
    )


@torch.no_grad()
def evaluate_psnr(qmodel, fp32_model, cfg: AppConfig, enable_all_fq: bool = True, eval_subdir: str = "eval"):
    device = _device(cfg)
    loader = make_loader(cfg, cfg.data.eval_num, cfg.data.eval_patch_size, "center")
    qmodel.eval()
    fp32_model.eval()
    if enable_all_fq:
        set_fakequant(qmodel, True)
    scores = []
    out_dir = Path(cfg.paths.output_dir) / eval_subdir
    for i, batch in enumerate(progress(loader, desc="eval", total=len(loader))):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        q_sr = qmodel(lr)
        fp_sr = fp32_model(lr)
        s_qfp = psnr_quant_vs_fp32(q_sr, fp_sr)
        s_qhr = psnr_vs_hr(q_sr, hr)
        s_fphr = psnr_vs_hr(fp_sr, hr)
        i_qfp = ssim_quant_vs_fp32(q_sr, fp_sr)
        i_qhr = ssim_vs_hr(q_sr, hr)
        i_fphr = ssim_vs_hr(fp_sr, hr)
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else batch["name"]
        print(
            f"[eval] {name}  q_vs_fp32={s_qfp:.3f}/{i_qfp:.4f}  "
            f"q_vs_hr={s_qhr:.3f}/{i_qhr:.4f}  fp32_vs_hr={s_fphr:.3f}/{i_fphr:.4f}"
        )
        scores.append((s_qfp, s_qhr, s_fphr, i_qfp, i_qhr, i_fphr))
        if cfg.eval.save_images:
            save_image(q_sr, out_dir / f"{name}_q.png")
            save_image(fp_sr, out_dir / f"{name}_fp32.png")
            save_image(hr, out_dir / f"{name}_hr.png")
    if scores:
        mean = np.mean(np.array(scores), axis=0)
        print(
            f"[eval] mean  q_vs_fp32={mean[0]:.3f}/{mean[3]:.4f}  "
            f"q_vs_hr={mean[1]:.3f}/{mean[4]:.4f}  fp32_vs_hr={mean[2]:.3f}/{mean[5]:.4f}"
        )
    return scores


def make_partitioner_and_trainer(qmodel, cfg: AppConfig):
    parts = SwinIRPartitioner(cfg.partition.grain, cfg.partition.block_type).split(qmodel)
    print(f"[partition] grain={cfg.partition.grain} block_type={cfg.partition.block_type} n_parts={len(parts)}")
    for i, p in enumerate(parts[:8]):
        print(f"  [{i}] {p.kind:6s} {p.name or '(network)'}")
    if len(parts) > 8:
        print(f"  ... {len(parts) - 8} more")
    trainer = build_trainer(cfg, parts)
    return parts, trainer

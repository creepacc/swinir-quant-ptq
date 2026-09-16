from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..blocks.unit import Part
from ..loss.adaround import AdaRoundRegLoss, adaround_softness
from ..loss.base import CombinedPTQLoss
from ..loss.image import build_image
from ..loss.reconstruction import build_recon
from ..quantize.config import AppConfig
from ..quantize.ops.base import (
    freeze_act_bounds,
    freeze_all,
    init_adaround_all,
    iter_quant_ops,
    set_fakequant,
    set_observer,
)
from ..utils.hooks import FeatureHook, GradHook
from ..utils.progress import progress
from .cache import CalibCache
from .loss_logger import LossLogger


def _move_args(args, device):
    out = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.to(device))
        else:
            out.append(a)
    return tuple(out)


def _alphas(part: Part):
    found = []
    for _, m in part.named_weight_ops():
        if m.weight_fq is not None and m.weight_fq.alpha is not None:
            found.append(m.weight_fq.alpha)
    return found


def _act_clip_params(part: Part, learn: bool):
    params = []
    if not learn:
        return params
    for _, m in part.named_quant_ops():
        for fq in (m.input_fq, m.output_fq):
            if fq is None:
                continue
            lo, hi = fq.enable_learn_bounds()
            params.extend([lo, hi])
    return params


def _feature_modules(model: nn.Module) -> dict:
    found = {}
    for name, m in model.named_modules():
        if getattr(m, "quant_block_kind", None) == "rstb":
            found[name] = m
    if hasattr(model, "conv_after_body") and model.conv_after_body is not None:
        found["conv_after_body"] = model.conv_after_body
    return found


def _attach_feat_hooks(modules: dict):
    store = {}
    handles = []

    def _make(name):
        def fn(_m, _inp, out):
            store[name] = out

        return fn

    for name, m in modules.items():
        handles.append(m.register_forward_hook(_make(name)))
    return store, handles


def _drop_hooks(handles):
    for h in handles:
        h.remove()
    handles.clear()


def _phase_lr(cfg: AppConfig, phase: str) -> float:
    if phase == "act":
        return float(getattr(cfg.train, "act_lr", cfg.train.lr))
    return cfg.train.lr


def _adaround_schedule(cfg: AppConfig, step: int, iters: int) -> tuple[float, bool]:
    """Warmup: no f_reg so h(V) can follow MSE. Then anneal β from high → low (eq. 24)."""
    warm = cfg.loss.adaround_warmup
    start, end = cfg.loss.adaround_beta_start, cfg.loss.adaround_beta_end
    if iters <= 1:
        return end, True
    t = step / max(iters - 1, 1)
    if t < warm:
        return start, False
    u = (t - warm) / max(1.0 - warm, 1e-6)
    return start + (end - start) * u, True


def _fisher_task_loss(sr: torch.Tensor, hr: torch.Tensor, fp32_sr: torch.Tensor, cfg: AppConfig):
    target = hr if getattr(cfg.loss, "fisher_target", "hr") != "fp32" else fp32_sr
    return F.mse_loss(sr, target)


class PTQTrainerBase:
    def __init__(self, cfg: AppConfig, parts: List[Part]):
        self.cfg = cfg
        self.parts = parts
        self.loss_fn = CombinedPTQLoss(
            cfg.loss,
            build_recon(cfg.loss.recon),
            build_image(cfg.loss.image),
            AdaRoundRegLoss(),
        )
        self.act_buf = None
        self.loss_log = LossLogger()

    def _stats_dir(self):
        from pathlib import Path

        d = getattr(self.cfg.deploy, "stat_dir", "") or str(Path(self.cfg.paths.output_dir) / "stats")
        return Path(d)

    def _record_loss(self, phase: str, part: str, step: int, y_q, y_fp, extra):
        loss, recon_w, reg_w, img_w = self.loss_fn.decompose(y_q, y_fp, extra)
        h_soft = 0.0
        beta = extra.get("beta", 0.0) if extra else 0.0
        if extra and extra.get("alphas"):
            h_soft = adaround_softness(extra["alphas"])
        self.loss_log.add(
            phase,
            part,
            step,
            float(loss.detach()),
            float(recon_w.detach()),
            float(reg_w.detach()),
            float(img_w.detach()),
            h_soft=h_soft,
            beta=float(beta) if beta is not None else 0.0,
        )
        return loss

    def _attach_act_buf(self, model: nn.Module, buf):
        for m in iter_quant_ops(model):
            for fq in (m.input_fq, m.output_fq):
                if fq is not None:
                    fq._act_buf = buf

    def _detach_act_buf(self, model: nn.Module):
        for m in iter_quant_ops(model):
            for fq in (m.input_fq, m.output_fq):
                if fq is not None and hasattr(fq, "_act_buf"):
                    delattr(fq, "_act_buf")

    def calibrate(self, model: nn.Module, loader: DataLoader, device: str):
        from ..quantize.scale_search import ActSampleBuf

        model.eval()
        set_observer(model, True)
        set_fakequant(model, False)
        self.act_buf = None
        search_cfg = getattr(self.cfg, "search", None)
        if search_cfg is not None and search_cfg.enable:
            self.act_buf = ActSampleBuf(search_cfg.max_samples)
            self._attach_act_buf(model, self.act_buf)
            print(f"[calibrate] collect act samples for percentile search (max={search_cfg.max_samples})")
        n = 0
        with torch.no_grad():
            for batch in progress(loader, desc="calibrate", total=len(loader)):
                lr = batch["lr"].to(device)
                model(lr)
                n += 1
        self._detach_act_buf(model)
        print(f"[calibrate] done, {n} batches")

    def freeze(self, model: nn.Module):
        freeze_all(
            model,
            adaround=self.cfg.train.adaround,
            search_cfg=getattr(self.cfg, "search", None),
            act_buf=self.act_buf,
            init_adaround=False,
        )
        print("[freeze] scales written, fakequant enabled (AdaRound alpha deferred)")

    def init_adaround(self, model: nn.Module):
        init_adaround_all(model)

    def cache_fp32(self, model: nn.Module, loader: DataLoader, device: str) -> CalibCache:
        model.eval()
        set_fakequant(model, False)
        for p in model.parameters():
            p.requires_grad_(False)
        hook_map = {p.name or "network": p.module for p in self.parts}
        feat_mods = _feature_modules(model)
        for n, m in feat_mods.items():
            if m not in hook_map.values():
                hook_map[f"feat:{n}"] = m
        fhook = FeatureHook(hook_map)
        want_fisher = bool(getattr(self.cfg.loss, "fisher", False))
        ghook = GradHook(hook_map) if want_fisher else None
        cache = CalibCache()
        n_fisher = 0

        def _store_image(lr, hr, sr, batch):
            cache.images.append(
                {
                    "lr": lr.detach().cpu(),
                    "hr": hr.detach().cpu(),
                    "fp32_sr": sr.detach().cpu(),
                    "name": batch.get("name", [""])[0]
                    if isinstance(batch.get("name"), (list, tuple))
                    else batch.get("name", ""),
                }
            )

        if ghook is None:
            with fhook, torch.no_grad():
                for batch in progress(loader, desc="cache fp32", total=len(loader)):
                    lr = batch["lr"].to(device)
                    hr = batch["hr"].to(device)
                    sr = model(lr)
                    _store_image(lr, hr, sr, batch)
        else:
            with _HookPair(fhook, ghook), torch.enable_grad():
                for batch in progress(loader, desc="cache fp32+fisher", total=len(loader)):
                    model.zero_grad(set_to_none=True)
                    lr = batch["lr"].to(device)
                    lr.requires_grad_(True)
                    hr = batch["hr"].to(device)
                    sr = model(lr)
                    _store_image(lr, hr, sr, batch)
                    task = _fisher_task_loss(sr, hr, sr, self.cfg)
                    task.backward()
                    img_g = ghook.grads.get("network")
                    if img_g is not None:
                        cache.images[-1]["fisher"] = img_g.detach().cpu()
                    for name, recs in fhook.records.items():
                        if not recs:
                            continue
                        g = ghook.grads.get(name)
                        recs[-1]["fisher"] = g.detach().cpu() if g is not None else None
                        if g is not None:
                            n_fisher += 1
        for name, recs in fhook.records.items():
            if name.startswith("feat:"):
                key = name[5:]
                for i, rec in enumerate(recs):
                    if i < len(cache.images):
                        cache.images[i].setdefault("feats", {})[key] = rec["output"]
                continue
            cache.parts[name] = recs
        set_fakequant(model, True)
        n_feat = sum(1 for k in fhook.records if k.startswith("feat:"))
        extra = f", fisher={n_fisher} tensors" if want_fisher else ""
        extra += f", dqc_feats={n_feat}" if n_feat else ""
        print(f"[cache] stored {len(cache.images)} images, {len(cache.parts)} parts{extra}")
        return cache

    def reconstruct_act_scales(self, model: nn.Module, cache: CalibCache, device: str):
        if not self.cfg.train.learn_act_scale:
            print("[reconstruct] phase A skip: learn_act_scale=false")
            return
        iters = int(getattr(self.cfg.train, "act_scale_iters", self.cfg.train.iters))
        if iters <= 0:
            print("[reconstruct] phase A skip: act_scale_iters=0")
            return
        dqc = bool(getattr(self.cfg.loss, "dqc", True))
        print(
            f"[reconstruct] phase A: 2DQuant DQC clip bounds, iters={iters} "
            f"lr={getattr(self.cfg.train, 'act_lr', self.cfg.train.lr)} dqc={dqc} "
            f"λ_feat={getattr(self.cfg.loss, 'lambda_feat', 1.0)}"
        )
        self._reconstruct(model, cache, device, phase="act", iters=iters)
        n_b = freeze_act_bounds(model)
        print(f"[reconstruct] phase A freeze {n_b} act clip bounds → scale/zp")
        self.loss_log.save(self._stats_dir(), phase="act")

    def reconstruct_adaround(self, model: nn.Module, cache: CalibCache, device: str):
        if not self.cfg.train.adaround:
            print("[reconstruct] phase B skip: adaround=false")
            return
        print(
            f"[reconstruct] phase B: AdaRound eq.21/24, iters={self.cfg.train.iters} "
            f"warmup={self.cfg.loss.adaround_warmup} "
            f"β={self.cfg.loss.adaround_beta_start}→{self.cfg.loss.adaround_beta_end} "
            f"λ_reg={self.cfg.loss.lambda_reg}"
        )
        self._reconstruct(model, cache, device, phase="adaround", iters=self.cfg.train.iters)
        self.loss_log.save(self._stats_dir(), phase="adaround")

    def reconstruct(self, model: nn.Module, cache: CalibCache, device: str):
        self.reconstruct_act_scales(model, cache, device)
        if self.cfg.train.adaround:
            self.init_adaround(model)
            self.reconstruct_adaround(model, cache, device)

    def _reconstruct(self, model: nn.Module, cache: CalibCache, device: str, phase: str, iters: int):
        raise NotImplementedError


class _HookPair:
    def __init__(self, fhook: FeatureHook, ghook: GradHook):
        self.fhook = fhook
        self.ghook = ghook

    def __enter__(self):
        self.fhook.register()
        self.ghook.register()
        return self

    def __exit__(self, *exc):
        self.ghook.close()
        self.fhook.close()


class LayerWiseTrainer(PTQTrainerBase):
    def _reconstruct(self, model: nn.Module, cache: CalibCache, device: str, phase: str, iters: int):
        return _reconstruct_parts(self, model, cache, device, phase, iters)


class BlockWiseTrainer(PTQTrainerBase):
    def _reconstruct(self, model: nn.Module, cache: CalibCache, device: str, phase: str, iters: int):
        return _reconstruct_parts(self, model, cache, device, phase, iters)


class NetworkWiseTrainer(PTQTrainerBase):
    def _reconstruct(self, model: nn.Module, cache: CalibCache, device: str, phase: str, iters: int):
        cfg = self.cfg
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        part = self.parts[0]
        if phase == "act":
            params = _act_clip_params(part, True)
        else:
            params = _alphas(part)
        if not params:
            print(f"[reconstruct] network {phase}: no trainable quant params, skip")
            return
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=_phase_lr(cfg, phase))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1)) if phase == "act" else None
        imgs = cache.images
        use_fisher = phase == "adaround" and bool(getattr(cfg.loss, "fisher", False))
        use_dqc = phase == "act" and bool(getattr(cfg.loss, "dqc", True))
        feat_mods = _feature_modules(model) if use_dqc else {}
        q_feats, feat_handles = _attach_feat_hooks(feat_mods) if feat_mods else ({}, [])
        print(f"[reconstruct] START network-wise phase={phase}, {len(imgs)} images, iters={iters}")
        bar = progress(range(iters), desc=f"network {phase}", total=iters)
        try:
            for step in bar:
                rec = imgs[step % len(imgs)]
                lr = rec["lr"].to(device)
                target = rec["fp32_sr"].to(device) if cfg.loss.image_target == "fp32" else rec["hr"].to(device)
                q_sr = model(lr)
                extra = {"use_fisher": use_fisher}
                if use_fisher and rec.get("fisher") is not None:
                    extra["fisher"] = rec["fisher"].to(device)
                if use_dqc:
                    extra["dqc"] = True
                    extra["q_feats"] = q_feats
                    extra["fp_feats"] = rec.get("feats") or {}
                if phase == "adaround":
                    beta, apply_reg = _adaround_schedule(cfg, step, iters)
                    extra["alphas"] = _alphas(part)
                    extra["beta"] = beta
                    extra["apply_reg"] = apply_reg
                    extra["recon_reduction"] = "sum"
                loss = self._record_loss(phase, "network", step + 1, q_sr, target, extra)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                row = self.loss_log.rows[-1] if self.loss_log.rows else {}
                postfix = {"loss": f"{loss.item():.4g}"}
                if phase == "adaround":
                    postfix.update(
                        recon=f"{row.get('recon', 0):.3g}",
                        reg=f"{row.get('reg', 0):.3g}",
                        h=f"{row.get('h_soft', 0):.3f}",
                        b=f"{row.get('beta', 0):.1f}",
                    )
                elif use_dqc:
                    postfix.update(L1=f"{row.get('recon', 0):.3g}", feat=f"{row.get('image', 0):.2g}")
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(postfix, refresh=False)
        finally:
            _drop_hooks(feat_handles)
        for p in params:
            p.requires_grad_(False)
        print(f"[reconstruct] network-wise phase={phase} done")


def _reconstruct_parts(
    trainer: PTQTrainerBase,
    model: nn.Module,
    cache: CalibCache,
    device: str,
    phase: str,
    iters: int,
):
    cfg = trainer.cfg
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    use_fisher = phase == "adaround" and bool(getattr(cfg.loss, "fisher", False))
    n_parts = len(trainer.parts)
    part_bar = progress(list(enumerate(trainer.parts)), desc=f"{phase} parts", total=n_parts)
    for i, part in part_bar:
        key = part.name or "network"
        recs = cache.parts.get(key, [])
        if not recs:
            print(f"[reconstruct] skip empty part {key}")
            continue
        params = _act_clip_params(part, True) if phase == "act" else _alphas(part)
        if not params:
            print(f"[reconstruct] {i + 1}/{n_parts} {key} {phase}: no trainable params, skip")
            continue
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=_phase_lr(cfg, phase))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1)) if phase == "act" else None
        if hasattr(part_bar, "set_postfix"):
            part_bar.set_postfix(name=key, kind=part.kind)
        step_bar = progress(range(iters), desc=f"{phase} {key}", total=iters)
        for step in step_bar:
            rec = recs[step % len(recs)]
            args = _move_args(rec["input"], device)
            y_fp = rec["output"]
            if torch.is_tensor(y_fp):
                y_fp = y_fp.to(device)
            y_q = part.module(*args)
            extra = {"use_fisher": use_fisher}
            if use_fisher and rec.get("fisher") is not None:
                extra["fisher"] = rec["fisher"].to(device)
            if phase == "adaround":
                beta, apply_reg = _adaround_schedule(cfg, step, iters)
                extra["alphas"] = _alphas(part)
                extra["beta"] = beta
                extra["apply_reg"] = apply_reg
                extra["recon_reduction"] = "sum"
            loss = trainer._record_loss(phase, key, step + 1, y_q, y_fp, extra)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            row = trainer.loss_log.rows[-1] if trainer.loss_log.rows else {}
            postfix = {"loss": f"{loss.item():.4g}"}
            if phase == "adaround":
                postfix.update(
                    recon=f"{row.get('recon', 0):.3g}",
                    h=f"{row.get('h_soft', 0):.3f}",
                )
            if hasattr(step_bar, "set_postfix"):
                step_bar.set_postfix(postfix, refresh=False)
        for p in params:
            p.requires_grad_(False)
    print(f"[reconstruct] phase={phase} all parts finished")


def build_trainer(cfg: AppConfig, parts: List[Part]) -> PTQTrainerBase:
    grain = cfg.partition.grain
    if grain == "network":
        return NetworkWiseTrainer(cfg, parts)
    if grain == "layer":
        return LayerWiseTrainer(cfg, parts)
    return BlockWiseTrainer(cfg, parts)

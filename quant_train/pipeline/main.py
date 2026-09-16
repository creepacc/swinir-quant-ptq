"""
SwinIR PTQ 入口。按 Step 0 -> 4 往下读即可看完整过程。

启动:
    python -m quant_train.pipeline.main --config quant_train/configs/default.ini
"""

from __future__ import annotations

import argparse

from . import steps


def parse_args():
    p = argparse.ArgumentParser(description="SwinIR PTQ / BRECQ pipeline")
    p.add_argument("--config", required=True, help="INI 超参数文件")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 0  读 INI（路径 / 量化开关 / 分块 / 训练 / 损失 / deploy）
    # ------------------------------------------------------------------
    cfg = steps.load_config(args.config)
    print(f"[step0] config={args.config}")
    print(f"[step0] scale=x{cfg.model.scale} grain={cfg.partition.grain} "
          f"device={cfg.train.device} quant_params={cfg.paths.quant_params or '(train)'}")
    print(f"[step0] weight_sym={cfg.quant.weight_symmetric} act_sym={cfg.quant.act_symmetric} "
          f"act_search={cfg.search.enable} method={cfg.search.method} dobi_k={cfg.search.dobi_k}")
    print(f"[step0] softmax_mode={cfg.quant.softmax_mode} "
          f"skip_attn_act={cfg.quant.skip_attn_act} skip_residual_act={cfg.quant.skip_residual_act} "
          f"residual_fix={cfg.quant.residual_fix} alpha={cfg.quant.smooth_alpha} "
          f"adaround={cfg.train.adaround} src={cfg.train.src} "
          f"src_grain={getattr(cfg.train, 'src_grain', 'layer')}")
    print(f"[step0] grain={cfg.partition.grain}/{cfg.partition.block_type} "
          f"act_scale_iters={cfg.train.act_scale_iters} adaround_iters={cfg.train.iters} "
          f"fisher={cfg.loss.fisher} fisher_target={cfg.loss.fisher_target}")
    print(f"[step0] calib_num={cfg.data.calib_num} eval_num={cfg.data.eval_num} "
          f"patch={cfg.data.patch_size}")

    # ------------------------------------------------------------------
    # Step 1  建原版 FP32 SwinIR 并加载预训练权重
    # ------------------------------------------------------------------
    fp32 = steps.build_fp32_swinir(cfg)
    print("[step1] FP32 SwinIR ready")

    # ------------------------------------------------------------------
    # Step 2  包装伪量化节点（Linear / Conv / ewise / 激活）
    # ------------------------------------------------------------------
    qmodel = steps.prepare_quant_model(fp32, cfg)
    parts, trainer = steps.make_partitioner_and_trainer(qmodel, cfg)
    print("[step2] quant model + partitioner ready")

    if cfg.paths.quant_params:
        # --------------------------------------------------------------
        # Step 3a  INI 已指定量化参数文件：直接加载，可跳过校准/训练
        # --------------------------------------------------------------
        print(f"[step3a] load quant params: {cfg.paths.quant_params}")
        steps.load_quant_params(qmodel, cfg)
        if not cfg.deploy.skip_calib_if_loaded:
            steps.calibrate_observers(qmodel, cfg, trainer)
            steps.freeze_qparams(qmodel, trainer)
        if not cfg.deploy.skip_train_if_loaded:
            cache = steps.cache_fp32_parts(qmodel, cfg, trainer)
            stat_lr, _, stat_name = steps.resolve_stat_image(cfg)
            steps.dump_debug_stats("before_act", qmodel, fp32, cfg, stat_lr, stat_name)
            steps.reconstruct_act_scales(qmodel, cache, cfg, trainer)
            steps.dump_debug_stats("before_adaround", qmodel, fp32, cfg, stat_lr, stat_name)
            steps.run_src(qmodel, cfg)
            if cfg.train.src:
                steps.dump_debug_stats("after_src", qmodel, fp32, cfg, stat_lr, stat_name)
            if cfg.train.adaround:
                steps.init_adaround(qmodel, trainer)
                steps.reconstruct_adaround(qmodel, cache, cfg, trainer)
        print("[step3a] fold AdaRound -> hard round, write back weights")
        weight_fp32 = steps.fold_weights(qmodel, cfg)
        if not cfg.deploy.skip_train_if_loaded:
            steps.dump_debug_stats("after_train", qmodel, fp32, cfg, stat_lr, stat_name)
        print("[step3a] export quant json / folded pth / histograms")
        steps.export_params(qmodel, cfg)
        steps.export_layer_histograms(qmodel, cfg, weight_fp32)
    else:
        # --------------------------------------------------------------
        # Step 3b  正常 PTQ：校准 -> 缓存 FP32 块 I/O -> freeze -> 重建
        #          （先缓存再 freeze，保证缓存的是伪量化关闭时的 FP32 输出）
        # --------------------------------------------------------------
        print("[step3b-0] residual reparam (SmoothQuant/OS+/QuaRot), then recalibrate")
        steps.apply_stream_reparam(qmodel, cfg)

        print("[step3b-1] calibrate observers")
        steps.calibrate_observers(qmodel, cfg, trainer)

        print("[step3b-2] cache FP32 part inputs/outputs + Fisher grads")
        cache = steps.cache_fp32_parts(qmodel, cfg, trainer)

        print("[step3b-3] freeze: weight minmax, act DOBI clip search, enable fakequant")
        steps.freeze_qparams(qmodel, trainer)

        stat_lr, _, stat_name = steps.resolve_stat_image(cfg)
        print("[stats] before phase A")
        steps.dump_debug_stats("before_act", qmodel, fp32, cfg, stat_lr, stat_name)

        print("[step3b-4] phase A: 2DQuant DQC (train act clip bounds, L1 + feature distill)")
        steps.reconstruct_act_scales(qmodel, cache, cfg, trainer)

        print("[stats] after phase A")
        steps.dump_debug_stats("before_adaround", qmodel, fp32, cfg, stat_lr, stat_name)

        print("[step3b-4b] HarmoQ SRC: closed-form weight correction")
        steps.run_src(qmodel, cfg)
        if cfg.train.src:
            steps.dump_debug_stats("after_src", qmodel, fp32, cfg, stat_lr, stat_name)

        if cfg.train.adaround:
            print("[step3b-5] init AdaRound alpha")
            steps.init_adaround(qmodel, trainer)
            print("[step3b-6] phase B: AdaRound + Fisher-weighted recon (scales frozen)")
            steps.reconstruct_adaround(qmodel, cache, cfg, trainer)
        else:
            print("[step3b-5] skip AdaRound (adaround=false)")

        print("[step3b-7] fold weights -> hard round, write back, drop alpha")
        weight_fp32 = steps.fold_weights(qmodel, cfg)

        print("[stats] after train")
        steps.dump_debug_stats("after_train", qmodel, fp32, cfg, stat_lr, stat_name)

        print("[step3b-8] export quant json / folded pth / per-tensor histograms")
        steps.export_params(qmodel, cfg)
        steps.export_layer_histograms(qmodel, cfg, weight_fp32)

    # ------------------------------------------------------------------
    # Step 4  评估：量化 SR vs FP32 SR，以及 vs Flickr2K HR
    # ------------------------------------------------------------------
    print("[step4] evaluate PSNR / SSIM")
    steps.evaluate_psnr(qmodel, fp32, cfg)
    print("[done] pipeline finished")


if __name__ == "__main__":
    main()

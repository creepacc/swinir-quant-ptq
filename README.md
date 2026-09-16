# SwinIR W8A8 PTQ

SwinIR-M ×4 的 PyTorch 伪量化训练 / 校准代码（不包含跑出来的 PSNR 图、CSV 和 `outputs*`）。

依赖原版 SwinIR 网络与权重（`sr_root` / `weights`），以及 Flickr2K 校准数据（`data_root`）。请在 INI 的 `[paths]` 里改成本机路径。

## 布局

```
quant_train/
  pipeline/     PTQ 入口（校准 → freeze → 可选 AdaRound/SRC → eval）
  quantize/     FakeQuant、observer、DoBi、SmoothQuant/OS+、QuaRot 旋转
  train/        块重建、AdaRound、SRC
  configs/      INI 超参
  tools/        残差分析脚本
  utils/        数据与指标
```

## 运行

在仓库根目录：

```bash
python -m quant_train.pipeline.main --config quant_train/configs/default.ini
```

残差 INT8 相关配置：

- `quant_train/configs/residual_study.ini` — 基线（残差仍 INT8，DoBi）
- `quant_train/configs/residual_m1_smooth.ini` — SmoothQuant + OS+
- `quant_train/configs/residual_m2_rotate.ini` — QuaRot 式正交旋转
- `quant_train/configs/residual_m12_smooth_rotate.ini` — 旋转后再平滑
- `quant_train/configs/residual_m3_clip.ini` — 残差 Add 单独百分位 clip

```bash
python -m quant_train.tools.run_residual_fixes
python -m quant_train.tools.analyze_residual --config quant_train/configs/residual_study.ini
```

## 量化方案（代码默认）

- 权重 INT8 per-channel 对称；激活 INT8 per-tensor 非对称
- Linear/Conv 量化输入；残差 `QuantAdd` 量化输出
- Softmax / LayerNorm 默认 FP32
- 激活 bound 搜索默认 DoBi

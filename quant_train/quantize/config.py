"""INI -> typed config. Missing keys fall back to defaults."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_int_list(v: str) -> List[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def _as_float_list(v: str) -> List[float]:
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def _as_float_set(v: str) -> set:
    return {x.strip() for x in v.split(",") if x.strip()}


def _opt_path(v: str) -> Optional[str]:
    v = (v or "").strip()
    return v if v else None


@dataclass
class PathsConfig:
    sr_root: str
    weights: str
    data_root: str
    output_dir: str
    quant_params: Optional[str] = None


@dataclass
class ModelConfig:
    scale: int = 4
    window_size: int = 8
    embed_dim: int = 180
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6)
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6)
    mlp_ratio: float = 2.0
    upsampler: str = "pixelshuffle"
    resi_connection: str = "1conv"
    img_size: int = 64
    img_range: float = 1.0


@dataclass
class QuantConfig:
    w_bits: int = 8
    a_bits: int = 8
    weight_symmetric: bool = True
    act_symmetric: bool = False
    per_channel_weight: bool = True
    quantize_weight: bool = True
    quantize_input: bool = True
    quantize_output: bool = False
    fp32_ops: set = field(default_factory=lambda: {"softmax", "layer_norm"})
    observer: str = "minmax"
    qmin: int = -128
    qmax: int = 127
    act_backend: str = "fakequant_out"
    min_scale: float = 1e-9
    # default: Softmax 计算保持 FP32，但前后邻接算子会量化其输入/输出
    # fp32: Softmax 输入输出均不量化
    # lut: Softmax 输入 8bit 后用 exp LUT + 归一化，输出再量化
    softmax_mode: str = "default"
    # 关掉注意力计算路径上的激活伪量化（QK/AV matmul、scale mul、bias/mask add）
    skip_attn_act: bool = False
    # 关掉残差 Add 输出伪量化（block attn/mlp、RSTB、网络 skip）
    skip_residual_act: bool = False
    residual_fix: str = "none"
    smooth_alpha: float = 0.5
    rotate_seed: int = 0


@dataclass
class SearchConfig:
    enable: bool = True
    method: str = "dobi"
    dobi_k: int = 100
    percentiles: Sequence[float] = (99.0, 99.5, 99.9, 99.95, 99.99, 100.0)
    max_samples: int = 50000
    residual_percentiles: Sequence[float] = ()
    residual_method: str = ""


@dataclass
class DataConfig:
    calib_num: int = 8
    eval_num: int = 2
    patch_size: int = 64
    eval_patch_size: int = 64
    crop: str = "random"
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class PartitionConfig:
    grain: str = "block"
    block_type: str = "swin_block"


@dataclass
class TrainConfig:
    iters: int = 50
    act_scale_iters: int = 50
    lr: float = 1e-3
    act_lr: float = 1e-2
    optimizer: str = "adam"
    seed: int = 0
    device: str = "cuda"
    learn_act_scale: bool = True
    adaround: bool = True
    src: bool = True
    src_reg: float = 1e-3
    src_laplacian: bool = True
    src_grain: str = "layer"
    log_interval: int = 10


@dataclass
class LossConfig:
    recon: str = "mse"
    image: str = "l1"
    image_target: str = "fp32"
    lambda_recon: float = 1.0
    lambda_img: float = 0.0
    lambda_feat: float = 1.0
    lambda_reg: float = 0.01
    dqc: bool = True
    adaround_beta_start: float = 20.0
    adaround_beta_end: float = 2.0
    adaround_warmup: float = 0.2
    fisher: bool = True
    fisher_target: str = "hr"


@dataclass
class EvalConfig:
    save_images: bool = True


@dataclass
class DeployConfig:
    save_path: str = ""
    weights_path: str = ""
    range_dir: str = ""
    export_ranges: bool = True
    hist_bins: int = 64
    format: str = "json"
    skip_calib_if_loaded: bool = True
    skip_train_if_loaded: bool = True
    export_stats: bool = True
    stat_dir: str = ""
    stat_image: str = ""


@dataclass
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    quant: QuantConfig
    search: SearchConfig
    data: DataConfig
    partition: PartitionConfig
    train: TrainConfig
    loss: LossConfig
    eval: EvalConfig
    deploy: DeployConfig

    @property
    def device(self) -> str:
        return self.train.device


def load_ini(path: str | Path) -> AppConfig:
    ini = Path(path).expanduser().resolve()
    if not ini.is_file():
        raise FileNotFoundError(f"INI not found: {ini}")

    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.read(ini, encoding="utf-8")

    def g(sec: str, key: str, default: str = "") -> str:
        if parser.has_option(sec, key):
            return parser.get(sec, key).strip()
        return default

    paths = PathsConfig(
        sr_root=g("paths", "sr_root"),
        weights=g("paths", "weights"),
        data_root=g("paths", "data_root"),
        output_dir=g("paths", "output_dir"),
        quant_params=_opt_path(g("paths", "quant_params")),
    )
    model = ModelConfig(
        scale=int(g("model", "scale", "4")),
        window_size=int(g("model", "window_size", "8")),
        embed_dim=int(g("model", "embed_dim", "180")),
        depths=_as_int_list(g("model", "depths", "6,6,6,6,6,6")),
        num_heads=_as_int_list(g("model", "num_heads", "6,6,6,6,6,6")),
        mlp_ratio=float(g("model", "mlp_ratio", "2")),
        upsampler=g("model", "upsampler", "pixelshuffle"),
        resi_connection=g("model", "resi_connection", "1conv"),
        img_size=int(g("model", "img_size", "64")),
        img_range=float(g("model", "img_range", "1.0")),
    )
    quant = QuantConfig(
        w_bits=int(g("quant", "w_bits", "8")),
        a_bits=int(g("quant", "a_bits", "8")),
        weight_symmetric=_as_bool(g("quant", "weight_symmetric", g("quant", "symmetric", "true"))),
        act_symmetric=_as_bool(g("quant", "act_symmetric", "false")),
        per_channel_weight=_as_bool(g("quant", "per_channel_weight", "true")),
        quantize_weight=_as_bool(g("quant", "quantize_weight", "true")),
        quantize_input=_as_bool(g("quant", "quantize_input", "true")),
        quantize_output=_as_bool(g("quant", "quantize_output", "false")),
        fp32_ops=_as_float_set(g("quant", "fp32_ops", "softmax,layer_norm")),
        observer=g("quant", "observer", "minmax"),
        qmin=int(g("quant", "qmin", "-128")),
        qmax=int(g("quant", "qmax", "127")),
        act_backend=g("quant", "act_backend", "fakequant_out"),
        min_scale=float(g("quant", "min_scale", "1e-9")),
        softmax_mode=g("quant", "softmax_mode", "default").lower(),
        skip_attn_act=_as_bool(g("quant", "skip_attn_act", "false")),
        skip_residual_act=_as_bool(g("quant", "skip_residual_act", "false")),
        residual_fix=g("quant", "residual_fix", "none").strip().lower(),
        smooth_alpha=float(g("quant", "smooth_alpha", "0.5")),
        rotate_seed=int(g("quant", "rotate_seed", "0")),
    )
    search = SearchConfig(
        enable=_as_bool(g("search", "enable", "true")),
        method=g("search", "method", "dobi"),
        dobi_k=int(g("search", "dobi_k", "100")),
        percentiles=tuple(_as_float_list(g("search", "percentiles", "99.0,99.5,99.9,99.95,99.99,100"))),
        max_samples=int(g("search", "max_samples", "50000")),
        residual_percentiles=tuple(_as_float_list(g("search", "residual_percentiles", "")))
        if g("search", "residual_percentiles", "").strip()
        else (),
        residual_method=g("search", "residual_method", "").strip().lower(),
    )
    data = DataConfig(
        calib_num=int(g("data", "calib_num", "8")),
        eval_num=int(g("data", "eval_num", "2")),
        patch_size=int(g("data", "patch_size", "64")),
        eval_patch_size=int(g("data", "eval_patch_size", "64")),
        crop=g("data", "crop", "random"),
        batch_size=int(g("data", "batch_size", "1")),
        num_workers=int(g("data", "num_workers", "0")),
    )
    partition = PartitionConfig(
        grain=g("partition", "grain", "block"),
        block_type=g("partition", "block_type", "swin_block"),
    )
    train = TrainConfig(
        iters=int(g("train", "iters", "50")),
        act_scale_iters=int(g("train", "act_scale_iters", g("train", "iters", "50"))),
        lr=float(g("train", "lr", "1e-3")),
        act_lr=float(g("train", "act_lr", "1e-2")),
        optimizer=g("train", "optimizer", "adam"),
        seed=int(g("train", "seed", "0")),
        device=g("train", "device", "cuda"),
        learn_act_scale=_as_bool(g("train", "learn_act_scale", "true")),
        adaround=_as_bool(g("train", "adaround", "true")),
        src=_as_bool(g("train", "src", "true")),
        src_reg=float(g("train", "src_reg", "1e-3")),
        src_laplacian=_as_bool(g("train", "src_laplacian", "true")),
        src_grain=g("train", "src_grain", "layer").lower(),
        log_interval=int(g("train", "log_interval", "10")),
    )
    loss = LossConfig(
        recon=g("loss", "recon", "mse"),
        image=g("loss", "image", "l1"),
        image_target=g("loss", "image_target", "fp32"),
        lambda_recon=float(g("loss", "lambda_recon", "1.0")),
        lambda_img=float(g("loss", "lambda_img", "0.0")),
        lambda_feat=float(g("loss", "lambda_feat", "1.0")),
        lambda_reg=float(g("loss", "lambda_reg", "0.01")),
        dqc=_as_bool(g("loss", "dqc", "true")),
        adaround_beta_start=float(g("loss", "adaround_beta_start", "20")),
        adaround_beta_end=float(g("loss", "adaround_beta_end", "2")),
        adaround_warmup=float(g("loss", "adaround_warmup", "0.2")),
        fisher=_as_bool(g("loss", "fisher", "true")),
        fisher_target=g("loss", "fisher_target", "hr"),
    )
    ev = EvalConfig(save_images=_as_bool(g("eval", "save_images", "true")))
    deploy = DeployConfig(
        save_path=g("deploy", "save_path", str(Path(paths.output_dir) / "swinir_w8a8_quant.json")),
        weights_path=g("deploy", "weights_path", str(Path(paths.output_dir) / "swinir_w8a8_folded.pth")),
        range_dir=g("deploy", "range_dir", str(Path(paths.output_dir) / "ranges")),
        export_ranges=_as_bool(g("deploy", "export_ranges", "true")),
        hist_bins=int(g("deploy", "hist_bins", "64")),
        format=g("deploy", "format", "json"),
        skip_calib_if_loaded=_as_bool(g("deploy", "skip_calib_if_loaded", "true")),
        skip_train_if_loaded=_as_bool(g("deploy", "skip_train_if_loaded", "true")),
        export_stats=_as_bool(g("deploy", "export_stats", "true")),
        stat_dir=g("deploy", "stat_dir", str(Path(paths.output_dir) / "stats")),
        stat_image=g("deploy", "stat_image", ""),
    )
    return AppConfig(paths, model, quant, search, data, partition, train, loss, ev, deploy)

"""ONNX エクスポート（scripts/export_onnx.py 準拠）と ONNX Runtime セッション生成。

エクスポート形式:
    single: inputs ["patch", "is_upper"] → output ["t"]
    dual:   inputs ["patch_a", "patch_b", "is_upper"] (+ aux_dim>0 で ["aux_features"])
            → output ["t"]（DualBenchmarkEnsemble で単一出力化）
    opset 17 / dynamic batch / constant folding — scripts/export_onnx.py と同一パラメータ。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tools.model_benchmark.inputs import DEFAULT_PATCH_SIZE, aux_dim_of
from tools.model_benchmark.loader import DualBenchmarkEnsemble, ModelFamily

logger = logging.getLogger(__name__)

OPSET_VERSION = 17
SINGLE_INPUT_NAMES = ("patch", "is_upper")
DUAL_INPUT_NAMES = ("patch_a", "patch_b", "is_upper")
OUTPUT_NAME = "t"


def _dummy_args(
    family: ModelFamily,
    aux_dim: int,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """エクスポート用の dummy 入力（batch=1）と入力名の組を返す。"""
    h, w = patch_size
    is_upper = torch.zeros(1, 1)
    if family == "dual":
        args: tuple[torch.Tensor, ...] = (
            torch.zeros(1, 3, h, w),
            torch.zeros(1, 3, h, w),
            is_upper,
        )
        names = list(DUAL_INPUT_NAMES)
        if aux_dim > 0:
            args = args + (torch.zeros(1, aux_dim),)
            names.append("aux_features")
        return args, names
    return (torch.zeros(1, 3, h, w), is_upper), list(SINGLE_INPUT_NAMES)


def export_onnx(
    model: nn.Module,
    family: ModelFamily,
    output_path: Path,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
) -> Path:
    """モデルを ONNX にエクスポートして検証する。

    エクスポートは常に CPU 上で trace する（scripts/export_onnx.py 準拠）。
    呼び出し時のデバイス配置は維持する。

    patch_size: ダミー入力の空間サイズ (H, W)。dynamic_axes は batch 軸のみ動的指定
    なので、ここで指定したサイズが ONNX グラフに固定形状として焼き込まれる。
    既定値は現行dual標準・EXP032モデルの実際の学習パッチサイズ（DEFAULT_PATCH_SIZE参照）。
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx が見つかりません。`pip install onnx` でインストールしてください。"
        ) from exc

    module = model
    if family == "dual" and not isinstance(model, DualBenchmarkEnsemble):
        module = DualBenchmarkEnsemble([model])
    device_before = next(module.parameters(), torch.empty(0)).device
    module = module.to("cpu")
    args, names = _dummy_args(family, aux_dim_of(module), patch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes: dict[str, dict[int, str]] = {name: {0: "batch"} for name in names}
    dynamic_axes[OUTPUT_NAME] = {0: "batch"}
    try:
        torch.onnx.export(
            module,
            args,
            str(output_path),
            input_names=names,
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    finally:
        module = module.to(device_before)
    onnx.checker.check_model(onnx.load(str(output_path)))
    logger.info("ONNX モデルをエクスポートしました: %s", output_path)
    return output_path


def quantize_dynamic_onnx(onnx_path: Path, output_path: Path) -> Path:
    """既存の ONNX モデルを動的量子化（重みINT8、活性化は実行時量子化）する。

    キャリブレーションデータ不要（onnxruntime.quantization.quantize_dynamic）。
    MatMul（regressorのLinear層）のみをINT8化する。Conv も対象にすると
    ConvInteger opがこの環境のCPUExecutionProviderで未実装（NOT_IMPLEMENTED）
    のため実行時エラーになる（onnxruntime 1.22.1で確認）。
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "onnxruntime.quantization が見つかりません。`pip install onnxruntime` を確認してください。"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(onnx_path), str(output_path),
        op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
    )
    logger.info("ONNX モデルを動的量子化(INT8, MatMulのみ)しました: %s", output_path)
    return output_path


def quantize_static_onnx(
    onnx_path: Path,
    output_path: Path,
    calibration_data_reader: Any,
    op_types_to_quantize: list[str] | None = None,
) -> Path:
    """既存の ONNX モデルを静的量子化（QDQ形式、キャリブレーションデータ必要）する。

    公式ドキュメント（onnxruntime.ai/docs/performance/model-optimizations/quantization.html）:
    "it is recommended to use dynamic quantization for RNNs and transformer-based models,
    and static quantization for CNN models"。動的量子化のConvIntegerがCPUExecutionProvider
    で未実装だった問題（quantize_dynamic_onnx参照）を回避するため、CNNであるこのモデル群には
    QDQ形式の静的量子化を使う（QuantizeLinear/DequantizeLinearノード対を挿入する形式で、
    ConvIntegerのような専用整数opを使わないためCPU実行時の対応が広い）。
    """
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise ImportError(
            "onnxruntime.quantization が見つかりません。`pip install onnxruntime` を確認してください。"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(onnx_path), str(output_path),
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types_to_quantize,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    logger.info("ONNX モデルを静的量子化(QDQ, INT8)しました: %s", output_path)
    return output_path


def create_session(path: Path, prefer_cuda: bool, threads: int | None) -> tuple[Any, str]:
    """ONNX Runtime の InferenceSession を生成する。

    providers は CPU が基本。prefer_cuda かつ CUDAExecutionProvider が
    利用可能な場合のみ GPU を優先する。

    Returns:
        (session, device_label): device_label は実際に使用された実行デバイス
        ("cuda" / "cpu")。CUDA を要求したが利用できない場合は CPU に
        フォールバックし、その旨を警告する。
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime が見つかりません。`pip install onnxruntime` でインストールしてください。"
        ) from exc

    providers: list[str] = ["CPUExecutionProvider"]
    if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)

    used = session.get_providers()
    if "CUDAExecutionProvider" in used:
        logger.info("ONNX Runtime: CUDAExecutionProvider を使用します")
        return session, "cuda"
    if prefer_cuda:
        logger.warning(
            "CUDAExecutionProvider が利用できないため ONNX Runtime は CPU で実行します"
            "（onnxruntime-gpu のインストールを検討してください）"
        )
    else:
        logger.info("ONNX Runtime: CPUExecutionProvider を使用します")
    return session, "cpu"

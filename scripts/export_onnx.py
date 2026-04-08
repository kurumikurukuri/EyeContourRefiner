"""
学習済み PyTorch モデルを ONNX 形式にエクスポートする。

使い方:
    python scripts/export_onnx.py \\
        --model  models/eyeliner_detection.pt \\
        --output models/eyeliner_detection.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def export(model_path: Path, output_path: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch が見つかりません。`pip install torch` でインストールしてください。"
        ) from exc

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx が見つかりません。`pip install onnx` でインストールしてください。"
        ) from exc

    from eyeliner.models.eyeliner_cnn import EyelinerCNN
    from eyeliner.models.ensemble import build_model, EnsembleModel
    from utils.patch_utils import PATCH_SIZE

    logger.info("モデルを読み込み中: %s", model_path)
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and checkpoint.get("type") == "ensemble":
        n = checkpoint["n_models"]
        backbone = checkpoint.get("backbone", "custom")
        models = [build_model(backbone, pretrained=False) for _ in range(n)]
        model = EnsembleModel(models)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("アンサンブルモデルを読み込みました (%d folds, backbone=%s)", n, backbone)
    elif isinstance(checkpoint, dict) and "backbone" in checkpoint:
        model = build_model(checkpoint["backbone"], pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = EyelinerCNN()
        model.load_state_dict(checkpoint)

    model.eval()

    patch_h, patch_w = PATCH_SIZE
    dummy_patch    = torch.zeros(1, 3, patch_h, patch_w)
    dummy_is_upper = torch.zeros(1, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_patch, dummy_is_upper),
        str(output_path),
        input_names=["patch", "is_upper"],
        output_names=["t"],
        dynamic_axes={
            "patch":    {0: "batch"},
            "is_upper": {0: "batch"},
            "t":        {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # 検証
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    logger.info("ONNX モデルをエクスポートしました: %s", output_path)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("パラメータ数: %d", total_params)


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="EyelinerCNN を ONNX にエクスポート")
    parser.add_argument("--model",  required=True, type=Path, help="eyeliner_detection.pt のパス")
    parser.add_argument("--output", default="models/eyeliner_detection.onnx", type=Path)
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()

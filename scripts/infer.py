"""
アイライン輪郭検出 CNN の推論スクリプト。

使い方:
    # manifest.csv の全サンプルに対して推論
    python scripts/infer.py \\
        --model   models/eyeliner_detection.pt \\
        --manifest data/manifest.csv \\
        --output  data/predictions.csv

    # 単一画像・座標を直接指定
    python scripts/infer.py \\
        --model models/eyeliner_detection.pt \\
        --image data/images/face_001.jpg \\
        --px 320 --py 210 --qx 180 --qy 215

出力 CSV の列:
    idx, image_path, part_name, sub_key, point_index,
    px, py, qx, qy, t_pred, rx_pred, ry_pred
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch が見つかりません。`pip install torch` でインストールしてください。"
    ) from exc

from eyeliner.models.eyeliner_cnn import EyelinerCNN
from eyeliner.models.ensemble import EnsembleModel, create_ensemble_from_dir
from eyeliner.models.ensemble import load_model as _load_model_from_ckpt
from utils.patch_utils import (
    T_MIN, T_MAX, PATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    extract_patch, reconstruct_point,
)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """単一モデルまたはアンサンブルモデルを読み込む。

    ensemble.py の load_model に委譲し、backbone / use_batchnorm / use_sigmoid /
    use_pq_ratio をチェックポイントから自動復元する。
    """
    model = _load_model_from_ckpt(model_path)
    model.eval()
    model.to(device)
    logger.info("モデルを読み込みました: %s", model_path)
    return model


def load_models(models_dir: Path, device: torch.device) -> EnsembleModel:
    """models_dir 内の eyeliner_detection_fold*.pt を全て読み込みアンサンブルモデルを返す。"""
    ensemble, _ = create_ensemble_from_dir(models_dir)
    ensemble.eval()
    ensemble.to(device)
    logger.info("アンサンブル: %d 個の fold モデルを読み込みました", len(ensemble._models))
    return ensemble


def predict(
    model: torch.nn.Module,
    device: torch.device,
    image: np.ndarray,
    px: float,
    py: float,
    qx: float,
    qy: float,
    is_upper: float,
    patch_t_min: float = T_MIN,
    patch_t_max: float = T_MAX,
    patch_perp_scale: float = 1.0,
    patch_size: tuple = PATCH_SIZE,
) -> float:
    """パッチを抽出して ImageNet 正規化を適用しモデルに通し、t_norm ∈ [0,1] を返す。"""
    patch = extract_patch(
        image, px, py, qx, qy,
        patch_size=patch_size,
        t_min=patch_t_min, t_max=patch_t_max, perp_scale=patch_perp_scale,
    )
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)
    with torch.no_grad():
        t_norm = torch.clamp(model(x, is_upper_t), 0.0, 1.0).item()
    return float(t_norm)


# ---------------------------------------------------------------------------
# Manifest mode
# ---------------------------------------------------------------------------

def run_manifest(
    model: EyelinerCNN,
    device: torch.device,
    manifest_path: Path,
    output_path: Path,
    patch_t_min: float = T_MIN,
    patch_t_max: float = T_MAX,
    label_t_min: float | None = None,
    label_t_max: float | None = None,
    patch_perp_scale: float = 1.0,
    patch_size: tuple = PATCH_SIZE,
) -> None:
    records = _load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest に有効なサンプルが見つかりません。")

    out_fieldnames = [
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy", "t_true", "t_pred", "rx_pred", "ry_pred",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for rec in records:
            image_path = rec["image_path"]
            px = float(rec["px"])
            py = float(rec["py"])
            qx = float(rec["qx"])
            qy = float(rec["qy"])

            bgr = cv2.imread(image_path)
            if bgr is None:
                logger.warning("画像を読み込めません: %s", image_path)
                errors += 1
                continue

            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            is_upper = 1.0 if rec.get("sub_key", "") == "upper" else 0.0
            t_pred = predict(
                model, device, image, px, py, qx, qy, is_upper,
                patch_t_min=patch_t_min, patch_t_max=patch_t_max,
                patch_perp_scale=patch_perp_scale,
                patch_size=patch_size,
            )
            # label 範囲（None のときは crop 範囲にフォールバック）
            eff_label_min = label_t_min if label_t_min is not None else patch_t_min
            eff_label_max = label_t_max if label_t_max is not None else patch_t_max
            # t_pred (t_norm [0,1]) → 物理 t → 座標復元
            t_actual_pred = t_pred * (eff_label_max - eff_label_min) + eff_label_min
            rx_pred, ry_pred = reconstruct_point(px, py, qx, qy, t_actual_pred)

            # t_true: manifest の t は生の物理 t 値をそのまま使用
            t_stored = rec.get("t", "")
            try:
                t_actual_true = float(t_stored)
                t_true_str = f"{t_actual_true:.6f}"
            except (ValueError, TypeError):
                t_true_str = t_stored

            writer.writerow({
                "idx":         rec.get("idx", ""),
                "image_path":  image_path,
                "part_name":   rec.get("part_name", ""),
                "sub_key":     rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "px": px, "py": py, "qx": qx, "qy": qy,
                "t_true":   t_true_str,
                "t_pred":   f"{t_actual_pred:.6f}",
                "rx_pred":  f"{rx_pred:.2f}",
                "ry_pred":  f"{ry_pred:.2f}",
            })

    n = len(records) - errors
    logger.info("推論完了: %d 件成功, %d 件スキップ -> %s", n, errors, output_path)
    if errors > 0:
        logger.warning("%d 件の画像読み込みに失敗しました。MAE は成功分のみで計算します。", errors)
    _print_mae(output_path)


def _load_manifest(manifest_path: Path) -> List[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest が見つかりません: {manifest_path}")
    records = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("t") in (None, "", "None"):
                continue
            try:
                float(row["t"])
            except (ValueError, KeyError):
                continue
            records.append(row)
    logger.info("manifest から %d サンプルを読み込みました。", len(records))
    return records


def _print_mae(predictions_path: Path) -> None:
    """t_true と t_pred の MAE をログ出力する。"""
    errors: List[float] = []
    with open(predictions_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                errors.append(abs(float(row["t_true"]) - float(row["t_pred"])))
            except (ValueError, KeyError):
                continue
    if errors:
        mae = sum(errors) / len(errors)
        logger.info("MAE (raw t): %.4f  (n=%d)", mae, len(errors))


# ---------------------------------------------------------------------------
# Single-image mode
# ---------------------------------------------------------------------------

def run_single(
    model: EyelinerCNN,
    device: torch.device,
    image_path: Path,
    px: float,
    py: float,
    qx: float,
    qy: float,
    is_upper: float,
    patch_t_min: float = T_MIN,
    patch_t_max: float = T_MAX,
    label_t_min: float | None = None,
    label_t_max: float | None = None,
    patch_perp_scale: float = 1.0,
    patch_size: tuple = PATCH_SIZE,
) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t_pred = predict(
        model, device, image, px, py, qx, qy, is_upper,
        patch_t_min=patch_t_min, patch_t_max=patch_t_max,
        patch_perp_scale=patch_perp_scale,
        patch_size=patch_size,
    )
    eff_label_min = label_t_min if label_t_min is not None else patch_t_min
    eff_label_max = label_t_max if label_t_max is not None else patch_t_max
    t_actual_pred = t_pred * (eff_label_max - eff_label_min) + eff_label_min
    rx_pred, ry_pred = reconstruct_point(px, py, qx, qy, t_actual_pred)

    logger.info("t_pred  : %.6f", t_actual_pred)
    logger.info("R_pred  : (%.2f, %.2f)", rx_pred, ry_pred)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="アイライン CNN 推論")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model",      type=Path, help="単一モデル (.pt)")
    model_group.add_argument("--models-dir", type=Path, dest="models_dir",
                             help="K-Fold モデルディレクトリ (eyeliner_detection_fold*.pt を全て使用)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="manifest.csv のパス")
    group.add_argument("--image",    type=Path, help="単一画像のパス")

    parser.add_argument("--output", type=Path, default=Path("data/predictions.csv"),
                        help="manifest モード時の出力 CSV (default: data/predictions.csv)")
    parser.add_argument("--px", type=float)
    parser.add_argument("--py", type=float)
    parser.add_argument("--qx", type=float)
    parser.add_argument("--qy", type=float)
    parser.add_argument(
        "--is_upper", action="store_true",
        help="upper アイラインの場合に指定（単一画像モード用）",
    )
    parser.add_argument(
        "--patch_t_min", type=float, default=T_MIN,
        help=f"パッチ抽出（crop）の t 下限（default: {T_MIN}）",
    )
    parser.add_argument(
        "--patch_t_max", type=float, default=T_MAX,
        help=f"パッチ抽出（crop）の t 上限（default: {T_MAX}）",
    )
    parser.add_argument(
        "--label_t_min", type=float, default=None,
        help="ラベル正規化の t 下限（default: None → patch_t_min と同じ）",
    )
    parser.add_argument(
        "--label_t_max", type=float, default=None,
        help="ラベル正規化の t 上限（default: None → patch_t_max と同じ）",
    )
    parser.add_argument(
        "--patch_perp_scale", type=float, default=1.0,
        help="縦軸カバレッジ = perp_scale * |PQ| ピクセル（default: 1.0）",
    )
    parser.add_argument(
        "--patch_h", type=int, default=64,
        help="入力パッチ高さ（default: 64）",
    )
    parser.add_argument(
        "--patch_w", type=int, default=128,
        help="入力パッチ幅（default: 128）",
    )
    args = parser.parse_args()

    if args.image and any(v is None for v in [args.px, args.py, args.qx, args.qy]):
        parser.error("--image 使用時は --px --py --qx --qy が必要です。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    model: torch.nn.Module = (
        load_models(args.models_dir, device)
        if args.models_dir
        else load_model(args.model, device)
    )

    patch_size = (args.patch_h, args.patch_w)
    if args.manifest:
        run_manifest(
            model, device, args.manifest, args.output,
            patch_t_min=args.patch_t_min,
            patch_t_max=args.patch_t_max,
            label_t_min=args.label_t_min,
            label_t_max=args.label_t_max,
            patch_perp_scale=args.patch_perp_scale,
            patch_size=patch_size,
        )
    else:
        run_single(
            model, device, args.image, args.px, args.py, args.qx, args.qy,
            is_upper=1.0 if args.is_upper else 0.0,
            patch_t_min=args.patch_t_min,
            patch_t_max=args.patch_t_max,
            label_t_min=args.label_t_min,
            label_t_max=args.label_t_max,
            patch_perp_scale=args.patch_perp_scale,
            patch_size=patch_size,
        )


if __name__ == "__main__":
    main()

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
    px, py, qx, qy,
    true_t_raw, true_t_raw_quant, true_t_norm,
    pred_t_raw, pred_t_raw_quant, pred_t_norm,
    rx_pred, ry_pred
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
from eyeliner.data.patch_utils import (
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
    use_t_norm: bool = True,
) -> float:
    """パッチを抽出して ImageNet 正規化を適用しモデルに通し、予測 t を返す。

    use_t_norm=True  : 出力を [0,1] にクランプして返す（呼び出し元でデノーマライズ）
    use_t_norm=False : モデル出力（生の物理 t 値）をそのまま返す
    """
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
        raw = model(x, is_upper_t)
        if use_t_norm:
            result = torch.clamp(raw, 0.0, 1.0).item()
        else:
            result = raw.item()
    return float(result)


# ---------------------------------------------------------------------------
# Manifest mode
# ---------------------------------------------------------------------------

def _quantize(value: float, step: float | None) -> float:
    """raw t 値を step 分解能に量子化する。step が None/0 以下ならそのまま返す。"""
    if step is None or step <= 0:
        return value
    return round(value / step) * step


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
    use_t_norm: bool = True,
    t_quant_step: float | None = None,
) -> None:
    records = _load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest に有効なサンプルが見つかりません。")

    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_fieldnames = [
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy",
        "true_t_raw", "true_t_raw_quant", "true_t_norm",
        "pred_t_raw", "pred_t_raw_quant", "pred_t_norm",
        "rx_pred", "ry_pred",
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
            del bgr
            is_upper = 1.0 if rec.get("sub_key", "") == "upper" else 0.0
            t_pred = predict(
                model, device, image, px, py, qx, qy, is_upper,
                patch_t_min=patch_t_min, patch_t_max=patch_t_max,
                patch_perp_scale=patch_perp_scale,
                patch_size=patch_size,
                use_t_norm=use_t_norm,
            )
            # label 範囲（None のときは crop 範囲にフォールバック）
            eff_label_min = label_t_min if label_t_min is not None else patch_t_min
            eff_label_max = label_t_max if label_t_max is not None else patch_t_max
            label_range = eff_label_max - eff_label_min

            if use_t_norm:
                # t_pred はモデル出力の [0,1] 正規化済み値
                pred_t_norm = float(np.clip(t_pred, 0.0, 1.0))
                pred_t_raw = pred_t_norm * label_range + eff_label_min
            else:
                # no_t_norm: モデル出力が既に生の物理 t 値
                pred_t_raw = t_pred
                pred_t_norm = (pred_t_raw - eff_label_min) / label_range if label_range != 0 else 0.0

            pred_t_raw_quant = _quantize(pred_t_raw, t_quant_step)
            rx_pred, ry_pred = reconstruct_point(px, py, qx, qy, pred_t_raw)

            # true_t: manifest の t は生の物理 t 値
            t_stored = rec.get("t", "")
            try:
                true_t_raw = float(t_stored)
                true_t_raw_quant = _quantize(true_t_raw, t_quant_step)
                true_t_norm = (true_t_raw - eff_label_min) / label_range if label_range != 0 else 0.0
                true_t_raw_str       = f"{true_t_raw:.6f}"
                true_t_raw_quant_str = f"{true_t_raw_quant:.6f}"
                true_t_norm_str      = f"{true_t_norm:.6f}"
            except (ValueError, TypeError):
                true_t_raw_str       = t_stored
                true_t_raw_quant_str = ""
                true_t_norm_str      = ""

            writer.writerow({
                "idx":              rec.get("idx", ""),
                "image_path":       image_path,
                "part_name":        rec.get("part_name", ""),
                "sub_key":          rec.get("sub_key", ""),
                "point_index":      rec.get("point_index", ""),
                "px": px, "py": py, "qx": qx, "qy": qy,
                "true_t_raw":       true_t_raw_str,
                "true_t_raw_quant": true_t_raw_quant_str,
                "true_t_norm":      true_t_norm_str,
                "pred_t_raw":       f"{pred_t_raw:.6f}",
                "pred_t_raw_quant": f"{pred_t_raw_quant:.6f}",
                "pred_t_norm":      f"{pred_t_norm:.6f}",
                "rx_pred":          f"{rx_pred:.2f}",
                "ry_pred":          f"{ry_pred:.2f}",
            })
            del image

    n = len(records) - errors
    logger.info("推論完了: %d 件成功, %d 件スキップ -> %s", n, errors, output_path)
    if errors > 0:
        logger.warning("%d 件の画像読み込みに失敗しました。MAE は成功分のみで計算します。", errors)
    mae_raw, mae_quant = _compute_mae(output_path, t_quant_step=t_quant_step)
    if mae_raw is not None:
        _update_experiment_json(output_path, mae_raw, mae_quant, t_quant_step)


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


def _compute_mae(
    predictions_path: Path,
    t_quant_step: float | None = None,
) -> tuple[float | None, float | None]:
    """MAE を二重計算する。

    Returns:
        (mae_raw, mae_quant)
        mae_raw   : |true_t_raw - pred_t_raw|                 （常に計算）
        mae_quant : |true_t_raw_quant - pred_t_raw_quant|     （t_quant_step 指定時のみ）
    """
    diffs_raw: List[float] = []
    diffs_quant: List[float] = []
    with open(predictions_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                diffs_raw.append(abs(float(row["true_t_raw"]) - float(row["pred_t_raw"])))
            except (ValueError, KeyError):
                continue
            if t_quant_step is not None and t_quant_step > 0:
                try:
                    diffs_quant.append(
                        abs(float(row["true_t_raw_quant"]) - float(row["pred_t_raw_quant"]))
                    )
                except (ValueError, KeyError):
                    continue
    if not diffs_raw:
        return None, None
    mae_raw = sum(diffs_raw) / len(diffs_raw)
    logger.info("MAE (raw t): %.4f  (n=%d)", mae_raw, len(diffs_raw))
    mae_quant: float | None = None
    if diffs_quant:
        mae_quant = sum(diffs_quant) / len(diffs_quant)
        logger.info(
            "MAE (quant t, step=%g): %.4f  (n=%d)",
            t_quant_step, mae_quant, len(diffs_quant),
        )
    return mae_raw, mae_quant


def _update_experiment_json(
    predictions_path: Path,
    mae_raw: float,
    mae_quant: float | None,
    t_quant_step: float | None,
) -> None:
    """experiment.json の results に test_mae / train_mae を書き込む。

    互換性のため test_mae / train_mae は raw 値を指す。量子化側は
    test_mae_quant / train_mae_quant として別キーに記録する。
    """
    import json
    exp_json = predictions_path.parent / "experiment.json"
    if not exp_json.exists():
        return
    name = predictions_path.name
    if "test" in name:
        prefix = "test"
    elif "train" in name:
        prefix = "train"
    else:
        return
    try:
        data = json.loads(exp_json.read_text(encoding="utf-8"))
        results = data.setdefault("results", {})
        results[f"{prefix}_mae"] = mae_raw
        if mae_quant is not None:
            results[f"{prefix}_mae_quant"] = mae_quant
            results[f"{prefix}_mae_quant_step"] = t_quant_step
        exp_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log_msg = f"experiment.json に {prefix}_mae={mae_raw:.6f}"
        if mae_quant is not None:
            log_msg += f" / {prefix}_mae_quant={mae_quant:.6f} (step={t_quant_step})"
        logger.info("%s を記録しました: %s", log_msg, exp_json)
    except Exception as exc:
        logger.warning("experiment.json の更新に失敗しました: %s", exc)


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
    use_t_norm: bool = True,
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
        use_t_norm=use_t_norm,
    )
    if use_t_norm:
        eff_label_min = label_t_min if label_t_min is not None else patch_t_min
        eff_label_max = label_t_max if label_t_max is not None else patch_t_max
        t_actual_pred = t_pred * (eff_label_max - eff_label_min) + eff_label_min
    else:
        t_actual_pred = t_pred
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
    parser.add_argument(
        "--no_t_norm", action="store_true",
        help="t正規化なしで学習したモデルの推論時に指定。モデル出力を生の物理 t 値として扱う。",
    )
    parser.add_argument(
        "--t_quant_step", type=float, default=None,
        help="生の物理 t 値の量子化ステップ（例: 0.1 / 0.02 / 0.01）。"
             "指定時は quantized 列と MAE_quant も出力する。",
    )
    parser.add_argument(
        "--round_t", type=int, default=None,
        help="(後方互換) --t_quant_step = 10**(-round_t) に翻訳される。新規は --t_quant_step を使うこと",
    )
    args = parser.parse_args()

    # --round_t エイリアス
    if args.round_t is not None:
        if args.t_quant_step is not None:
            parser.error("--round_t と --t_quant_step は同時指定できません。--t_quant_step を使ってください。")
        args.t_quant_step = 10.0 ** (-args.round_t)
        logger.warning(
            "--round_t=%d は非推奨です。--t_quant_step=%g に翻訳しました。",
            args.round_t, args.t_quant_step,
        )

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
    use_t_norm = not args.no_t_norm
    if args.manifest:
        run_manifest(
            model, device, args.manifest, args.output,
            patch_t_min=args.patch_t_min,
            patch_t_max=args.patch_t_max,
            label_t_min=args.label_t_min,
            label_t_max=args.label_t_max,
            patch_perp_scale=args.patch_perp_scale,
            patch_size=patch_size,
            use_t_norm=use_t_norm,
            t_quant_step=args.t_quant_step,
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
            use_t_norm=use_t_norm,
        )


if __name__ == "__main__":
    main()

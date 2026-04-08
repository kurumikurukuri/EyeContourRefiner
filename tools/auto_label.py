"""
自動ラベリングスクリプト。

学習済みK-Foldモデルのアンサンブルを使って未ラベルサンプルに自動ラベルを付与する。

信頼度 = 1 - 2 * std(fold_predictions)
  - 各フォールドの予測値 [0,1] の標準偏差が 0 → confidence=1.0（完全一致）
  - 標準偏差が 0.5（最大）→ confidence=0.0

高信頼度サンプル（>= threshold）は t と label_source を samples.json に書き込む。
低信頼度サンプルは t=null のまま維持（annotator でのレビューキュー）。

使用方法:
    python scripts/auto_label.py
    python scripts/auto_label.py --threshold 0.90 --model-version 2
    python scripts/auto_label.py --dry-run --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    T_MIN,
    T_MAX,
    extract_patch,
)

logger = logging.getLogger(__name__)

SAMPLES_PATH = Path("data/samples.json")
MANIFEST_PATH = Path("data/manifest.csv")
MANIFEST_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "label_source", "confidence",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_kfold_model_paths() -> list[Path]:
    """K-Foldモデルファイルを優先順位順に探索する。"""
    kfold_paths = sorted(Path("models_kfold").glob("eyeliner_detection_fold*.pt"))
    if kfold_paths:
        return kfold_paths
    fold_paths = sorted(Path("models").glob("eyeliner_detection_fold*.pt"))
    if fold_paths:
        return fold_paths
    single = Path("models/eyeliner_detection.pt")
    if single.exists():
        logger.warning("K-Foldモデルが見つかりません。単体モデルを使用します（信頼度は常に1.0になります）")
        return [single]
    return []


def _load_models(model_paths: list[Path]):
    """モデルをロードしてリストで返す。torch と EyelinerCNN に依存。"""
    import torch
    from eyeliner.models.eyeliner_cnn import EyelinerCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in model_paths:
        try:
            model = EyelinerCNN()
            model.load_state_dict(
                torch.load(str(path), map_location=device, weights_only=True)
            )
            model.eval()
            model.to(device)
            models.append(model)
            logger.info("モデルを読み込みました: %s", path)
        except Exception as exc:
            logger.warning("モデルの読み込みをスキップしました %s: %s", path, exc)

    return models, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_with_confidence(
    models: list,
    device,
    mean_t,
    std_t,
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    is_upper: float,
) -> tuple[float, float]:
    """全フォールドモデルの予測を平均・標準偏差から (t_mean, confidence) を返す。"""
    import torch

    patch = extract_patch(image_rgb, px, py, qx, qy)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    x = (x - mean_t) / std_t
    is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = [model(x, is_upper_t).item() for model in models]

    t_mean = float(np.mean(preds))
    if len(preds) > 1:
        t_std = float(np.std(preds))
        # std の最大理論値は 0.5 (半分が0、半分が1の場合)
        confidence = max(0.0, 1.0 - 2.0 * t_std)
    else:
        # 単体モデルは信頼度を計算できないため 1.0 とみなす
        confidence = 1.0

    return t_mean, confidence


# ---------------------------------------------------------------------------
# Main auto-labeling logic
# ---------------------------------------------------------------------------

def auto_label(
    threshold: float,
    model_version: int,
    dry_run: bool,
    verbose: bool,
) -> dict:
    import torch

    # ---- モデル読み込み ----
    model_paths = _find_kfold_model_paths()
    if not model_paths:
        logger.error("モデルが見つかりません。先にモデルを学習してください。")
        sys.exit(1)

    models, device = _load_models(model_paths)
    if not models:
        logger.error("有効なモデルを読み込めませんでした。")
        sys.exit(1)

    logger.info("%d 個のモデルを使用 (device=%s)", len(models), device)

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)

    label_source_tag = f"auto_v{model_version}"

    # ---- サンプル読み込み ----
    samples: list[dict] = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    unannotated = [s for s in samples if s.get("t") is None]
    logger.info("未ラベルサンプル: %d / %d", len(unannotated), len(samples))

    # ---- 推論 ----
    stats = {"accepted": 0, "rejected": 0, "error": 0}
    image_cache: dict[str, np.ndarray] = {}
    updated_indices: dict[int, tuple[float, float]] = {}  # idx → (t, confidence)

    for s in unannotated:
        img_path = s["image_path"]
        if img_path not in image_cache:
            bgr = cv2.imread(img_path)
            if bgr is None:
                logger.warning("画像を読み込めません: %s", img_path)
                stats["error"] += 1
                continue
            image_cache[img_path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_rgb = image_cache[img_path]
        is_upper = 1.0 if s.get("sub_key") == "upper" else 0.0

        try:
            t_mean, confidence = _predict_with_confidence(
                models, device, mean_t, std_t,
                image_rgb,
                float(s["px"]), float(s["py"]),
                float(s["qx"]), float(s["qy"]),
                is_upper,
            )
        except Exception as exc:
            logger.warning("推論エラー (idx=%s): %s", s.get("idx"), exc)
            stats["error"] += 1
            continue

        if verbose:
            logger.debug(
                "idx=%d  t=%.4f  confidence=%.4f  %s",
                s.get("idx", -1), t_mean, confidence,
                "✓" if confidence >= threshold else "✗",
            )

        if confidence >= threshold:
            updated_indices[s["idx"]] = (t_mean, confidence)
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1

    logger.info(
        "結果: accepted=%d  rejected=%d  error=%d  (threshold=%.2f)",
        stats["accepted"], stats["rejected"], stats["error"], threshold,
    )

    if dry_run:
        logger.info("[DRY RUN] samples.json / manifest.csv への書き込みをスキップしました。")
        return stats

    # ---- samples.json 更新 ----
    updated_samples = []
    for s in samples:
        if s["idx"] in updated_indices:
            t_val, conf = updated_indices[s["idx"]]
            # t_val はモデル出力の t_norm [0,1]。samples.json には生の物理 t 値で保存する。
            t_raw = t_val * (T_MAX - T_MIN) + T_MIN
            updated_samples.append({
                **s,
                "t": round(t_raw, 6),
                "label_source": label_source_tag,
                "confidence": round(conf, 6),
            })
        else:
            updated_samples.append(s)

    SAMPLES_PATH.write_text(
        json.dumps(updated_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("samples.json を更新しました。")

    # ---- manifest.csv 更新（自動ラベルの追記） ----
    _update_manifest(updated_samples, label_source_tag)

    return stats


def _update_manifest(samples: list[dict], label_source_tag: str) -> None:
    """samples.json の内容から manifest.csv を再生成する。"""
    annotated = [s for s in samples if s.get("t") is not None]

    # 既存 manifest があれば読み込んで label_source=manual の行は保持
    existing_manual: dict[int, dict] = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "label_source" in reader.fieldnames:
                for row in reader:
                    try:
                        idx = int(row["idx"])
                        existing_manual[idx] = row
                    except (ValueError, KeyError):
                        pass

    rows: list[dict] = []
    for s in annotated:
        idx = s["idx"]
        if idx in existing_manual:
            # 既存行のラベルソースを上書きせず保持（manual/auto_reviewed を守る）
            existing_row = existing_manual[idx]
            rows.append({
                "idx": idx,
                "image_path": s["image_path"],
                "part_name": s["part_name"],
                "sub_key": s["sub_key"],
                "point_index": s["point_index"],
                "px": s["px"],
                "py": s["py"],
                "qx": s["qx"],
                "qy": s["qy"],
                "t": s["t"],
                "label_source": existing_row.get("label_source", s.get("label_source", "manual")),
                "confidence": existing_row.get("confidence", s.get("confidence", "")),
            })
        else:
            rows.append({
                "idx": idx,
                "image_path": s["image_path"],
                "part_name": s["part_name"],
                "sub_key": s["sub_key"],
                "point_index": s["point_index"],
                "px": s["px"],
                "py": s["py"],
                "qx": s["qx"],
                "qy": s["qy"],
                "t": s["t"],
                "label_source": s.get("label_source", "manual"),
                "confidence": s.get("confidence") or "",
            })

    rows.sort(key=lambda r: int(r["idx"]))

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("manifest.csv を更新しました: %d 件", len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="学習済みモデルで未ラベルサンプルに自動ラベルを付与")
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="信頼度閾値 (default: 0.85)。これ以上の場合のみ自動ラベルを受理",
    )
    parser.add_argument(
        "--model-version", type=int, default=1,
        help="ラベルソースのバージョン番号 (default: 1) → label_source=auto_v1",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="推論のみ実行し、ファイルへの書き込みは行わない",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="各サンプルの予測値と信頼度を表示",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.dry_run:
        logger.info("[DRY RUN] ファイルへの書き込みは行いません")

    stats = auto_label(
        threshold=args.threshold,
        model_version=args.model_version,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f"\n自動ラベリング完了:")
    print(f"  受理 (confidence >= {args.threshold}): {stats['accepted']} 件")
    print(f"  要レビュー (confidence < {args.threshold}):  {stats['rejected']} 件")
    print(f"  エラー:                                {stats['error']} 件")
    if not args.dry_run:
        print(f"  label_source: {f'auto_v{args.model_version}'}")


if __name__ == "__main__":
    main()

"""
未ラベルデータ（t IS NULL）に対する t_raw 予測スクリプト。

DB から未ラベル行を読み取り、指定された .pt モデル（単一 or K-Fold アンサンブル）
で推論し、予測結果を CSV に出力する。

使い方:
    python analyses/pred_unlabel/predict_unlabel.py \
        --models-dir experiments/EXP015_interior_corner_model/outputs/S1_baseline/S1_baseline \
        --output analyses/pred_unlabel/unlabel_predictions.csv

    python analyses/pred_unlabel/predict_unlabel.py \
        --model path/to/model.pt \
        --output analyses/pred_unlabel/unlabel_predictions.csv

    # skip_flag=1 も含めて全未ラベルを予測
    python analyses/pred_unlabel/predict_unlabel.py \
        --models-dir ... \
        --include-skipped

出力 CSV 列:
    idx, image_path, part_name, sub_key, point_index,
    px, py, qx, qy,
    pred_t_raw, rx_pred, ry_pred
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from eyeliner.data.patch_utils import (
    T_MIN,
    T_MAX,
    PATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_patch,
    reconstruct_point,
)
from eyeliner.models.ensemble import EnsembleModel, create_ensemble_from_dir
from eyeliner.models.ensemble import load_model as load_ckpt

DB_PATH = Path("data/samples.db")


def load_single_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = load_ckpt(model_path)
    model.eval()
    model.to(device)
    return model


def load_ensemble(models_dir: Path, device: torch.device) -> EnsembleModel:
    ensemble, _ = create_ensemble_from_dir(models_dir)
    ensemble.eval()
    ensemble.to(device)
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
) -> float:
    """パッチを抽出して推論し、予測 t_raw を返す。"""
    patch = extract_patch(image, px, py, qx, qy)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)
    with torch.no_grad():
        raw = model(x, is_upper_t)
        t_norm = torch.clamp(raw, 0.0, 1.0).item()
    return t_norm * (T_MAX - T_MIN) + T_MIN


def fetch_unlabeled(include_skipped: bool = False) -> List[dict]:
    extra = "" if include_skipped else "AND skip_flag = 0"
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT idx, image_path, part_name, sub_key, point_index, px, py, qx, qy "
        f"FROM samples WHERE t IS NULL {extra} "
        f"ORDER BY idx"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run(
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
    include_skipped: bool = False,
) -> None:
    records = fetch_unlabeled(include_skipped)
    logger.info("未ラベルサンプル数: %d", len(records))
    if not records:
        logger.warning("未ラベルサンプルが見つかりません。")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy",
        "pred_t_raw", "rx_pred", "ry_pred",
    ]
    errors = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(records):
            image_path = rec["image_path"]
            bgr = cv2.imread(image_path)
            if bgr is None:
                logger.warning("画像を読み込めません: %s", image_path)
                errors += 1
                continue

            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            px = float(rec["px"])
            py = float(rec["py"])
            qx = float(rec["qx"])
            qy = float(rec["qy"])
            is_upper = 1.0 if rec.get("sub_key", "") == "upper" else 0.0

            t_pred = predict(model, device, image, px, py, qx, qy, is_upper)
            rx_pred, ry_pred = reconstruct_point(px, py, qx, qy, t_pred)

            writer.writerow({
                "idx":         rec["idx"],
                "image_path":  image_path,
                "part_name":   rec["part_name"],
                "sub_key":     rec["sub_key"],
                "point_index": rec["point_index"],
                "px": px, "py": py, "qx": qx, "qy": qy,
                "pred_t_raw":  f"{t_pred:.6f}",
                "rx_pred":     f"{rx_pred:.2f}",
                "ry_pred":     f"{ry_pred:.2f}",
            })

            if (i + 1) % 1000 == 0:
                logger.info("進捗: %d/%d", i + 1, len(records))

    n = len(records) - errors
    logger.info("推論完了: %d 件成功, %d 件スキップ → %s", n, errors, output_path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="未ラベルデータに対する t_raw 予測")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, help="単一モデル (.pt)")
    model_group.add_argument("--models-dir", type=Path, dest="models_dir",
                             help="K-Fold モデルディレクトリ")

    parser.add_argument("--output", type=Path,
                        default=Path("analyses/pred_unlabel/unlabel_predictions.csv"),
                        help="出力 CSV パス")
    parser.add_argument("--include-skipped", action="store_true",
                        help="skip_flag=1 の行も予測対象に含める")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    model: torch.nn.Module = (
        load_ensemble(args.models_dir, device)
        if args.models_dir
        else load_single_model(args.model, device)
    )
    logger.info("モデルを読み込みました")

    run(model, device, args.output, include_skipped=args.include_skipped)


if __name__ == "__main__":
    main()

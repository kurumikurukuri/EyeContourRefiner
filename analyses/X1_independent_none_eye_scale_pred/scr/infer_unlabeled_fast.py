"""X1_independent_none_eye_scale (EXP020_dual_branch) で unlabeled サンプルを高速推論。

枝A: 生PQ座標系 (raw_t)
枝B: P'Q' = eye_scale × 0.2 (t_prime → 生t に逆変換)
アンサンブル: fold モデルの生t平均

infer_unlabeled_fast.py の設計方針:
- image_path でグループ化し同一画像を1回だけ読む
- compute_eye_meta で eye_scale を一括計算
- --resume で既処理 idx をスキップ

Usage:
    python analyses/X1_independent_none_eye_scale_pred/scr/infer_unlabeled_fast.py
    python analyses/X1_independent_none_eye_scale_pred/scr/infer_unlabeled_fast.py --no-resume
"""
from __future__ import annotations

import csv
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

import torch

from eyeliner.data.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_patch,
    reconstruct_point,
    compute_pq_rescale,
)
from eyeliner.data.weights import compute_eye_meta
from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t
from eyeliner.models.dual_ensemble import load_dual_fold_models

logger = logging.getLogger(__name__)

DB_PATH = ROOT / "data" / "samples.db"
MANIFEST_OUT = ANALYSIS_DIR / "manifests" / "manifest_unlabeled.csv"
PREDICTIONS_OUT = ANALYSIS_DIR / "predictions_unlabeled.csv"
MODEL_DIR = ROOT / "experiments" / "EXP020_dual_branch" / "outputs" / "X1_independent_none_eye_scale"
FUSION_MODE = "independent"
RESCALE_B = "eye_scale"

OUT_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy",
    "true_t_raw", "pred_t_raw", "pred_t_norm",
    "rx_pred", "ry_pred",
]

# 出力正規化: 生tをWebUI表示用の[0,1]にマップするための参照範囲（枝Aのlabel空間）
_LABEL_T_MIN = -1.0
_LABEL_T_MAX = 2.0
_LABEL_RANGE = _LABEL_T_MAX - _LABEL_T_MIN


def load_records() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.execute(
        "SELECT idx, image_path, face_index, part_name, sub_key, point_index, "
        "px, py, qx, qy, eye_scale, skip_flag "
        "FROM samples "
        "WHERE t IS NULL AND skip_flag = 0 AND eye_scale > 60"
    )
    cols = [d[0] for d in cursor.description]
    records = [dict(zip(cols, row)) for row in cursor.fetchall()]
    conn.close()
    logger.info("DB から %d 件を取得", len(records))

    for rec in records:
        rec["t"] = 0.5  # compute_eye_meta が必要とするダミー値

    return records


def save_manifest(records: list[dict]) -> None:
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "image_path", "face_index", "part_name", "sub_key",
                  "point_index", "px", "py", "qx", "qy", "t", "skip_flag"]
    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("manifest 保存: %s (%d 行)", MANIFEST_OUT, len(records))


def load_done_idxs(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["idx"]))
            except (KeyError, ValueError):
                pass
    logger.info("resume: 既処理 %d 件をスキップ", len(done))
    return done


def _to_tensor(patch: np.ndarray, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return (x - mean) / std


def process_record(
    rec: dict,
    image: np.ndarray,
    device: torch.device,
    models: list,
    cfg_a,
    cfg_b,
) -> dict:
    orig_px, orig_py = float(rec["px"]), float(rec["py"])
    orig_qx, orig_qy = float(rec["qx"]), float(rec["qy"])
    orig_pq_dist = math.sqrt((orig_qx - orig_px) ** 2 + (orig_qy - orig_py) ** 2)

    # Branch A: 生PQ座標でパッチ抽出
    patch_a = extract_patch(
        image, orig_px, orig_py, orig_qx, orig_qy,
        patch_size=cfg_a.patch_size,
        t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max,
        perp_scale=cfg_a.patch_perp_scale,
    )

    # Branch B: eye_scale × pq_rescale_factor の P'Q' 座標でパッチ抽出
    px_b, py_b, qx_b, qy_b, _, pq_prime_dist = compute_pq_rescale(
        orig_px, orig_py, orig_qx, orig_qy, 0.0,
        rec, cfg_b.pq_rescale_mode, cfg_b.pq_rescale_factor,
    )
    patch_b = extract_patch(
        image, px_b, py_b, qx_b, qy_b,
        patch_size=cfg_b.patch_size,
        t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max,
        perp_scale=cfg_b.patch_perp_scale,
    )

    x_a = _to_tensor(patch_a, device)
    x_b = _to_tensor(patch_b, device)
    is_upper = 1.0 if rec.get("sub_key", "") == "upper" else 0.0
    is_upper_t  = torch.tensor([[is_upper]],        dtype=torch.float32, device=device)
    pq_dist_t   = torch.tensor([[orig_pq_dist]],    dtype=torch.float32, device=device)
    pq_prime_t  = torch.tensor([[pq_prime_dist]],   dtype=torch.float32, device=device)

    with torch.no_grad():
        t_raw_tensor = fold_ensemble_raw_t(
            models, x_a, x_b, is_upper_t,
            FUSION_MODE, cfg_a, cfg_b,
            pq_dist_t, pq_prime_t,
        )
    pred_t_raw = float(t_raw_tensor.squeeze().item())
    pred_t_norm = float(np.clip(
        (pred_t_raw - _LABEL_T_MIN) / _LABEL_RANGE, 0.0, 1.0
    ))

    rx_pred, ry_pred = reconstruct_point(orig_px, orig_py, orig_qx, orig_qy, pred_t_raw)

    return {
        "idx":          rec["idx"],
        "image_path":   rec["image_path"],
        "part_name":    rec.get("part_name", ""),
        "sub_key":      rec.get("sub_key", ""),
        "point_index":  rec.get("point_index", ""),
        "px": rec["px"], "py": rec["py"], "qx": rec["qx"], "qy": rec["qy"],
        "true_t_raw":   "0.500000",   # t IS NULL なのでダミー
        "pred_t_raw":   f"{pred_t_raw:.6f}",
        "pred_t_norm":  f"{pred_t_norm:.6f}",
        "rx_pred":      f"{rx_pred:.2f}",
        "ry_pred":      f"{ry_pred:.2f}",
    }


def main(resume: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not MODEL_DIR.exists():
        logger.error("モデルディレクトリが見つかりません: %s", MODEL_DIR)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    cfg_a, cfg_b = build_branch_configs(RESCALE_B)
    logger.info("枝A: %s", cfg_a)
    logger.info("枝B: %s", cfg_b)

    fold_models = [m.to(device).eval() for m in load_dual_fold_models(MODEL_DIR, FUSION_MODE)]
    logger.info("fold モデル数: %d", len(fold_models))

    records = load_records()
    save_manifest(records)
    compute_eye_meta(records)  # eye_scale を一括計算

    done_idxs: set[int] = load_done_idxs(PREDICTIONS_OUT) if resume else set()
    records = [r for r in records if int(r["idx"]) not in done_idxs]
    logger.info("推論対象: %d 件", len(records))

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["image_path"]].append(rec)
    n_images = len(groups)
    logger.info("ユニーク画像数: %d", n_images)

    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and PREDICTIONS_OUT.exists() else "w"

    try:
        from tqdm import tqdm
        image_iter = tqdm(groups.items(), total=n_images, unit="img")
    except ImportError:
        image_iter = groups.items()

    errors = 0
    done_count = len(done_idxs)
    total = done_count + len(records)
    log_interval = max(n_images // 100, 1)

    with open(PREDICTIONS_OUT, open_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES)
        if open_mode == "w":
            writer.writeheader()

        for i, (image_path, recs) in enumerate(image_iter):
            bgr = cv2.imread(image_path)
            if bgr is None:
                logger.warning("画像読み込み失敗: %s", image_path)
                errors += len(recs)
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            del bgr

            for rec in recs:
                try:
                    row = process_record(rec, image, device, fold_models, cfg_a, cfg_b)
                    writer.writerow(row)
                except Exception as exc:
                    logger.warning("推論失敗 idx=%s: %s", rec.get("idx"), exc)
                    errors += 1

            del image

            if not hasattr(image_iter, "update") and i % log_interval == 0:
                done_count += len(recs)
                pct = done_count / total * 100 if total > 0 else 0
                logger.info("[%d/%d 画像] %.1f%% 完了", i + 1, n_images, pct)

    logger.info("完了: エラー %d 件 -> %s", errors, PREDICTIONS_OUT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="既存出力を上書き（デフォルト: resume）")
    args = parser.parse_args()
    main(resume=not args.no_resume)

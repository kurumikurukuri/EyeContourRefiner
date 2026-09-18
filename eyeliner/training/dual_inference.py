"""デュアルブランチ fold アンサンブルによる manifest 推論。

旧 experiments/EXP020_dual_branch/scr/infer_exp020.py の infer_dual() から移植。
EXP020/EXP021/EXP022/EXP024 が共有する推論実装。

枝A=生PQ（raw_t）、枝B=P'Q'（t_prime）の各予測を生 t 空間へ逆変換してアンサンブルし、
manifest の各レコードについて生 t の正解と予測を CSV に書き出す。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eyeliner.data.dataset import load_manifest
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset
from eyeliner.data.weights import compute_eye_meta
from eyeliner.models.dual_ensemble import load_dual_fold_models
from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t

logger = logging.getLogger(__name__)

PREDICTION_FIELDNAMES = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]


def infer_dual(
    model_dir: Path,
    manifest_path: Path,
    output_path: Path,
    fusion_mode: str,
    rescale_b: str = "eye_scale",
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """model_dir の fold モデル群で manifest を推論し、生 t の予測 CSV を output_path に書き出す。

    Raises:
        FileNotFoundError: manifest または fold モデル（eyeliner_detection_fold*.pt）が存在しない場合。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    logger.info("サンプル数: %d", len(records))

    cfg_a, cfg_b = build_branch_configs(rescale_b)
    ds = EyelinerPatchDualDataset(records, cfg_a, cfg_b, augment_data=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    models = [m.to(device) for m in load_dual_fold_models(model_dir, fusion_mode)]
    logger.info("fold モデル数: %d", len(models))

    all_preds: list[float] = []   # 生 t 予測
    label_range_a = cfg_a.label_t_max - cfg_a.label_t_min
    with torch.no_grad():
        for batch in loader:
            x_a = batch[0].to(device)
            x_b = batch[1].to(device)
            is_upper = batch[2].to(device)
            pq_dist = batch[12].to(device)
            pq_prime = batch[13].to(device)
            t_ens = fold_ensemble_raw_t(
                models, x_a, x_b, is_upper, fusion_mode, cfg_a, cfg_b, pq_dist, pq_prime,
            )
            all_preds.extend(t_ens.squeeze(-1).cpu().tolist())

    # 真値は manifest の生 t を使う。batch[4]（y_a）は学習ロス用に枝A の label 範囲
    # [label_t_min, label_t_max] へクランプ済みで、範囲外の t を持つサンプルの真値が頭打ちされてしまう。
    all_trues = [float(rec["t"]) for rec in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        for rec, t_true, t_pred in zip(records, all_trues, all_preds):
            writer.writerow({
                "idx":         rec.get("idx", ""),
                "image_path":  rec["image_path"],
                "part_name":   rec.get("part_name", ""),
                "sub_key":     rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true":      f"{t_true:.6f}",
                "t_pred":      f"{t_pred:.6f}",
            })

    mae_raw = sum(abs(pt - tt) for pt, tt in zip(all_preds, all_trues)) / max(len(all_preds), 1)
    logger.info("推論完了: MAE=%.6f（生t） / %.6f（正規化）", mae_raw, mae_raw / label_range_a)
    logger.info("保存: %s", output_path)

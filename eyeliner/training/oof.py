"""OOF (Out-Of-Fold) 予測の永続化。

K-Fold CV で各サンプルを「それを学習に使わなかった fold のモデル」で予測した
値を CSV に保存する。GroupKFold（image_path 単位）では各 image はちょうど 1 つの
fold の検証セットに入るため、訓練データ全件が重複なく 1 回ずつ OOF 予測を持てる。

スコアリング（残差ランキング・z-score・品質フィルタ等）は本モジュールの責務外。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

OOF_CSV_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "t_true", "t_pred", "fold_idx",
]


@dataclass(frozen=True)
class OofPrediction:
    """1サンプルの OOF 予測（生 t 空間）。"""

    idx: str
    image_path: str
    part_name: str
    sub_key: str
    point_index: str
    t_true: float
    t_pred: float
    fold_idx: int


def write_oof_csv(predictions: list[OofPrediction], path: str | Path) -> None:
    """OofPrediction のリストを CSV に書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OOF_CSV_FIELDNAMES)
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "idx":         p.idx,
                "image_path":  p.image_path,
                "part_name":   p.part_name,
                "sub_key":     p.sub_key,
                "point_index": p.point_index,
                "t_true":      f"{p.t_true:.6f}",
                "t_pred":      f"{p.t_pred:.6f}",
                "fold_idx":    p.fold_idx,
            })

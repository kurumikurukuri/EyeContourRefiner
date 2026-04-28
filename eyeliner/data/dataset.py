"""EyelinerPatchDataset と manifest 読み込みユーティリティ。"""
from __future__ import annotations

import csv
import hashlib
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eyeliner.data.augmentation import AUG_ALL, augment, augment_endpoints
from eyeliner.data.patch_utils import (
    PATCH_SIZE, T_MIN, T_MAX, IMAGENET_MEAN, IMAGENET_STD,
    extract_patch,
)

logger = logging.getLogger(__name__)


def get_manifest_version(manifest_path: str | Path) -> str:
    """manifest.csv の MD5 ハッシュを返す（データバージョンとして使用）。

    データが変化したことを検知し、過去実験との比較可否を判断するために使う。
    """
    with open(manifest_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def load_manifest(
    path: str | Path,
) -> list[dict]:
    """manifest.csv を読み込み、有効な t 値を持つレコードのリストを返す。

    Args:
        path: manifest.csv のパス

    Returns:
        レコードのリスト（dict のリスト）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest が見つかりません: {path}")

    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                float(row["t"])
            except (ValueError, KeyError):
                continue
            records.append(row)

    return records


def load_manifest_df(
    path: str | Path,
) -> pd.DataFrame:
    """manifest.csv を pandas DataFrame として読み込む。"""
    path = Path(path)
    return pd.read_csv(path)


class EyelinerPatchDataset(Dataset):
    """アイライン輪郭検出用パッチデータセット。

    各サンプルは以下を返す（8-tuple）:
        x               : (3, H, W) float32 正規化済みパッチ
        is_upper        : (1,) float32  上アイライン=1.0, 下=0.0
        pq_ratio        : (1,) float32  pq_dist / eye_scale（目スケールに対するPQ比率）
        y               : (1,) float32  正規化位置 t
        weight          : (1,) float32  訓練損失用重み（目スケール正規化）
        val_weight      : (1,) float32  検証ピクセル誤差用重み
        dist_from_center: (1,) float32  点番号に基づく中心からの距離
        norm_eye_scale  : (1,) float32  eye_scale / sqrt(img_H * img_W)
    """

    def __init__(
        self,
        records: list[dict],
        augment_data: bool = False,
        aug_flags: int = AUG_ALL,
        preload_images: bool = True,
        t_quant_step: float | None = None,
        norm_mean: list | None = None,
        norm_std: list | None = None,
        patch_t_min: float = T_MIN,
        patch_t_max: float = T_MAX,
        patch_perp_scale: float = 1.0,
        patch_size: tuple[int, int] = PATCH_SIZE,
        use_t_norm: bool = True,
        label_t_min: float | None = None,
        label_t_max: float | None = None,
    ) -> None:
        self.records = records
        self.augment_data = augment_data
        self.aug_flags = aug_flags
        self.t_quant_step = t_quant_step
        self.patch_t_min = patch_t_min
        self.patch_t_max = patch_t_max
        # label 範囲: None のときは crop 範囲と同じ（後方互換）
        self._label_t_min = label_t_min if label_t_min is not None else patch_t_min
        self._label_t_max = label_t_max if label_t_max is not None else patch_t_max
        self.use_t_norm = use_t_norm
        self.patch_perp_scale = patch_perp_scale
        self.patch_size = patch_size
        self._image_cache: dict[str, np.ndarray] = {}

        _mean = norm_mean if norm_mean is not None else IMAGENET_MEAN
        _std = norm_std if norm_std is not None else IMAGENET_STD
        self._norm_mean = torch.tensor(_mean).view(3, 1, 1)
        self._norm_std = torch.tensor(_std).view(3, 1, 1)

        if preload_images:
            unique_paths = {rec["image_path"] for rec in records}
            for img_path in unique_paths:
                bgr = cv2.imread(img_path)
                if bgr is not None:
                    self._image_cache[img_path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            logger.info("画像キャッシュ完了: %d 枚", len(self._image_cache))

        # 画像対角長マップ（norm_eye_scale 計算用）
        self._img_diag_map: dict[str, float] = {
            img_path: math.sqrt(img.shape[0] * img.shape[1])
            for img_path, img in self._image_cache.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = rec["image_path"]
        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])

        # manifest の t は生の物理 t 値（PQ 座標系）。
        # label 範囲でクランプして正規化する（crop 範囲とは独立）。
        t_actual = float(rec["t"])
        if self.t_quant_step is not None and self.t_quant_step > 0:
            t_actual = round(t_actual / self.t_quant_step) * self.t_quant_step
        if t_actual < self._label_t_min:
            t_actual = self._label_t_min
        elif t_actual > self._label_t_max:
            t_actual = self._label_t_max

        # 端点ブレンド augmentation（画素 augment よりも前、extract_patch よりも前に適用）。
        # t' が [label_t_min, label_t_max] 外になった場合は identity にフォールバックする。
        if self.augment_data:
            px, py, qx, qy, t_actual = augment_endpoints(
                px, py, qx, qy, t_actual,
                label_t_min=self._label_t_min,
                label_t_max=self._label_t_max,
                aug_flags=self.aug_flags,
            )

        if self.use_t_norm:
            t = (t_actual - self._label_t_min) / (self._label_t_max - self._label_t_min)
        else:
            t = t_actual

        is_upper_val = 1.0 if rec.get("sub_key", "") == "upper" else 0.0

        image = self._image_cache.get(image_path)
        if image is None:
            bgr = cv2.imread(image_path)
            if bgr is not None:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if image is None:
            patch = np.zeros((*self.patch_size, 3), dtype=np.float32)
        else:
            patch = extract_patch(
                image, px, py, qx, qy,
                patch_size=self.patch_size,
                t_min=self.patch_t_min,
                t_max=self.patch_t_max,
                perp_scale=self.patch_perp_scale,
            )

        if self.augment_data:
            if self.use_t_norm:
                patch, t = augment(patch, t, aug_flags=self.aug_flags)
            else:
                patch, t = augment(patch, t, t_min=self._label_t_min, t_max=self._label_t_max, aug_flags=self.aug_flags)

        x = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        x = (x - self._norm_mean) / self._norm_std

        is_upper = torch.tensor([is_upper_val], dtype=torch.float32)
        pq_ratio = torch.tensor([float(rec["pq_ratio"])], dtype=torch.float32)
        y = torch.tensor([t], dtype=torch.float32)
        weight = torch.tensor([float(rec["weight"])], dtype=torch.float32)
        val_weight = torch.tensor([float(rec["val_weight"])], dtype=torch.float32)
        pi = int(rec.get("point_index", 3))
        dist_from_center = torch.tensor([min(pi, 6 - pi) / 3.0], dtype=torch.float32)

        eye_scale = float(rec.get("eye_scale", 1.0))
        img_diag = self._img_diag_map.get(image_path)
        if img_diag is None and image is not None:
            img_diag = math.sqrt(image.shape[0] * image.shape[1])
            self._img_diag_map[image_path] = img_diag
        norm_eye_scale_val = eye_scale / img_diag if img_diag else 0.0
        norm_eye_scale = torch.tensor([norm_eye_scale_val], dtype=torch.float32)

        return x, is_upper, pq_ratio, y, weight, val_weight, dist_from_center, norm_eye_scale

"""EyelinerPatchDualDataset: 2系統パッチ（生PQ + P'Q'）を返すデータセット。"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from eyeliner.data.augmentation import AUG_ALL, ENDPOINT_BLEND_RANGE, augment, augment_endpoints
from eyeliner.data.patch_utils import (
    PATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    extract_patch, compute_pq_rescale,
)

logger = logging.getLogger(__name__)


@dataclass
class BranchConfig:
    """1系統のパッチ抽出設定。"""
    patch_t_min: float
    patch_t_max: float
    label_t_min: float
    label_t_max: float
    pq_rescale_mode: str = "none"
    pq_rescale_factor: float = 1.0
    patch_perp_scale: float = 1.0
    patch_size: tuple[int, int] = field(default_factory=lambda: PATCH_SIZE)
    # ラベル空間: "raw_t"  → 生 t をこの枝の範囲でクランプ（既定）
    #            "t_prime" → P'Q' 座標系の t' を予測（pq_rescale 時の配線）
    label_space: str = "raw_t"


class EyelinerPatchDualDataset(Dataset):
    """2系統パッチ（枝A:生PQ / 枝B:P'Q'）を返すデータセット。

    Returns 14-tuple:
        [0]  x_a             : (3, H, W) 枝A パッチ（生PQ座標系）
        [1]  x_b             : (3, H, W) 枝B パッチ（P'Q'座標系）
        [2]  is_upper        : (1,)
        [3]  pq_ratio        : (1,) 枝A の pq_dist / eye_scale
        [4]  y_a             : (1,) 枝A ラベル空間で正規化した t ∈ [0, 1]
        [5]  y_b             : (1,) 枝B ラベル空間で正規化した値 ∈ [0, 1]
        [6]  weight          : (1,) 訓練損失重み
        [7]  val_weight      : (1,) 検証ピクセル誤差重み
        [8]  dist_from_center: (1,)
        [9]  norm_eye_scale  : (1,)
        [10] point_polar_cos : (1,)
        [11] point_polar_sin : (1,)
        [12] pq_dist         : (1,) 枝A（augmented）の |PQ|
        [13] pq_prime_dist   : (1,) 枝B の |P'Q'| (=eye_scale*factor)。逆変換 t'→t に使用

    枝A は生 t_raw を予測。枝B は cfg_b.label_space により:
        "raw_t"  → 生 t_raw をクランプ（既定）
        "t_prime"→ P'Q' 座標系の t' を予測（pq_dist/pq_prime_dist で t_raw に逆変換可能）
    """

    def __init__(
        self,
        records: list[dict],
        cfg_a: BranchConfig,
        cfg_b: BranchConfig,
        augment_data: bool = False,
        aug_flags: int = AUG_ALL,
        endpoint_blend_range: float = ENDPOINT_BLEND_RANGE,
        preload_images: bool = True,
        norm_mean: list | None = None,
        norm_std: list | None = None,
    ) -> None:
        self.records = records
        self.cfg_a = cfg_a
        self.cfg_b = cfg_b
        self.augment_data = augment_data
        self.aug_flags = aug_flags
        self.endpoint_blend_range = endpoint_blend_range
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

        self._img_diag_map: dict[str, float] = {
            img_path: math.sqrt(img.shape[0] * img.shape[1])
            for img_path, img in self._image_cache.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = rec["image_path"]
        px_raw = float(rec["px"])
        py_raw = float(rec["py"])
        qx_raw = float(rec["qx"])
        qy_raw = float(rec["qy"])
        t_raw = float(rec["t"])

        # Endpoint augmentation: applied once, shared between branches.
        if self.augment_data:
            px_a, py_a, qx_a, qy_a, t_aug = augment_endpoints(
                px_raw, py_raw, qx_raw, qy_raw, t_raw,
                label_t_min=self.cfg_a.label_t_min,
                label_t_max=self.cfg_a.label_t_max,
                aug_flags=self.aug_flags,
                endpoint_blend_range=self.endpoint_blend_range,
            )
        else:
            px_a, py_a, qx_a, qy_a, t_aug = px_raw, py_raw, qx_raw, qy_raw, t_raw

        # Branch A の augmented |PQ|（t'→t 逆変換・記録用）
        pq_dist = math.hypot(qx_a - px_a, qy_a - py_a)

        # Branch B: compute P'Q' from (augmented) branch A endpoints.
        # t_aug (unclamped) is passed to get correct P'Q' positions and t'.
        if self.cfg_b.pq_rescale_mode != "none":
            px_b, py_b, qx_b, qy_b, t_b_prime, pq_prime_dist = compute_pq_rescale(
                px_a, py_a, qx_a, qy_a, t_aug, rec,
                self.cfg_b.pq_rescale_mode, self.cfg_b.pq_rescale_factor,
            )
        else:
            px_b, py_b, qx_b, qy_b = px_a, py_a, qx_a, qy_a
            t_b_prime, pq_prime_dist = t_aug, pq_dist

        # Branch A は生 t_raw を予測（自身の範囲でクランプ）。
        label_range_a = self.cfg_a.label_t_max - self.cfg_a.label_t_min
        t_a = max(self.cfg_a.label_t_min, min(self.cfg_a.label_t_max, t_aug))
        y_a = (t_a - self.cfg_a.label_t_min) / label_range_a

        # Branch B: label_space により予測対象を切替。
        #   "t_prime" → P'Q' 座標系の t'（pq_rescale 時に正しい配線。極端 t を圧縮表現）
        #   "raw_t"   → 生 t_raw
        label_range_b = self.cfg_b.label_t_max - self.cfg_b.label_t_min
        t_b_target = t_b_prime if self.cfg_b.label_space == "t_prime" else t_aug
        t_b = max(self.cfg_b.label_t_min, min(self.cfg_b.label_t_max, t_b_target))
        y_b = (t_b - self.cfg_b.label_t_min) / label_range_b

        image = self._image_cache.get(image_path)
        if image is None:
            bgr = cv2.imread(image_path)
            if bgr is not None:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if image is None:
            patch_a = np.zeros((*self.cfg_a.patch_size, 3), dtype=np.float32)
            patch_b = np.zeros((*self.cfg_b.patch_size, 3), dtype=np.float32)
        else:
            patch_a = extract_patch(
                image, px_a, py_a, qx_a, qy_a,
                patch_size=self.cfg_a.patch_size,
                t_min=self.cfg_a.patch_t_min,
                t_max=self.cfg_a.patch_t_max,
                perp_scale=self.cfg_a.patch_perp_scale,
            )
            patch_b = extract_patch(
                image, px_b, py_b, qx_b, qy_b,
                patch_size=self.cfg_b.patch_size,
                t_min=self.cfg_b.patch_t_min,
                t_max=self.cfg_b.patch_t_max,
                perp_scale=self.cfg_b.patch_perp_scale,
            )

        if self.augment_data:
            # Save random state before augmenting branch A.
            # Restoring for branch B ensures the same flip decision is applied.
            # Flip maps t_norm → 1 - t_norm, which is the same operation for both y_a and y_b.
            py_state = random.getstate()
            np_state = np.random.get_state()

            patch_a, y_a = augment(patch_a, y_a, aug_flags=self.aug_flags)

            random.setstate(py_state)
            np.random.set_state(np_state)
            patch_b, y_b = augment(patch_b, y_b, aug_flags=self.aug_flags)

        x_a = torch.from_numpy(patch_a.transpose(2, 0, 1)).float()
        x_a = (x_a - self._norm_mean) / self._norm_std
        x_b = torch.from_numpy(patch_b.transpose(2, 0, 1)).float()
        x_b = (x_b - self._norm_mean) / self._norm_std

        is_upper_val = 1.0 if rec.get("sub_key", "") == "upper" else 0.0
        is_upper = torch.tensor([is_upper_val], dtype=torch.float32)
        pq_ratio = torch.tensor([float(rec["pq_ratio"])], dtype=torch.float32)
        y_a_t = torch.tensor([y_a], dtype=torch.float32)
        y_b_t = torch.tensor([y_b], dtype=torch.float32)
        weight = torch.tensor([float(rec["weight"])], dtype=torch.float32)
        val_weight = torch.tensor([float(rec["val_weight"])], dtype=torch.float32)

        pi = int(rec.get("point_index", 3))
        dist_from_center = torch.tensor([min(pi, 6 - pi) / 3.0], dtype=torch.float32)

        degree = (pi + 1) * 22.5 if is_upper_val == 1.0 else (pi + 9) * 22.5
        rad = degree * math.pi / 180.0
        point_polar_cos = torch.tensor([math.cos(rad)], dtype=torch.float32)
        point_polar_sin = torch.tensor([math.sin(rad)], dtype=torch.float32)

        eye_scale = float(rec.get("eye_scale", 1.0))
        img_diag = self._img_diag_map.get(image_path)
        if img_diag is None and image is not None:
            img_diag = math.sqrt(image.shape[0] * image.shape[1])
            self._img_diag_map[image_path] = img_diag
        norm_eye_scale_val = eye_scale / img_diag if img_diag else 0.0
        norm_eye_scale = torch.tensor([norm_eye_scale_val], dtype=torch.float32)

        pq_dist_t = torch.tensor([pq_dist], dtype=torch.float32)
        pq_prime_dist_t = torch.tensor([pq_prime_dist], dtype=torch.float32)

        return (
            x_a, x_b, is_upper, pq_ratio,
            y_a_t, y_b_t,
            weight, val_weight,
            dist_from_center, norm_eye_scale, point_polar_cos, point_polar_sin,
            pq_dist_t, pq_prime_dist_t,
        )

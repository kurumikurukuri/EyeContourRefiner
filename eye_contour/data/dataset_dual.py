"""EyeContourPatchDualDataset: 2系統パッチ（生PQ + P'Q'）を返すデータセット。"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from eye_contour.data.augmentation import (
    AUG_ALL, ENDPOINT_BLEND_RANGE, augment, augment_endpoints, augment_pq_jitter,
)
from eye_contour.data.source_degradation import (
    SourceDegradeConfig,
    SourceDegradeSample,
    compute_patch_source_bbox,
    degrade_region,
    sample_source_degradation,
)
from eye_contour.data.patch_utils import (
    PATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    extract_patch, compute_pq_rescale,
    compute_t_from_point, reconstruct_point,
    homography_from_pitch_yaw, warp_pq_with_homography,
)
from eye_contour.data.weights import DEFAULT_AUX_FEATURES, build_aux_vector

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


class EyeContourPatchDualDataset(Dataset):
    """2系統パッチ（枝A:生PQ / 枝B:P'Q'）を返すデータセット。

    Returns 14-tuple (use_aux_face_shape=False) or 15-tuple ([14]=aux, use_aux_face_shape=True):
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
        [14] aux_features    : (11,) 生 sin/cos aux ベクトル（use_aux_face_shape 時のみ）

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
        aug_prob: float = 1.0,
        endpoint_blend_range: float = ENDPOINT_BLEND_RANGE,
        preload_images: bool = True,
        norm_mean: list | None = None,
        norm_std: list | None = None,
        use_aux_face_shape: bool = False,
        aux_features: list[str] | None = None,
        perspective_alpha: float = 2.0,
        source_degrade: SourceDegradeConfig | None = None,
        source_degrade_seed: int = 42,
        pq_jitter_rot_deg: float = 0.0,
        pq_jitter_prob: float = 1.0,
        pq_jitter_seed: int = 42,
    ) -> None:
        self.records = records
        self.perspective_alpha = perspective_alpha
        self.cfg_a = cfg_a
        self.cfg_b = cfg_b
        self.augment_data = augment_data
        self.aug_flags = aug_flags
        self.aug_prob = aug_prob
        self.endpoint_blend_range = endpoint_blend_range
        # EXP035: PQ 回転摂動。AUG_PQ_JITTER ビットと rot_deg>0 の両方が必要。
        # 他 aug の乱数列を乱さないよう専用 RNG を持つ。
        self.pq_jitter_rot_deg = pq_jitter_rot_deg
        self.pq_jitter_prob = pq_jitter_prob
        self._pq_jitter_rng = random.Random(pq_jitter_seed)
        self.use_aux_face_shape = use_aux_face_shape
        # aux_features 指定時はその部分集合、未指定で use_aux_face_shape ならデフォルト11次元。
        self.aux_features = list(aux_features) if aux_features else (
            list(DEFAULT_AUX_FEATURES) if use_aux_face_shape else []
        )

        # EXP031: 元画像ピクセル空間の劣化 aug（JPEG / 解像度ジッタ）。
        # augment_data=True のときのみ発火する。既存 aug の乱数列を乱さないよう
        # 専用の Generator を持たせる（config 間で photometric aug の系列が揃う）。
        self._source_degrade = source_degrade if (source_degrade and source_degrade.enabled) else None
        self._deg_rng = random.Random(source_degrade_seed)
        self._persp_conflict_warned = False

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

    def _extract_patch_maybe_degraded(
        self,
        image: np.ndarray,
        px: float, py: float, qx: float, qy: float,
        cfg: BranchConfig,
        H: np.ndarray | None,
        deg: SourceDegradeSample | None,
    ) -> np.ndarray:
        """パッチを抽出する。deg が指定された場合は元画像 ROI に劣化を掛けてから抽出する。

        ROI は footprint を完全に含むため、劣化なしの場合は ROI 経由でも
        元画像から直接抽出した場合と同一のパッチになる。
        """
        if deg is not None and not deg.is_identity:
            bbox = compute_patch_source_bbox(
                px, py, qx, qy,
                cfg.patch_t_min, cfg.patch_t_max, cfg.patch_perp_scale,
            )
            degraded = degrade_region(image, bbox, deg) if bbox is not None else None
            if degraded is not None:
                roi, x0, y0 = degraded
                return extract_patch(
                    roi, px - x0, py - y0, qx - x0, qy - y0,
                    patch_size=cfg.patch_size,
                    t_min=cfg.patch_t_min,
                    t_max=cfg.patch_t_max,
                    perp_scale=cfg.patch_perp_scale,
                    H=None,
                )
        return extract_patch(
            image, px, py, qx, qy,
            patch_size=cfg.patch_size,
            t_min=cfg.patch_t_min,
            t_max=cfg.patch_t_max,
            perp_scale=cfg.patch_perp_scale,
            H=H,
        )

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

        # PQ 回転摂動（EXP035）: R 中心の回転なので t は不変。endpoint blend の後、perspective の前。
        if self.augment_data:
            px_a, py_a, qx_a, qy_a, t_aug = augment_pq_jitter(
                px_a, py_a, qx_a, qy_a, t_aug,
                aug_flags=self.aug_flags,
                rot_deg=self.pq_jitter_rot_deg,
                prob=self.pq_jitter_prob,
                rng=self._pq_jitter_rng,
            )

        # Branch A の augmented |PQ|（t'→t 逆変換・記録用）
        pq_dist = math.hypot(qx_a - px_a, qy_a - py_a)

        # 面外回転（pitch/yaw）透視変換 aug。
        # 展開済みレコードに _persp_dp/_persp_dy タグがある場合のみ適用（train 専用）。
        H_persp: np.ndarray | None = None
        if self.augment_data and "_persp_dp" in rec:
            dp_deg = float(rec["_persp_dp"])
            dy_deg = float(rec["_persp_dy"])
            if rec.get("nose_x") is not None and rec.get("nose_y") is not None:
                # 鼻先（顔中心）を pivot にした回転。目領域の orbit 移動を含む。
                cx = float(rec["nose_x"])
                cy = float(rec["nose_y"])
                eye_scale = float(rec.get("eye_scale", 1.0))
                H_persp = homography_from_pitch_yaw(dp_deg, dy_deg, cx, cy, eye_scale, alpha=self.perspective_alpha)
                # R を現在の t から復元し、P, Q, R を一括ワープ
                rx_a, ry_a = reconstruct_point(px_a, py_a, qx_a, qy_a, t_aug)
                px_a, py_a, qx_a, qy_a, rx_a, ry_a = warp_pq_with_homography(
                    H_persp, px_a, py_a, qx_a, qy_a, rx_a, ry_a
                )
                # ワープ後の t を再計算（透視変換は比率非保存のため t' ≠ t）
                t_aug = compute_t_from_point(px_a, py_a, qx_a, qy_a, rx_a, ry_a)
                pq_dist = math.hypot(qx_a - px_a, qy_a - py_a)
            else:
                # nose 座標が未抽出のレコードは homography をスキップ。
                # 事前に extract_nose_coords.py を実行していれば到達しない。
                logger.warning(
                    "nose_x/nose_y が未設定のため perspective aug をスキップ: idx=%s image=%s",
                    rec.get("idx", "?"), rec.get("image_path", "?"),
                )

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

        # EXP031: 元画像空間の劣化パラメータを 1 サンプルにつき 1 回だけ引き、
        # 枝A・枝B で同一のものを共有する（両枝で異なる劣化が乗るのを防ぐ）。
        # 画像読み込みの成否によらず引くことで、乱数列がサンプル数と同期する。
        deg_sample: SourceDegradeSample | None = None
        if self.augment_data and self._source_degrade is not None:
            if H_persp is not None:
                # perspective aug 併用時は ROI が homography の逆像になるため未対応。
                if not self._persp_conflict_warned:
                    logger.warning(
                        "perspective_aug と source degradation の併用は未対応のため "
                        "劣化をスキップします（EXP031 では perspective_aug=false 前提）"
                    )
                    self._persp_conflict_warned = True
            else:
                deg_sample = sample_source_degradation(self._source_degrade, self._deg_rng)

        image = self._image_cache.get(image_path)
        if image is None:
            bgr = cv2.imread(image_path)
            if bgr is not None:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if image is None:
            patch_a = np.zeros((*self.cfg_a.patch_size, 3), dtype=np.float32)
            patch_b = np.zeros((*self.cfg_b.patch_size, 3), dtype=np.float32)
        else:
            patch_a = self._extract_patch_maybe_degraded(
                image, px_a, py_a, qx_a, qy_a, self.cfg_a, H_persp, deg_sample,
            )
            patch_b = self._extract_patch_maybe_degraded(
                image, px_b, py_b, qx_b, qy_b, self.cfg_b, H_persp, deg_sample,
            )

        if self.augment_data:
            # Save random state before augmenting branch A.
            # Restoring for branch B ensures the same flip decision is applied.
            # Flip maps t_norm → 1 - t_norm, which is the same operation for both y_a and y_b.
            py_state = random.getstate()
            np_state = np.random.get_state()

            patch_a, y_a = augment(patch_a, y_a, aug_flags=self.aug_flags, aug_prob=self.aug_prob)

            random.setstate(py_state)
            np.random.set_state(np_state)
            patch_b, y_b = augment(patch_b, y_b, aug_flags=self.aug_flags, aug_prob=self.aug_prob)

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

        out = (
            x_a, x_b, is_upper, pq_ratio,
            y_a_t, y_b_t,
            weight, val_weight,
            dist_from_center, norm_eye_scale, point_polar_cos, point_polar_sin,
            pq_dist_t, pq_prime_dist_t,
        )
        # aux ベクトルを付与（use_aux_face_shape のフル11次元、または aux_features の部分集合）。
        # 生 sin/cos（z-score なし）。pose は DB 由来、
        # polar / pq_angle は compute_aux_face_shape_meta が付与。
        if self.aux_features:
            aux = torch.tensor(build_aux_vector(rec, self.aux_features), dtype=torch.float32)
            return (*out, aux)
        return out

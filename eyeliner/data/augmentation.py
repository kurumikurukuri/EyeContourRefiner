"""データ拡張（訓練時のみ使用）。"""
from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
from scipy.ndimage import gaussian_filter

# オーグメンテーションのビットフラグ定数
AUG_BRIGHTNESS: int = 0x01  # 輝度スケール・コントラスト・ガンマ補正
AUG_BLUR:       int = 0x02  # ガウスぼかし
AUG_NOISE:      int = 0x04  # ガウスノイズ
AUG_COLOR_TEMP: int = 0x08  # 色温度変換（ホワイトバランス）
AUG_FLIP:       int = 0x40  # 水平反転

AUG_ALL:  int = 0x4F  # 全オーグメンテーション有効（BRIGHTNESS|BLUR|NOISE|COLOR_TEMP|FLIP）
AUG_NONE: int = 0x00  # 全オーグメンテーション無効


def augment(
    patch: np.ndarray,
    t: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
    aug_flags: int = AUG_ALL,
) -> tuple[np.ndarray, float]:
    """パッチと t 値にランダムなデータ拡張を適用する。

    aug_flags のビットで各オーグメンテーションを個別に ON/OFF する。

        AUG_BRIGHTNESS (0x01): 輝度スケール・コントラスト・ガンマ補正
        AUG_BLUR       (0x02): ガウスぼかし
        AUG_NOISE      (0x04): ガウスノイズ
        AUG_COLOR_TEMP (0x08): 色温度変換（ホワイトバランス）
        AUG_FLIP       (0x40): 水平反転（t → t_min + t_max - t）

    Args:
        patch:     float32 パッチ [0, 1] の範囲、shape (H, W, 3)
        t:         位置 t ∈ [t_min, t_max]
        t_min:     t の最小値（正規化時は 0.0、raw 時は patch_t_min）
        t_max:     t の最大値（正規化時は 1.0、raw 時は patch_t_max）
        aug_flags: ビットフラグ（デフォルト AUG_ALL=0x4F で全て有効）

    Returns:
        (拡張後パッチ, 拡張後 t)
    """
    if not (t_min <= t <= t_max):
        logger.warning("t=%s が範囲 [%s, %s] 外 — クランプして続行", t, t_min, t_max)
        t = max(t_min, min(t_max, t))

    if aug_flags & AUG_BRIGHTNESS:
        patch = patch * random.uniform(0.8, 1.2)
        mean = patch.mean()
        patch = (patch - mean) * random.uniform(0.8, 1.2) + mean
        patch = np.clip(patch, 0, 1)
        gamma = random.uniform(0.7, 1.4)
        patch = patch**gamma

    if (aug_flags & AUG_BLUR) and random.random() < 0.5:
        sigma = random.uniform(0.3, 1.0)
        patch = patch.copy()
        for c in range(patch.shape[2]):
            patch[:, :, c] = gaussian_filter(patch[:, :, c], sigma=sigma)

    if aug_flags & AUG_NOISE:
        sigma = random.uniform(0.0, 5.0) / 255.0
        patch = patch + np.random.default_rng().standard_normal(patch.shape, dtype=np.float32) * sigma

    if (aug_flags & AUG_COLOR_TEMP) and random.random() < 0.5:
        r_scale = random.uniform(0.85, 1.15)
        b_scale = 2.0 - r_scale
        patch = patch.copy()
        patch[:, :, 0] = np.clip(patch[:, :, 0] * r_scale, 0, 1)
        patch[:, :, 2] = np.clip(patch[:, :, 2] * b_scale, 0, 1)

    if (aug_flags & AUG_FLIP) and random.random() < 0.5:
        patch = patch[:, ::-1].copy()  # .copy() で連続配列にする
        t = t_min + t_max - t

    patch = np.clip(patch, 0, 1).astype(np.float32)
    return patch, t

"""データ拡張（訓練時のみ使用）。"""
from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
from scipy.ndimage import gaussian_filter

# オーグメンテーションのビットフラグ定数
AUG_BRIGHTNESS:      int = 0x01  # 輝度スケール・コントラスト・ガンマ補正
AUG_BLUR:            int = 0x02  # ガウスぼかし
AUG_NOISE:           int = 0x04  # ガウスノイズ
AUG_COLOR_TEMP:      int = 0x08  # 色温度変換（ホワイトバランス）
AUG_GRAYSCALE:       int = 0x10  # グレースケール化（輝度のみに変換して3ch複製）
AUG_ENDPOINT_BLEND:  int = 0x20  # 端点 P, Q の線形ブレンド（R は幾何保存）
AUG_FLIP:            int = 0x40  # 水平反転

# --- 元画像空間の劣化（EXP031）--------------------------------------------
# bit0-6 は augment() がパッチに適用するが、以下 2 つは **ワープ前の元画像 ROI**
# に適用するため適用箇所が違う（dataset_dual の _extract_patch_maybe_degraded 経路、
# 実処理は eyeliner/data/source_degradation.py）。augment() はこれらのビットを見ない。
# quality レンジ等の連続パラメータは source_jpeg_quality_min など別キーで指定する。
AUG_SOURCE_JPEG:       int = 0x080  # 元画像空間 JPEG 圧縮
AUG_SOURCE_RES_JITTER: int = 0x100  # 元画像空間 解像度ジッタ（縮小→拡大）

AUG_ALL:        int = 0x1FF  # 全オーグメンテーション有効
AUG_NONE:       int = 0x000  # 全オーグメンテーション無効


def check_aug_flags(aug_flags: int) -> None:
    """未定義ビットが立っていたら警告する（値そのものは変更しない）。

    ビット演算で判定しているだけなので範囲外の値はエラーにならず黙って無視される。
    設定ミス（例: 存在しないビットに劣化 aug を割り当てたつもり）を検出するための保険。
    """
    unknown = aug_flags & ~AUG_ALL
    if unknown:
        logger.warning(
            "aug_flags に未定義ビットが含まれています: 0x%X（未定義部 0x%X）— 無視されます",
            aug_flags, unknown,
        )

ENDPOINT_BLEND_RANGE: float = 0.25  # a, b ∈ [-ENDPOINT_BLEND_RANGE, ENDPOINT_BLEND_RANGE]


def augment(
    patch: np.ndarray,
    t: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
    aug_flags: int = AUG_ALL,
    aug_prob: float = 1.0,
) -> tuple[np.ndarray, float]:
    """パッチと t 値にランダムなデータ拡張を適用する。

    aug_flags のビットで各オーグメンテーションを個別に ON/OFF する。

        AUG_BRIGHTNESS (0x01): 輝度スケール・コントラスト・ガンマ補正（基準確率 0.5）
        AUG_BLUR       (0x02): ガウスぼかし（基準確率 0.5）
        AUG_NOISE      (0x04): ガウスノイズ（常時発火、aug_prob の対象外）
        AUG_COLOR_TEMP (0x08): 色温度変換（ホワイトバランス、基準確率 0.5）
        AUG_GRAYSCALE  (0x10): グレースケール化（輝度のみ・確率 0.2、aug_prob の対象外）
        AUG_FLIP       (0x40): 水平反転（t → t_min + t_max - t、基準確率 0.5）

    AUG_SOURCE_JPEG (0x080) / AUG_SOURCE_RES_JITTER (0x100) は元画像 ROI に適用する
    劣化なのでここでは解釈しない（dataset 側が処理する）。渡されても無視される。

    Args:
        patch:     float32 パッチ [0, 1] の範囲、shape (H, W, 3)
        t:         位置 t ∈ [t_min, t_max]
        t_min:     t の最小値（正規化時は 0.0、raw 時は patch_t_min）
        t_max:     t の最大値（正規化時は 1.0、raw 時は patch_t_max）
        aug_flags: ビットフラグ（デフォルト AUG_ALL=0x1FF で全て有効）
        aug_prob:  BRIGHTNESS/BLUR/COLOR_TEMP/FLIP（基準確率 0.5 のもの）の発火確率に
                   掛ける倍率。0.5*aug_prob を [0, 1] にクランプして使う（EXP030 発火頻度
                   sweep用）。NOISE（常時発火）・GRAYSCALE（基準0.2）は対象外で固定のまま。

    Returns:
        (拡張後パッチ, 拡張後 t)
    """
    if not (t_min <= t <= t_max):
        logger.warning("t=%s が範囲 [%s, %s] 外 — クランプして続行", t, t_min, t_max)
        t = max(t_min, min(t_max, t))

    fire_p = min(1.0, max(0.0, 0.5 * aug_prob))

    if (aug_flags & AUG_BRIGHTNESS) and random.random() < fire_p:
        patch = patch * random.uniform(0.8, 1.2)
        mean = patch.mean()
        patch = (patch - mean) * random.uniform(0.8, 1.2) + mean
        patch = np.clip(patch, 0, 1)
        gamma = random.uniform(0.7, 1.4)
        patch = patch**gamma

    if (aug_flags & AUG_BLUR) and random.random() < fire_p:
        sigma = random.uniform(0.3, 1.0)
        patch = patch.copy()
        for c in range(patch.shape[2]):
            patch[:, :, c] = gaussian_filter(patch[:, :, c], sigma=sigma)

    if aug_flags & AUG_NOISE:
        sigma = random.uniform(0.0, 5.0) / 255.0
        patch = patch + np.random.standard_normal(patch.shape).astype(np.float32) * sigma

    if (aug_flags & AUG_COLOR_TEMP) and random.random() < fire_p:
        r_scale = random.uniform(0.85, 1.15)
        b_scale = 2.0 - r_scale
        patch = patch.copy()
        patch[:, :, 0] = np.clip(patch[:, :, 0] * r_scale, 0, 1)
        patch[:, :, 2] = np.clip(patch[:, :, 2] * b_scale, 0, 1)

    if (aug_flags & AUG_GRAYSCALE) and random.random() < 0.2:
        # ITU-R BT.601 輝度係数
        gray = (0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2])
        patch = np.stack([gray, gray, gray], axis=2).astype(np.float32)

    if (aug_flags & AUG_FLIP) and random.random() < fire_p:
        patch = patch[:, ::-1].copy()  # .copy() で連続配列にする
        t = t_min + t_max - t

    patch = np.clip(patch, 0, 1).astype(np.float32)
    return patch, t


def augment_endpoints(
    px: float,
    py: float,
    qx: float,
    qy: float,
    t: float,
    label_t_min: float,
    label_t_max: float,
    aug_flags: int = AUG_ALL,
    endpoint_blend_range: float = ENDPOINT_BLEND_RANGE,
) -> tuple[float, float, float, float, float]:
    """端点 P, Q を独立に線形混合して擾乱を与える（アノテーション点 R は幾何保存）。

        Q' = a·P + (1-a)·Q
        P' = b·Q + (1-b)·P
        t' = (t - a) / (1 - a - b)

    ここで a, b ∈ [-ENDPOINT_BLEND_RANGE, ENDPOINT_BLEND_RANGE] を独立にサンプリングする。
    1 - a - b ∈ [0.5, 1.5] で常に非零、向きを保ったままスケール変化のみ生じる。

    計算後の t' が [label_t_min, label_t_max] を外れた場合は、そのサンプルについて
    端点ブレンドを適用せず元の (px, py, qx, qy, t) を返す（identity fallback）。

    AUG_ENDPOINT_BLEND フラグが OFF のときも identity を返す。
    """
    if not (aug_flags & AUG_ENDPOINT_BLEND):
        return px, py, qx, qy, t

    a = random.uniform(-endpoint_blend_range, endpoint_blend_range)
    b = random.uniform(-endpoint_blend_range, endpoint_blend_range)
    denom = 1.0 - a - b
    new_t = (t - a) / denom

    if new_t < label_t_min or new_t > label_t_max:
        return px, py, qx, qy, t

    new_qx = a * px + (1.0 - a) * qx
    new_qy = a * py + (1.0 - a) * qy
    new_px = b * qx + (1.0 - b) * px
    new_py = b * qy + (1.0 - b) * py
    return new_px, new_py, new_qx, new_qy, new_t

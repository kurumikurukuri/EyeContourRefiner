"""
Shared patch extraction utilities used by both the annotation server
and the training pipeline.

Mirrors the logic in the main library's patch_extractor.py so that
this repository can be used standalone without installing the library.
"""
from typing import Tuple

import cv2
import numpy as np

PATCH_SIZE: Tuple[int, int] = (64, 128)  # (height, width)

T_MIN: float = -1.0   # t の下限（Q を超えて外側 100%）
T_MAX: float =  2.0   # t の上限（P を超えて外側 100%）

# ImageNet 正規化定数（学習・推論・サーバーで共通）
IMAGENET_MEAN: list = [0.485, 0.456, 0.406]
IMAGENET_STD: list  = [0.229, 0.224, 0.225]

# データセット固有の正規化定数（scripts/compute_dataset_stats.py で計算）
# 未計算の場合は None のままにしておき、学習時に動的ロードする
DATASET_MEAN: list | None = None
DATASET_STD:  list | None = None

NORM_TYPE_CHOICES = ["imagenet", "dataset", "none"]


def get_norm_stats(norm_type: str, stats_json_path=None) -> tuple[list, list]:
    """正規化タイプに応じた mean・std を返す。

    norm_type:
      "imagenet" : ImageNet 統計 (デフォルト)
      "dataset"  : データセット固有統計 (stats_json_path から読み込み)
      "none"     : 正規化なし (mean=0, std=1)

    Returns: (mean, std) — 各チャンネル [R, G, B] の list[float]
    """
    if norm_type == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD

    if norm_type == "none":
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    if norm_type == "dataset":
        if stats_json_path is not None:
            import json
            from pathlib import Path
            p = Path(stats_json_path)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                return data["dataset"]["mean"], data["dataset"]["std"]

        # モジュールレベル定数があれば使用
        if DATASET_MEAN is not None and DATASET_STD is not None:
            return DATASET_MEAN, DATASET_STD

        raise FileNotFoundError(
            "データセット統計が見つかりません。"
            "先に scripts/compute_dataset_stats.py を実行するか、"
            "--norm_stats_json でJSONパスを指定してください。"
        )

    raise ValueError(f"未知の norm_type: {norm_type!r}  選択肢: {NORM_TYPE_CHOICES}")


def extract_patch(
    image: np.ndarray,
    px: float,
    py: float,
    qx: float,
    qy: float,
    patch_size: Tuple[int, int] = PATCH_SIZE,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    perp_scale: float = 1.0,
) -> np.ndarray:
    """
    Extract a rotated ROI patch aligned along P->Q, covering t ∈ [t_min, t_max].

    The affine transform maps:
      t=t_max point  →  (0,          patch_h // 2)   [left edge]
      t=t_min point  →  (patch_w-1,  patch_h // 2)   [right edge]

    Within the patch:
      t=1.0 (R=P) ↔ patch_x = (t_max-1)/(t_max-t_min) * (patch_w-1)
      t=0.0 (R=Q) ↔ patch_x = (t_max-0)/(t_max-t_min) * (patch_w-1)

    Args:
        image: RGB image (H, W, 3) uint8.
        px, py: Landmark P in image pixel coordinates.
        qx, qy: Delta landmark Q in image pixel coordinates.
        patch_size: (height, width) of the output patch.
        t_min: Lower bound of t range (default T_MIN=-0.5).
        t_max: Upper bound of t range (default T_MAX=1.5).
        perp_scale: 縦軸カバレッジ = perp_scale * |PQ| 画像ピクセル → patch_h パッチピクセル。
                    値を大きくすると目の上下をより広く取り込む（default: 1.0）。

    Returns:
        float32 patch with values in [0, 1], shape (patch_h, patch_w, 3).
        If P==Q, returns a uniform 0.5 patch.
    """
    patch_h, patch_w = patch_size
    dx, dy = qx - px, qy - py
    norm = float(np.hypot(dx, dy))

    if norm == 0.0:
        return np.full((patch_h, patch_w, 3), 0.5, dtype=np.float32)

    ux, uy = dx / norm, dy / norm
    # t=t_max の画像座標を原点にして、範囲 (t_max-t_min) 分のスケールで変換
    scale = (patch_w - 1) / ((t_max - t_min) * norm)
    sx = t_max * px + (1.0 - t_max) * qx   # t=t_max 点の x
    sy = t_max * py + (1.0 - t_max) * qy   # t=t_max 点の y

    # 縦軸カバレッジ = perp_scale * |PQ| 画像ピクセル → patch_h パッチピクセル
    actual_perp = (patch_h - 1) / (perp_scale * norm)

    M = np.float32([
        [scale * ux,            scale * uy,            -scale * (ux * sx + uy * sy)],
        [-uy * actual_perp,     ux * actual_perp,       actual_perp * (uy * sx - ux * sy) + patch_h // 2],
    ])

    patch = cv2.warpAffine(
        image, M, (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return patch.astype(np.float32) / 255.0


def compute_t_from_point(
    px: float, py: float,
    qx: float, qy: float,
    rx: float, ry: float,
) -> float:
    """
    t such that R = t·P + (1-t)·Q.

    t = dot(R-Q, P-Q) / |P-Q|^2,  clamped to [T_MIN, T_MAX].
    Returns 1.0 when P==Q.
    """
    pq_x, pq_y = px - qx, py - qy
    norm_sq = pq_x ** 2 + pq_y ** 2
    if norm_sq == 0.0:
        return 1.0
    rq_x, rq_y = rx - qx, ry - qy
    t = (rq_x * pq_x + rq_y * pq_y) / norm_sq
    return float(np.clip(t, T_MIN, T_MAX))


def reconstruct_point(
    px: float, py: float,
    qx: float, qy: float,
    t: float,
) -> Tuple[float, float]:
    """R = t·P + (1-t)·Q."""
    return t * px + (1.0 - t) * qx, t * py + (1.0 - t) * qy


def build_display_patch(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    display_scale: int = 6,
) -> np.ndarray:
    """
    Build a display-size RGB patch with the P and Q markers overlaid.

    P marker: blue vertical line at x ≈ 25% of patch width (t=1.0).
    Q marker: red  vertical line at x ≈ 75% of patch width (t=0.0).

    Args:
        image_rgb: RGB image (H, W, 3) uint8.
        px, py, qx, qy: Landmark coordinates.
        display_scale: Integer scale factor for display (default 6×).

    Returns:
        uint8 RGB image of shape (patch_h*scale, patch_w*scale, 3).
    """
    patch_h, patch_w = PATCH_SIZE
    patch = extract_patch(image_rgb, px, py, qx, qy)
    patch_u8 = (patch * 255).clip(0, 255).astype(np.uint8)

    disp_h = patch_h * display_scale
    disp_w = patch_w * display_scale
    disp_rgb = cv2.resize(patch_u8, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    # P marker (blue) at t=1.0: x = (T_MAX-1)/(T_MAX-T_MIN) * (disp_w-1) ≈ 25%
    p_x = int(round((T_MAX - 1.0) / (T_MAX - T_MIN) * (disp_w - 1)))
    cv2.line(disp_rgb, (p_x, 0), (p_x, disp_h - 1), (0, 0, 255), 2)
    # Q marker (red) at t=0.0: x = (T_MAX-0)/(T_MAX-T_MIN) * (disp_w-1) ≈ 75%
    q_x = int(round(T_MAX / (T_MAX - T_MIN) * (disp_w - 1)))
    cv2.line(disp_rgb, (q_x, 0), (q_x, disp_h - 1), (255, 0, 0), 2)

    return disp_rgb


def overlay_landmarks(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    rx: float | None = None,
    ry: float | None = None,
) -> np.ndarray:
    """
    Draw P (blue), Q (red), and optionally R (green) on a copy of the image.

    Returns a new RGB uint8 array; does not modify the input.
    """
    out = image_rgb.copy()
    cv2.circle(out, (int(px), int(py)), 2, (0, 0, 255), -1)   # P blue
    cv2.circle(out, (int(qx), int(qy)), 2, (255, 0, 0), -1)   # Q red
    if rx is not None and ry is not None:
        cv2.circle(out, (int(rx), int(ry)), 2, (0, 255, 0), -1)  # R green
    cv2.line(out, (int(px), int(py)), (int(qx), int(qy)), (200, 200, 0), 1)
    return out

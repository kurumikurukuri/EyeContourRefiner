"""
Shared patch extraction utilities used by both the annotation server
and the training pipeline.

Mirrors the logic in the main library's patch_extractor.py so that
this repository can be used standalone without installing the library.
"""
import math
from typing import Tuple

import cv2
import numpy as np

PATCH_SIZE: Tuple[int, int] = (64, 128)  # (height, width)

# EXP020 の pq_rescale（P'Q' 座標系）導入に伴い拡張。invert_pq_rescale_t() で
# P'Q' 座標系の t' を生 PQ 座標系に逆変換すると、pq_rescale_factor > 1 の場合に
# 旧範囲 (-1.0, 2.0) を超える値になり得るため、それを許容できる範囲まで広げてある。
T_MIN: float = -3.0   # t の下限（Q を超えて外側 300%）
T_MAX: float =  4.0   # t の上限（P を超えて外側 300%）

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


def homography_from_pitch_yaw(
    dp_deg: float,
    dy_deg: float,
    cx: float,
    cy: float,
    eye_scale: float,
    alpha: float = 2.0,
) -> np.ndarray:
    """Homography for a planar face patch rotating (pitch/yaw) about its own center.

    Models a FIXED camera viewing a fronto-parallel plane (the eye region) that
    tilts in 3D about the 3D point projecting to (cx, cy) — i.e. the eye stays
    put on the sensor while its apparent shape foreshortens (keystone effect).
    This is the correct convention for "simulate the face turning in place".

    It intentionally differs from a camera-rotation homography (H = K·R·K⁻¹,
    which pans the optical axis): that formulation does NOT fix (cx, cy) when
    rotation is nonzero, so for a fixed rotation angle its pivot drifts by
    f·tan(angle) — growing unboundedly as alpha increases, instead of the
    bounded, orthographic-limit distortion this function guarantees.

    Rotation axes: pitch = around horizontal image axis (X), yaw = around vertical image axis (Y).
    Focal length f = alpha * eye_scale sets the virtual depth of the image plane:
    larger alpha (more distant / more telephoto) → weaker keystone effect,
    converging to a pure affine foreshortening (scale by cos(angle), no shift)
    as alpha → ∞. Smaller alpha (closer / wider-angle) → stronger, more visible
    perspective distortion.

    Args:
        dp_deg: Pitch angle in degrees (rotation around horizontal X axis).
        dy_deg: Yaw angle in degrees (rotation around vertical Y axis).
        cx, cy: Eye center (rotation pivot) in image pixels.
        eye_scale: Eye scale in pixels (bounding-box diagonal).
        alpha: Focal length multiplier (f = alpha * eye_scale). Default 2.0.

    Returns:
        H: (3, 3) float64 homography matrix. H = I when dp_deg = dy_deg = 0.
        H always fixes (cx, cy): H @ [cx, cy, 1]^T ∝ [cx, cy, 1]^T.
    """
    f = max(alpha * eye_scale, 1.0)

    dp = math.radians(dp_deg)
    dy = math.radians(dy_deg)

    cos_p, sin_p = math.cos(dp), math.sin(dp)
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
        dtype=np.float64,
    )
    cos_y, sin_y = math.cos(dy), math.sin(dy)
    Ry = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float64,
    )

    R = Ry @ Rx

    # Homography (in coordinates relative to the pivot) for a plane rotating by
    # R about a point at depth f along the principal ray, viewed by a pinhole
    # camera with focal length f: [u2;v2;1] ~ H_rel @ [u1;v1;1]. By
    # construction H_rel(0,0) = (0,0), i.e. the pivot is a fixed point.
    H_rel = np.array([
        [R[0, 0], R[0, 1], 0.0],
        [R[1, 0], R[1, 1], 0.0],
        [R[2, 0] / f, R[2, 1] / f, 1.0],
    ], dtype=np.float64)

    T = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_inv = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return T @ H_rel @ T_inv


def warp_pq_with_homography(
    H: np.ndarray,
    px: float,
    py: float,
    qx: float,
    qy: float,
    rx: float,
    ry: float,
) -> Tuple[float, float, float, float, float, float]:
    """Apply homography H to P, Q, R using homogeneous coordinates.

    Args:
        H: (3, 3) homography matrix.
        px, py: Landmark P in image pixel coordinates.
        qx, qy: Landmark Q in image pixel coordinates.
        rx, ry: Point R = t·P + (1-t)·Q in image pixel coordinates.

    Returns:
        (px', py', qx', qy', rx', ry'): Warped coordinates.
    """
    def _apply(x: float, y: float) -> Tuple[float, float]:
        v = H @ np.array([x, y, 1.0], dtype=np.float64)
        return float(v[0] / v[2]), float(v[1] / v[2])

    px_, py_ = _apply(px, py)
    qx_, qy_ = _apply(qx, qy)
    rx_, ry_ = _apply(rx, ry)
    return px_, py_, qx_, qy_, rx_, ry_


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
    H: np.ndarray | None = None,
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
        t_min: Lower bound of t range (default T_MIN=-3.0).
        t_max: Upper bound of t range (default T_MAX=4.0).
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

    if H is not None:
        # Fused perspective warp: M_fused = M_affine_3×3 @ H.
        # M_affine maps warped-image coords (P', Q') to patch coords.
        # Composing with H maps original-image coords to patch coords in one step.
        M_3x3 = np.zeros((3, 3), dtype=np.float64)
        M_3x3[:2, :] = M.astype(np.float64)
        M_3x3[2, :] = [0.0, 0.0, 1.0]
        M_fused = M_3x3 @ H.astype(np.float64)
        patch = cv2.warpPerspective(
            image, M_fused, (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
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

    # P marker (blue) at t=1.0: x = (T_MAX-1)/(T_MAX-T_MIN) * (disp_w-1) ≈ 43%
    p_x = int(round((T_MAX - 1.0) / (T_MAX - T_MIN) * (disp_w - 1)))
    cv2.line(disp_rgb, (p_x, 0), (p_x, disp_h - 1), (0, 0, 255), 2)
    # Q marker (red) at t=0.0: x = (T_MAX-0)/(T_MAX-T_MIN) * (disp_w-1) ≈ 57%
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


# ---------------------------------------------------------------------------
# P'Q' rescale utilities (shared across dataset, infer, annotator)
# ---------------------------------------------------------------------------

def compute_pq_rescale(
    px: float, py: float, qx: float, qy: float,
    t: float,
    rec: dict,
    pq_rescale_mode: str,
    pq_rescale_factor: float,
) -> tuple[float, float, float, float, float, float]:
    """P'Q' 座標系に変換する。pq_rescale_mode=='none' の場合は入力をそのまま返す。

    Returns: (px', py', qx', qy', t', pq_prime_dist)
    """
    orig_pq_dist = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
    if pq_rescale_mode == "none" or orig_pq_dist <= 1e-6:
        return px, py, qx, qy, t, orig_pq_dist
    if pq_rescale_mode == "eye_height":
        xxx = float(rec.get("eye_height", orig_pq_dist)) * pq_rescale_factor
    elif pq_rescale_mode == "mean_pq_dist":
        xxx = float(rec.get("mean_pq_dist", orig_pq_dist)) * pq_rescale_factor
    else:  # eye_scale
        xxx = float(rec.get("eye_scale", orig_pq_dist)) * pq_rescale_factor
    xxx = max(xxx, 1.0)
    ox, oy = (px + qx) / 2.0, (py + qy) / 2.0
    ux, uy = (qx - px) / orig_pq_dist, (qy - py) / orig_pq_dist
    px_ = ox - (xxx / 2.0) * ux
    py_ = oy - (xxx / 2.0) * uy
    qx_ = ox + (xxx / 2.0) * ux
    qy_ = oy + (xxx / 2.0) * uy
    t_ = 0.5 + (t - 0.5) * orig_pq_dist / xxx
    return px_, py_, qx_, qy_, t_, xxx


def get_pq_prime_dist(
    rec: dict,
    orig_pq_dist: float,
    pq_rescale_mode: str,
    pq_rescale_factor: float,
) -> float:
    """逆変換用に P'Q' 長さを返す。対応するフィールドが rec に存在しない場合は orig_pq_dist を返す。"""
    if pq_rescale_mode == "none":
        return orig_pq_dist
    if pq_rescale_mode == "eye_height":
        val = rec.get("eye_height")
    elif pq_rescale_mode == "mean_pq_dist":
        val = rec.get("mean_pq_dist")
    else:  # eye_scale
        val = rec.get("eye_scale")
    if val is None:
        return orig_pq_dist
    return max(float(val) * pq_rescale_factor, 1.0)


def invert_pq_rescale_t(
    t_prime: float,
    pq_dist: float,
    pq_prime_dist: float,
) -> float:
    """P'Q' 座標系の t' を生 PQ 座標系の t に戻す。

    compute_pq_rescale の `t_ = 0.5 + (t - 0.5) * pq_dist / xxx` の逆変換。
    （xxx == pq_prime_dist == |P'Q'| == eye_scale * factor）

    pq_dist が 0 に近い場合は t_prime をそのまま返す（変換不能）。
    """
    if pq_dist <= 1e-6:
        return t_prime
    return 0.5 + (t_prime - 0.5) * pq_prime_dist / pq_dist

"""顔姿勢推定（MediaPipe face_mesh の 3D ランドマークから yaw / pitch / roll を算出）。

ベクトルベースの幾何推定:
    - roll:  目線（左外眼角 → 右外眼角）の xy 平面内角度
    - yaw:   目線の z 成分 / xy 平面成分 の atan2（右目が奥なら正）
    - pitch: 額 → 顎ベクトルの z 成分 / xy 平面成分 の atan2（顎が奥なら正）

MediaPipe face_mesh が返す (N, 3) ランドマーク（x, y は画素、z は同等スケールの相対深度）を想定する。
solvePnP は近平面の 6 点でしばしば 180° flip する曖昧性があるため採用しない。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# MediaPipe face_mesh ランドマーク ID
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_FOREHEAD = 10

LANDMARK_INDICES_FOR_POSE: Tuple[int, ...] = (
    _NOSE_TIP,
    _CHIN,
    _LEFT_EYE_OUTER,
    _RIGHT_EYE_OUTER,
    _FOREHEAD,
)


def compute_face_orientation(
    landmarks_xyz: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float]]:
    """3D ランドマークから顔姿勢 (yaw, pitch, roll) [deg] を返す。

    Args:
        landmarks_xyz: shape (N, 3) の 3D ランドマーク。x, y は画素座標、
            z は MediaPipe 規約の相対深度（x と同スケール）。
        img_w: 画像幅 [px]（将来の拡張用、現在未使用でも受け取る）
        img_h: 画像高さ [px]

    Returns:
        (yaw_deg, pitch_deg, roll_deg) or None（入力不正/退化時）。

    Convention:
        - roll > 0: 右眼が画像下（時計回り傾き、y 軸下向き画像座標系）
        - yaw  > 0: 顔が画像左（撮影者視点で右目が奥）
        - pitch> 0: 顔が下を向く（顎が奥へ引ける）
    """
    if landmarks_xyz.ndim != 2 or landmarks_xyz.shape[1] != 3:
        return None
    if landmarks_xyz.shape[0] <= max(LANDMARK_INDICES_FOR_POSE):
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    nose = landmarks_xyz[_NOSE_TIP]
    chin = landmarks_xyz[_CHIN]
    l_eye = landmarks_xyz[_LEFT_EYE_OUTER]
    r_eye = landmarks_xyz[_RIGHT_EYE_OUTER]
    forehead = landmarks_xyz[_FOREHEAD]

    # roll: 目線ベクトル xy 角（画像内傾き）
    eye_vec = r_eye - l_eye
    eye_xy_mag = math.hypot(float(eye_vec[0]), float(eye_vec[1]))
    if eye_xy_mag < 1e-6:
        return None
    roll_deg = math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0])))

    # yaw: 目線の z 成分（正なら右眼が奥 = 顔が画像左向き）
    # 分母は xy 平面成分。右眼が左眼より奥なら z 差は正。
    yaw_deg = math.degrees(math.atan2(float(eye_vec[2]), eye_xy_mag))

    # pitch: 額 → 顎ベクトルの z 成分。顎が奥なら正（下向き）。
    vert_vec = chin - forehead
    vert_xy_mag = math.hypot(float(vert_vec[0]), float(vert_vec[1]))
    if vert_xy_mag < 1e-6:
        return None
    pitch_deg = math.degrees(math.atan2(float(vert_vec[2]), vert_xy_mag))

    # nose の使用: 将来のキャリブレーションに備えて参照確保（現時点では使わない）
    _ = nose

    return yaw_deg, pitch_deg, roll_deg

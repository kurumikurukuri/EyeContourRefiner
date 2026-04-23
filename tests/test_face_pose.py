"""annotator/face_pose.py および preprocessor の顔順序対応ロジックのテスト。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from annotator.face_pose import (
    LANDMARK_INDICES_FOR_POSE,
    compute_face_orientation,
)
from annotator.preprocessor import _build_existing_centroids, _match_face_order


# -----------------------------------------------------------------------------
# ヘルパー: 合成 3D ランドマーク（MediaPipe 相当の 468 点）
# -----------------------------------------------------------------------------
def _make_frontal_landmarks(img_w: int = 512, img_h: int = 512) -> np.ndarray:
    """正面顔相当の合成 3D ランドマーク (468, 3)。x,y は画素、z=0。"""
    cx, cy = img_w / 2.0, img_h / 2.0
    pts = np.zeros((468, 3), dtype=np.float64)
    # (nose, chin, L-eye outer, R-eye outer, forehead)
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 150.0, 0.0),
        (-80.0, -50.0, 0.0),
        (80.0, -50.0, 0.0),
        (0.0, -120.0, 0.0),
    ]
    for idx, (dx, dy, dz) in zip(LANDMARK_INDICES_FOR_POSE, coords):
        pts[idx] = (cx + dx, cy + dy, dz)
    return pts


# -----------------------------------------------------------------------------
# compute_face_orientation
# -----------------------------------------------------------------------------
class TestComputeFaceOrientation:
    def test_frontal_face_near_zero(self):
        pts = _make_frontal_landmarks(512, 512)
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        yaw, pitch, roll = result
        assert abs(yaw) < 1e-3
        assert abs(pitch) < 1e-3
        assert abs(roll) < 1e-3

    def test_roll_positive_when_right_eye_lower(self):
        """右眼が画像下 → roll 正（画像 y 下向き座標系の時計回り）。"""
        pts = _make_frontal_landmarks(512, 512)
        # 右眼外角を y 方向に下げる（画像座標で y 増加）
        pts[263, 1] += 30.0
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        _yaw, _pitch, roll = result
        assert roll > 5.0
        assert roll < 45.0

    def test_roll_negative_when_left_eye_lower(self):
        pts = _make_frontal_landmarks(512, 512)
        pts[33, 1] += 30.0
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        _yaw, _pitch, roll = result
        assert roll < -5.0

    def test_yaw_positive_when_right_eye_deeper(self):
        """右眼の z が大きい（奥）→ yaw 正（被写体が自分から見て右を向く= 画像左向き）。"""
        pts = _make_frontal_landmarks(512, 512)
        pts[263, 2] = 40.0  # right eye deeper
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        yaw, _pitch, _roll = result
        assert yaw > 10.0

    def test_pitch_positive_when_chin_deeper(self):
        """顎 z が大きい（奥）→ pitch 正（下を向く）。"""
        pts = _make_frontal_landmarks(512, 512)
        pts[152, 2] = 50.0
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        _yaw, pitch, _roll = result
        assert pitch > 10.0

    def test_returns_none_for_insufficient_landmarks(self):
        pts = np.zeros((5, 3), dtype=np.float64)
        assert compute_face_orientation(pts, 512, 512) is None

    def test_returns_none_for_invalid_shape(self):
        pts = np.zeros((468, 2), dtype=np.float64)
        assert compute_face_orientation(pts, 512, 512) is None

    def test_returns_none_for_zero_image_size(self):
        pts = _make_frontal_landmarks()
        assert compute_face_orientation(pts, 0, 512) is None
        assert compute_face_orientation(pts, 512, 0) is None

    def test_returns_none_for_degenerate_eye_vector(self):
        """左右眼が同一点 → None。"""
        pts = _make_frontal_landmarks(512, 512)
        pts[33] = pts[263]
        assert compute_face_orientation(pts, 512, 512) is None

    def test_returns_three_floats(self):
        pts = _make_frontal_landmarks()
        result = compute_face_orientation(pts, 512, 512)
        assert result is not None
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)


# -----------------------------------------------------------------------------
# _match_face_order
# -----------------------------------------------------------------------------
def _make_coords(cx: float, cy: float, n: int = 10) -> np.ndarray:
    """重心が (cx, cy) の合成ランドマーク (n, 2)。"""
    pts = np.zeros((n, 2), dtype=np.float32)
    pts[:, 0] = cx
    pts[:, 1] = cy
    return pts


class TestMatchFaceOrder:
    def test_no_existing_sorts_by_x(self):
        """既存データなし → x 昇順で face_index を割り当てる。"""
        coords_list = [
            _make_coords(300.0, 100.0),
            _make_coords(100.0, 100.0),
            _make_coords(200.0, 100.0),
        ]
        result = _match_face_order(coords_list, {})
        # 検出順 0=x300, 1=x100, 2=x200 → x昇順で rank: 1,0,2
        assert result[1] == 0  # x=100 が face_index 0
        assert result[2] == 1  # x=200 が face_index 1
        assert result[0] == 2  # x=300 が face_index 2

    def test_matches_existing_by_nearest_centroid(self):
        """既存重心に最近傍マッチングで face_index を割り当てる。"""
        existing = {
            0: (100.0, 100.0),
            1: (300.0, 100.0),
        }
        coords_list = [
            _make_coords(310.0, 105.0),  # 既存 face_index 1 に近い
            _make_coords(95.0, 98.0),    # 既存 face_index 0 に近い
        ]
        result = _match_face_order(coords_list, existing)
        assert result[0] == 1  # 検出 0 → face_index 1
        assert result[1] == 0  # 検出 1 → face_index 0

    def test_single_face_with_existing(self):
        existing = {0: (200.0, 200.0)}
        coords_list = [_make_coords(198.0, 202.0)]
        result = _match_face_order(coords_list, existing)
        assert result == [0]

    def test_new_face_gets_next_index(self):
        """既存より多く顔が検出された場合、余剰顔は max+1 以降のインデックスを得る。"""
        existing = {0: (100.0, 100.0)}
        coords_list = [
            _make_coords(100.0, 100.0),
            _make_coords(400.0, 100.0),  # 対応なし
        ]
        result = _match_face_order(coords_list, existing)
        assert result[0] == 0
        assert result[1] == 1  # max(existing)=0 + 1

    def test_empty_coords_list(self):
        result = _match_face_order([], {})
        assert result == []


# -----------------------------------------------------------------------------
# _build_existing_centroids
# -----------------------------------------------------------------------------
class TestBuildExistingCentroids:
    def _make_sample(self, image_path: str, face_index: int, px: float, py: float) -> dict:
        return {
            "image_path": image_path,
            "face_index": face_index,
            "part_name": "left",
            "sub_key": "upper",
            "point_index": 0,
            "px": px, "py": py, "qx": px + 10, "qy": py + 5,
            "t": None, "skip_flag": False,
        }

    def test_returns_empty_for_missing_image(self):
        samples = [self._make_sample("data/images/a.jpg", 0, 100.0, 100.0)]
        result = _build_existing_centroids("data/images/b.jpg", samples)
        assert result == {}

    def test_single_face_centroid(self):
        samples = [self._make_sample("data/images/a.jpg", 0, 100.0, 200.0)]
        result = _build_existing_centroids("data/images/a.jpg", samples)
        assert 0 in result
        cx, cy = result[0]
        # midpoint of (100,200) and (110,205) = (105, 202.5)
        assert abs(cx - 105.0) < 1e-6
        assert abs(cy - 202.5) < 1e-6

    def test_two_faces(self):
        samples = [
            self._make_sample("data/images/a.jpg", 0, 100.0, 100.0),
            self._make_sample("data/images/a.jpg", 1, 300.0, 100.0),
        ]
        result = _build_existing_centroids("data/images/a.jpg", samples)
        assert set(result.keys()) == {0, 1}
        assert result[0][0] < result[1][0]

    def test_ignores_other_image(self):
        samples = [
            self._make_sample("data/images/a.jpg", 0, 100.0, 100.0),
            self._make_sample("data/images/b.jpg", 0, 500.0, 500.0),
        ]
        result = _build_existing_centroids("data/images/a.jpg", samples)
        assert set(result.keys()) == {0}
        assert result[0][0] < 200

"""サンプル生成（P/Q 抽出）の回帰テスト。

annotator/preprocessor.py の `_extract_samples_from_image` と、同じ対応表を使う
tools/imagesdb_maintenance/preprocess_new_images.py の `process_image` の双方が、
1顔あたり 28 サンプル（両眼 × upper/lower × 7点）を作り、各サンプルの P/Q が
face_makeup と同じ edge ID 定義どおりの MediaPipe ランドマーク座標になることを保証する。

修正前は対応表が整数 ID のリストで `isinstance(entry, tuple)` が常に偽となり、
どちらも 0 件を返していた。MediaPipe 本体には依存せず、FaceMesh は偽物に差し替える。
"""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import annotator.preprocessor as pp
from eyeliner.data.eyeliner_landmarks import EDGE_LANDMARK_DIRECTIONS, EYELINER_SAMPLE_EDGE_IDS

IMG_W, IMG_H = 200, 100
NUM_MP_LANDMARKS = 478
PARTS = ("eyeliner_left", "eyeliner_right")
SUB_KEYS = ("upper", "lower")


def _landmarks(x_shift: float = 0.0) -> list[SimpleNamespace]:
    """全インデックスが一意な (x, y) を持つ 478 点（x はインデックスに対し単調増加）。"""
    return [
        SimpleNamespace(
            x=0.05 + 0.4 * i / NUM_MP_LANDMARKS + x_shift,
            y=0.05 + 0.9 * ((i * 37) % NUM_MP_LANDMARKS) / NUM_MP_LANDMARKS,
            z=0.0,
        )
        for i in range(NUM_MP_LANDMARKS)
    ]


def _pixel_coords(landmarks: list[SimpleNamespace]) -> np.ndarray:
    return np.array([[lm.x * IMG_W, lm.y * IMG_H] for lm in landmarks], dtype=np.float32)


class _FakeFaceMesh:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self._faces = faces

    def process(self, image_rgb):
        return SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=f) for f in self._faces]
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_mediapipe(monkeypatch, faces: list[list[SimpleNamespace]]) -> None:
    fake_mp = SimpleNamespace(
        solutions=SimpleNamespace(
            face_mesh=SimpleNamespace(FaceMesh=lambda **kwargs: _FakeFaceMesh(faces))
        )
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setattr(pp, "compute_face_orientation", lambda *args, **kwargs: None)


def _write_image(tmp_path):
    path = tmp_path / "face.jpg"
    cv2.imwrite(str(path), np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))
    return path


def _assert_pq_matches_landmarks(samples: list[dict], coords: np.ndarray) -> None:
    assert len(samples) == 28
    for sample in samples:
        edge_id = EYELINER_SAMPLE_EDGE_IDS[sample["part_name"]][sample["sub_key"]][
            sample["point_index"]
        ]
        p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[edge_id]
        assert (sample["px"], sample["py"]) == pytest.approx(tuple(coords[p_idx]))
        assert (sample["qx"], sample["qy"]) == pytest.approx(tuple(coords[q_idx]))


class TestExtractSamplesFromImage:

    def test_single_face_yields_seven_points_per_part_and_sub_key(self, tmp_path, monkeypatch):
        _install_fake_mediapipe(monkeypatch, [_landmarks()])

        samples = pp._extract_samples_from_image(_write_image(tmp_path), {})

        assert samples is not None and len(samples) == 28
        for part in PARTS:
            for sub_key in SUB_KEYS:
                indices = sorted(
                    s["point_index"] for s in samples
                    if s["part_name"] == part and s["sub_key"] == sub_key
                )
                assert indices == list(range(7))

    def test_p_and_q_come_from_the_edge_landmark_definitions(self, tmp_path, monkeypatch):
        landmarks = _landmarks()
        _install_fake_mediapipe(monkeypatch, [landmarks])

        samples = pp._extract_samples_from_image(_write_image(tmp_path), {})

        _assert_pq_matches_landmarks(samples, _pixel_coords(landmarks))

    def test_two_faces_are_indexed_left_to_right(self, tmp_path, monkeypatch):
        right_face, left_face = _landmarks(x_shift=0.5), _landmarks()
        _install_fake_mediapipe(monkeypatch, [right_face, left_face])  # 検出順は右→左

        samples = pp._extract_samples_from_image(_write_image(tmp_path), {})

        assert len(samples) == 56
        by_face = {
            face_index: [s for s in samples if s["face_index"] == face_index]
            for face_index in (0, 1)
        }
        _assert_pq_matches_landmarks(by_face[0], _pixel_coords(left_face))
        _assert_pq_matches_landmarks(by_face[1], _pixel_coords(right_face))


class TestPreprocessNewImages:
    """増分マージ用ツールも同じ対応表を使う。"""

    @pytest.fixture
    def tool(self):
        return importlib.import_module("tools.imagesdb_maintenance.preprocess_new_images")

    def test_process_image_yields_the_same_28_samples(self, tool, tmp_path):
        landmarks = _landmarks()

        result = tool.process_image(_FakeFaceMesh([landmarks]), str(_write_image(tmp_path)))

        assert result is not None and len(result["faces"]) == 1
        _assert_pq_matches_landmarks(result["faces"][0]["samples"], _pixel_coords(landmarks))

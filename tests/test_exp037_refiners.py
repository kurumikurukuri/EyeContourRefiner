"""EXP037 ルールベース補正（face_makeup）と ML 補正（ONNX）のラッパーの単体テスト。

検証:
  1. ルールベース側が face_makeup の EyelinerCorrector.correct_points と同じ点を返すこと
  2. ルールベースの出力点をモデルの t 規約（R = t·P + (1-t)·Q）に変換できること
  3. eye_scale が学習時（compute_eye_meta）と同じ値になること
  4. ML 側のバッチ推論が 1 点ずつの推論と一致すること（models/ に ONNX がある場合のみ）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
_EXP037_SCR = ROOT / "experiments" / "EXP037_rule_vs_ml" / "scr"
if str(_EXP037_SCR) not in sys.path:
    sys.path.insert(0, str(_EXP037_SCR))

face_makeup_corrector = pytest.importorskip("face_makeup.core.eyeliner_corrector")

from refiners_exp037 import (
    eye_scale_of,
    refine_ml,
    refine_rule,
    rule_point_to_t,
)

from eyeliner.data.eyeliner_landmarks import (
    EDGE_LANDMARK_DIRECTIONS,
    EYELINER_SAMPLE_EDGE_IDS,
)
from eyeliner.data.weights import compute_eye_meta

MODEL_PATH = ROOT / "models" / "eye_contour_refiner.onnx"


def _synthetic_eye_image(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """暗い横帯（アイライン相当）+ ノイズの RGB 画像。エッジ検出が働く入力にする。"""
    rng = np.random.default_rng(seed)
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    image[100:130, :, :] = 40
    noise = rng.integers(-15, 16, size=image.shape)
    return np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def _random_coords(seed: int = 1) -> np.ndarray:
    """face_makeup の拡張ランドマーク配列（1708 点）を、目の付近にばらまいて作る。"""
    rng = np.random.default_rng(seed)
    coords = np.empty((1708, 2), dtype=np.float32)
    coords[:, 0] = rng.uniform(80, 240, size=1708)
    coords[:, 1] = rng.uniform(60, 170, size=1708)
    return coords


def _eye_pq(coords: np.ndarray, part_name: str) -> tuple[list[int], np.ndarray]:
    edge_ids = [*EYELINER_SAMPLE_EDGE_IDS[part_name]["upper"], *EYELINER_SAMPLE_EDGE_IDS[part_name]["lower"]]
    pq = np.array(
        [[*coords[EDGE_LANDMARK_DIRECTIONS[e][0]], *coords[EDGE_LANDMARK_DIRECTIONS[e][1]]] for e in edge_ids],
        dtype=np.float64,
    )
    return edge_ids, pq


@pytest.mark.parametrize("part_name", ["eyeliner_left", "eyeliner_right"])
def test_refine_rule_matches_face_makeup_correct_points(part_name):
    from face_makeup.constants import EYELINER_CORRECTION_MAP

    image = _synthetic_eye_image()
    coords = _random_coords()
    expected_all = face_makeup_corrector.EyelinerCorrector().correct_points(coords, part_name, image)
    order = [idx for indices in EYELINER_CORRECTION_MAP[part_name].values() for idx in indices]
    expected_by_edge = {idx: tuple(expected_all[i]) for i, idx in enumerate(order) if idx in EDGE_LANDMARK_DIRECTIONS}

    edge_ids, pq = _eye_pq(coords, part_name)
    result = refine_rule(image, pq)

    assert result.points.shape == (14, 2)
    for i, edge_id in enumerate(edge_ids):
        assert tuple(result.points[i]) == pytest.approx(expected_by_edge[edge_id])


def test_refine_rule_flags_fallback_when_no_edge():
    """一様画像ではエッジが見つからず、P がそのまま返る（fallback）。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    pq = np.array([[50.0, 50.0, 50.0, 20.0]])
    result = refine_rule(image, pq)
    assert result.fallback.tolist() == [True]
    assert tuple(result.points[0]) == (50.0, 50.0)
    assert result.t_untruncated.tolist() == [1.0]


def test_refine_rule_untruncated_t_is_within_one_pixel_of_output_point():
    """t_untruncated は整数切り捨て前の検出位置。出力点（切り捨て後）との差は 1px 未満になる。"""
    image = _synthetic_eye_image()
    pq = np.array([[160.3, 140.7, 160.3, 80.7], [100.5, 150.2, 104.5, 70.2]])
    result = refine_rule(image, pq)
    assert not result.fallback.any()
    t = result.t_untruncated[:, None]
    untruncated = t * pq[:, :2] + (1 - t) * pq[:, 2:]
    assert np.all(np.abs(untruncated - result.points) < 1.0)
    assert np.all((0.5 < result.t_untruncated) & (result.t_untruncated <= 1.0))


def _count_bilateral_filter(monkeypatch) -> list[int]:
    """face_makeup の前処理（bilateralFilter）の呼び出し回数を数える。"""
    import cv2

    calls = [0]
    original = cv2.bilateralFilter

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "bilateralFilter", counting)
    return calls


def test_refine_rule_does_not_reuse_preprocess_cache_between_calls(monkeypatch):
    """計測の公平性: 呼び出しごとに新しい corrector を使い、同じ画像でも毎回 bilateralFilter が走る。"""
    calls = _count_bilateral_filter(monkeypatch)
    image = _synthetic_eye_image()
    pq = np.array([[160.0, 140.0, 160.0, 80.0]])
    first = refine_rule(image, pq)
    second = refine_rule(image, pq)
    assert calls[0] == 2
    np.testing.assert_array_equal(first.points, second.points)


def test_run_rule_per_eye_preprocesses_each_eye_like_production(monkeypatch):
    """本番（左右の EyelinerPass が別々の corrector）と同じく 1 顔で前処理が 2 回走る。共有版は 1 回。"""
    from timing_exp037 import run_rule, run_rule_per_eye

    image = _synthetic_eye_image()
    coords = _random_coords()
    pq = np.concatenate([_eye_pq(coords, "eyeliner_left")[1], _eye_pq(coords, "eyeliner_right")[1]])

    calls = _count_bilateral_filter(monkeypatch)
    per_eye = run_rule_per_eye(image, pq)
    assert calls[0] == 2
    shared = run_rule(image, pq)
    assert calls[0] == 3
    assert per_eye["total_s"] == pytest.approx(per_eye["part1_s"] + per_eye["part2_s"])
    assert shared["total_s"] == pytest.approx(shared["part1_s"] + shared["part2_s"])


@pytest.mark.parametrize(
    ("point", "expected_t"),
    [
        ((10.0, 0.0), 1.0),   # R = P
        ((0.0, 0.0), 0.0),    # R = Q
        ((7.0, 0.0), 0.7),    # P から Q へ 0.3 進んだ点
        ((7.0, 3.0), 0.7),    # 直交方向のずれは t に影響しない（射影）
    ],
)
def test_rule_point_to_t_uses_model_convention(point, expected_t):
    pq = np.array([[10.0, 0.0, 0.0, 0.0]])
    t = rule_point_to_t(pq, np.array([point]))
    assert t[0] == pytest.approx(expected_t)


def test_eye_scale_matches_training_eye_meta():
    rng = np.random.default_rng(3)
    pq = rng.uniform(0, 300, size=(14, 4))
    records = [
        {"image_path": "a.jpg", "face_index": 0, "part_name": "eyeliner_left", "px": p[0], "py": p[1], "qx": p[2], "qy": p[3]}
        for p in pq
    ]
    compute_eye_meta(records)
    assert eye_scale_of(pq) == pytest.approx(records[0]["eye_scale"])


def test_eye_scale_has_floor_of_one():
    pq = np.zeros((14, 4))
    assert eye_scale_of(pq) == 1.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="models/eye_contour_refiner.onnx が無い")
def test_refine_ml_batch_matches_single_point_inference():
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    image = _synthetic_eye_image()
    _, pq = _eye_pq(_random_coords(), "eyeliner_left")
    is_upper = np.array([True] * 7 + [False] * 7)
    eye_scale = eye_scale_of(pq)

    batch = refine_ml(session, image, pq, is_upper, np.full(14, eye_scale))
    singles = [
        refine_ml(session, image, pq[i : i + 1], is_upper[i : i + 1], np.array([eye_scale])).t[0] for i in range(14)
    ]

    assert batch.t.shape == (14,)
    np.testing.assert_allclose(batch.t, singles, atol=1e-5)
    expected_r = batch.t[:, None] * pq[:, :2] + (1 - batch.t[:, None]) * pq[:, 2:]
    np.testing.assert_allclose(batch.points, expected_r, atol=1e-6)
    assert batch.patch_s > 0 and batch.infer_s > 0


def test_select_full_faces_keeps_only_28_point_faces_split_by_eye():
    import pandas as pd
    from timing_exp037 import select_full_faces

    rows = []
    for image, parts in (("both.jpg", ("eyeliner_right", "eyeliner_left")), ("one.jpg", ("eyeliner_left",))):
        for part in parts:
            for sub_key in ("lower", "upper"):
                for point_index in range(7):
                    rows.append({"image_path": image, "face_index": 0, "part_name": part,
                                 "sub_key": sub_key, "point_index": point_index})
    faces = select_full_faces(pd.DataFrame(rows))

    assert len(faces) == 1
    face = faces[0]
    assert set(face["image_path"]) == {"both.jpg"}
    assert set(face["part_name"].iloc[:14]) == {"eyeliner_left"}
    assert set(face["part_name"].iloc[14:]) == {"eyeliner_right"}
    assert face["sub_key"].iloc[:7].eq("upper").all() and face["point_index"].iloc[:7].tolist() == list(range(7))


def test_to_production_coords_matches_face_makeup_to_image_coords():
    """ルールベースの入力は face_makeup 本番（FaceLandmarks.to_image_coords）と同じ丸めにする。"""
    from face_makeup.core.types import FaceLandmarks
    from refiners_exp037 import to_production_coords

    h, w = 480, 640
    rng = np.random.default_rng(5)
    normalized = rng.uniform(0, 1, size=(14, 2)).astype(np.float32)
    expected = FaceLandmarks(points=normalized).to_image_coords(w, h)

    pixel = normalized * np.array([w, h], dtype=np.float32)
    pq = np.concatenate([pixel[:7], pixel[7:]], axis=1)
    result = to_production_coords(pq)
    np.testing.assert_array_equal(result[:, :2], expected[:7])
    np.testing.assert_array_equal(result[:, 2:], expected[7:])


def test_check_eye_scale_against_db_does_not_create_missing_db(tmp_path):
    """sqlite3.connect が存在しないパスに空 DB を作らないよう、開く前に中断する。"""
    import pandas as pd
    from infer_exp037 import check_eye_scale_against_db

    db_path = tmp_path / "samples.db"
    with pytest.raises(FileNotFoundError):
        check_eye_scale_against_db(pd.DataFrame({"idx": [0], "eye_scale": [1.0]}), db_path)
    assert not db_path.exists()


def test_face_makeup_commit_matches_pyproject_pin():
    """計測結果に記録する commit は、導入済みの face_makeup から読み、pyproject の固定と一致する。"""
    import tomllib

    from timing_exp037 import face_makeup_commit

    (pin,) = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["exp037"]
    installed = face_makeup_commit()
    if installed is None:
        pytest.skip("face_makeup が git 以外から導入されている")
    assert pin.endswith(f"@{installed}")

"""eye_contour_landmarks の P/Q 対応表の回帰テスト。

`EDGE_LANDMARK_DIRECTIONS` は face_makeup の `custom_landmark.py`（commit 09034270）から
移植した「edge ID → MediaPipe (P, Q) インデックス」。`EYE_CONTOUR_SAMPLE_EDGE_IDS` は
samples.db の (part_name, sub_key, point_index) に対応する edge ID の並びで、mediapipe 0.10.21 を
400 画像に実行して全 28 点が DB の px/py/qx/qy と誤差 0.0000 px で一致することを確認した値。

face_makeup の EYELINER_*_INDICES（描画用ポリライン）は上まぶたに 1408/1608 と下まぶた末尾の
重複エッジを含むが、アノテーション対象は upper/lower とも 7 本。
"""
from __future__ import annotations

from eye_contour.data.eye_contour_landmarks import (
    EDGE_LANDMARK_DIRECTIONS,
    EYE_CONTOUR_SAMPLE_EDGE_IDS,
    EYE_CONTOUR_SAMPLE_PQ,
)

NUM_MP_LANDMARKS = 478
PARTS = ("eyeliner_left", "eyeliner_right")
SUB_KEYS = ("upper", "lower")


class TestEdgeLandmarkDirections:

    def test_defines_all_thirty_edges(self):
        """上まぶた 8 本 + 下まぶた 7 本を左右で 2 組（計 30 本）。"""
        expected = (
            set(range(1401, 1409)) | set(range(1501, 1508))
            | set(range(1601, 1609)) | set(range(1701, 1708))
        )
        assert set(EDGE_LANDMARK_DIRECTIONS) == expected
        assert len(EDGE_LANDMARK_DIRECTIONS) == 30

    def test_pq_indices_are_distinct_mediapipe_landmarks(self):
        for edge_id, (p_idx, q_idx) in EDGE_LANDMARK_DIRECTIONS.items():
            assert 0 <= p_idx < NUM_MP_LANDMARKS, edge_id
            assert 0 <= q_idx < NUM_MP_LANDMARKS, edge_id
            assert p_idx != q_idx, edge_id

    def test_spot_check_against_face_makeup(self):
        assert EDGE_LANDMARK_DIRECTIONS[1401] == (173, 190)
        assert EDGE_LANDMARK_DIRECTIONS[1507] == (155, 112)
        assert EDGE_LANDMARK_DIRECTIONS[1608] == (263, 342)
        assert EDGE_LANDMARK_DIRECTIONS[1707] == (382, 341)


class TestSampleEdgeIds:

    def test_matches_verified_point_index_order(self):
        """point_index 0〜6 の並び（DB 実測で確定）。1408/1608 は含まない。"""
        assert EYE_CONTOUR_SAMPLE_EDGE_IDS == {
            "eyeliner_left": {
                "upper": tuple(range(1401, 1408)),
                "lower": tuple(range(1501, 1508)),
            },
            "eyeliner_right": {
                "upper": tuple(range(1601, 1608)),
                "lower": tuple(range(1701, 1708)),
            },
        }

    def test_seven_points_per_sub_key(self):
        for part in PARTS:
            for sub_key in SUB_KEYS:
                assert len(EYE_CONTOUR_SAMPLE_EDGE_IDS[part][sub_key]) == 7

    def test_no_edge_is_shared_between_samples(self):
        """upper/lower で同じエッジを二重にサンプル化しない（1眼14点・両眼28点）。"""
        all_ids = [
            edge_id
            for part in PARTS for sub_key in SUB_KEYS
            for edge_id in EYE_CONTOUR_SAMPLE_EDGE_IDS[part][sub_key]
        ]
        assert len(all_ids) == 28
        assert len(set(all_ids)) == 28

    def test_every_sample_edge_has_a_pq_definition(self):
        for part in PARTS:
            for sub_key in SUB_KEYS:
                for edge_id in EYE_CONTOUR_SAMPLE_EDGE_IDS[part][sub_key]:
                    assert edge_id in EDGE_LANDMARK_DIRECTIONS


class TestSamplePq:

    def test_entries_are_id_p_q_tuples_in_point_index_order(self):
        """preprocessor が `_, p_idx, q_idx = entry` で展開する (id, P, Q) 形式。"""
        for part in PARTS:
            for sub_key in SUB_KEYS:
                entries = EYE_CONTOUR_SAMPLE_PQ[part][sub_key]
                assert len(entries) == 7
                for point_index, entry in enumerate(entries):
                    assert isinstance(entry, tuple)
                    edge_id, p_idx, q_idx = entry
                    assert edge_id == EYE_CONTOUR_SAMPLE_EDGE_IDS[part][sub_key][point_index]
                    assert (p_idx, q_idx) == EDGE_LANDMARK_DIRECTIONS[edge_id]

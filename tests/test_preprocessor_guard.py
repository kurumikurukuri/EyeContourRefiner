"""annotator/preprocessor.py の DB 保護ガードの回帰テスト。

generate_samples() は既存 DB を replace_all_samples で全置換する。mediapipe の API
互換性が崩れる等で全サンプルが 0 件になった場合、既存アノテーション（21,504 行）を
失わないよう、サンプルが空 or 既存から急減した場合は replace せず中断する。
"""
from __future__ import annotations

import cv2
import numpy as np

import annotator.preprocessor as pp


def _make_existing_sample(i: int) -> dict:
    return {
        "image_path": f"data/images/img_{i}.jpg",
        "face_index": 0,
        "part_name": "eyeliner_right",
        "sub_key": "upper",
        "point_index": 0,
        "px": 0.0, "py": 0.0, "qx": 1.0, "qy": 1.0,
        "t": 0.5,
        "skip_flag": False,
    }


def _new_sample() -> dict:
    return {
        "image_path": "data/images/new.jpg",
        "face_index": 0,
        "part_name": "eyeliner_right",
        "sub_key": "upper",
        "point_index": 0,
        "px": 0.0, "py": 0.0, "qx": 1.0, "qy": 1.0,
        "t": None,
        "skip_flag": False,
    }


def _write_dummy_image(images_dir) -> None:
    cv2.imwrite(str(images_dir / "a.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))


class TestGenerateSamplesDbGuard:

    def test_aborts_when_no_samples_extracted(self, tmp_path, monkeypatch):
        """全画像からサンプルが 0 件のとき、replace_all_samples を呼ばない。"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_dummy_image(images_dir)
        db_path = tmp_path / "samples.db"

        monkeypatch.setattr(pp._db, "init_db", lambda path: None)
        monkeypatch.setattr(pp._db, "load_all_samples", lambda path: [])
        monkeypatch.setattr(pp, "_extract_samples_from_image", lambda *a, **k: None)

        calls = []
        monkeypatch.setattr(
            pp._db, "replace_all_samples", lambda samples, path: calls.append(samples)
        )

        pp.generate_samples(images_dir, db_path, workers=1)

        assert calls == []

    def test_aborts_when_samples_shrink_drastically(self, tmp_path, monkeypatch):
        """既存 100 件に対し生成が 1 件に急減したとき、replace を呼ばない。"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_dummy_image(images_dir)
        db_path = tmp_path / "samples.db"

        existing = [_make_existing_sample(i) for i in range(100)]
        monkeypatch.setattr(pp._db, "init_db", lambda path: None)
        monkeypatch.setattr(pp._db, "load_all_samples", lambda path: existing)
        monkeypatch.setattr(pp, "_extract_samples_from_image", lambda *a, **k: [_new_sample()])

        calls = []
        monkeypatch.setattr(
            pp._db, "replace_all_samples", lambda samples, path: calls.append(samples)
        )

        pp.generate_samples(images_dir, db_path, workers=1)

        assert calls == []

    def test_replaces_when_samples_are_healthy(self, tmp_path, monkeypatch):
        """正常に生成できた場合は replace を呼ぶ（ガードの誤爆がないこと）。"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_dummy_image(images_dir)
        db_path = tmp_path / "samples.db"

        monkeypatch.setattr(pp._db, "init_db", lambda path: None)
        monkeypatch.setattr(pp._db, "load_all_samples", lambda path: [])
        monkeypatch.setattr(pp, "_extract_samples_from_image", lambda *a, **k: [_new_sample()])

        calls = []
        monkeypatch.setattr(
            pp._db, "replace_all_samples", lambda samples, path: calls.append(samples)
        )

        pp.generate_samples(images_dir, db_path, workers=1)

        assert len(calls) == 1

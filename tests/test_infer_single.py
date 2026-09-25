"""scripts/infer.py 単一画像モード（run_single）の回帰テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import infer  # noqa: E402


def test_run_single_passes_point_index_to_predict(tmp_path, monkeypatch):
    """run_single が main() の args に依存せず point_index を predict に渡す。"""
    image_path = tmp_path / "face.jpg"
    cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))

    captured = {}

    def _fake_predict(*args, **kwargs):
        captured.update(kwargs)
        return 0.5

    monkeypatch.setattr(infer, "predict", _fake_predict)

    infer.run_single(
        None, None, image_path, 1.0, 1.0, 10.0, 10.0, is_upper=1.0, point_index=3,
    )

    assert captured["point_index"] == 3

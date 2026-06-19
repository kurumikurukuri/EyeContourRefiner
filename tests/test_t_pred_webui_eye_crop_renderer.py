"""tools/annotation_support/t_pred_webui/eye_crop_renderer.py の render_eye_crop_b64 テスト。

実モデルはロードせず、predict_t_raw を記録する Fake 推論器で検証する。
"""
from __future__ import annotations

import base64

import numpy as np

from tools.annotation_support.t_pred_webui import eye_crop_renderer


class _RecordingPredictor:
    """predict_t_raw の引数を記録し、固定の生 t を返す Fake 推論器。"""

    def __init__(self) -> None:
        self.calls = []

    def predict_t_raw(self, image_rgb, px, py, qx, qy, is_upper, eye_scale=None):
        self.calls.append(
            {"px": px, "py": py, "qx": qx, "qy": qy,
             "is_upper": is_upper, "eye_scale": eye_scale}
        )
        return 0.5


def _eye_samples(eye_scale=72.0):
    """upper/lower 各1点の最小サンプル（座標は 200x200 画像内に収まる）。"""
    return [
        {"sub_key": "upper", "point_index": 0,
         "px": 50.0, "py": 50.0, "qx": 60.0, "qy": 50.0, "eye_scale": eye_scale},
        {"sub_key": "lower", "point_index": 0,
         "px": 50.0, "py": 60.0, "qx": 60.0, "qy": 60.0, "eye_scale": eye_scale},
    ]


def test_eye_scale_is_forwarded_to_predict_t_raw():
    """回帰テスト: eye_samples の eye_scale が predict_t_raw に伝播する。

    デュアルブランチ枝B の P'Q' 計算に eye_scale が必須なため、
    プレビューがスキャン (t_pred_scan) と同じ入力を使うことを保証する。
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    predictor = _RecordingPredictor()

    b64 = eye_crop_renderer.render_eye_crop_b64(img, _eye_samples(eye_scale=72.0), predictor)

    assert b64 is not None
    assert predictor.calls, "predict_t_raw が呼ばれていない"
    assert all(c["eye_scale"] == 72.0 for c in predictor.calls)


def test_missing_eye_scale_forwards_none():
    """eye_scale が欠損している場合は None が渡る（フォールバック動作）。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    samples = _eye_samples()
    for s in samples:
        s["eye_scale"] = None
    predictor = _RecordingPredictor()

    eye_crop_renderer.render_eye_crop_b64(img, samples, predictor)

    assert predictor.calls
    assert all(c["eye_scale"] is None for c in predictor.calls)


def test_returns_valid_base64_png():
    """戻り値が base64 デコード可能で PNG シグネチャを持つ。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    b64 = eye_crop_renderer.render_eye_crop_b64(img, _eye_samples(), _RecordingPredictor())

    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_r_points_returns_none():
    """全点の予測が None の場合は None を返す。"""
    class _NonePredictor:
        def predict_t_raw(self, *args, **kwargs):
            return None

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    assert eye_crop_renderer.render_eye_crop_b64(img, _eye_samples(), _NonePredictor()) is None

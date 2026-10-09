"""配布ONNXのスモークテスト。models/eyeliner_detection_dual.onnx のみに
依存する（.pt チェックポイント・実験データ不要）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODEL_PATH = Path(__file__).parent.parent / "models" / "eyeliner_detection_dual.onnx"

EXPECTED_INPUTS = ["patch_a", "patch_b", "is_upper", "px", "py", "qx", "qy", "eye_scale"]
EXPECTED_OUTPUTS = ["t"]


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        pytest.skip(f"モデルが見つかりません: {MODEL_PATH}")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def _dummy_feed(batch: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(0)
    return {
        "patch_a": rng.rand(batch, 3, 64, 64).astype(np.float32),
        "patch_b": rng.rand(batch, 3, 64, 64).astype(np.float32),
        "is_upper": np.ones((batch, 1), dtype=np.float32),
        "px": np.zeros((batch, 1), dtype=np.float32),
        "py": np.zeros((batch, 1), dtype=np.float32),
        "qx": np.full((batch, 1), 10.0, dtype=np.float32),  # px==qx==qy==0 だと pq_dist=0 の退化ケースになるため非退化値
        "qy": np.zeros((batch, 1), dtype=np.float32),
        "eye_scale": np.full((batch, 1), 20.0, dtype=np.float32),
    }


class TestModelContract:
    def test_input_names(self, session: ort.InferenceSession) -> None:
        names = [i.name for i in session.get_inputs()]
        assert names == EXPECTED_INPUTS

    def test_output_names(self, session: ort.InferenceSession) -> None:
        names = [o.name for o in session.get_outputs()]
        assert names == EXPECTED_OUTPUTS


class TestInference:
    def test_batch1_shape_and_finite(self, session: ort.InferenceSession) -> None:
        out = session.run(None, _dummy_feed(1))[0]
        assert out.shape == (1, 1)
        assert np.isfinite(out).all()

    def test_batch8_shape(self, session: ort.InferenceSession) -> None:
        out = session.run(None, _dummy_feed(8))[0]
        assert out.shape == (8, 1)

    def test_deterministic(self, session: ort.InferenceSession) -> None:
        feed = _dummy_feed(1)
        out1 = session.run(None, feed)[0]
        out2 = session.run(None, feed)[0]
        np.testing.assert_array_equal(out1, out2)

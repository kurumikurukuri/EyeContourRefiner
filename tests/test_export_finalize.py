"""scripts/export_dual_onnx.finalize_onnx のテスト。

小さな torch モデルを実際に torch.onnx.export し、エクスポータが埋め込む
スタックトレース等のデバッグ情報が finalize 後に残らないこと、
外部データが単一 .onnx にインライン化されること、推論結果が変わらないことを確認する。
学習済み checkpoint は不要。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from export_dual_onnx import finalize_onnx

from eye_contour.models.onnx_sanitize import find_debug_metadata

LEAKED_TRACE = r'File "F:\somewhere\project\.venv\Lib\site-packages\torch\nn\modules\linear.py", line 134'


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _inject_stack_trace(path: Path) -> None:
    """配布ONNXを書き出した torch 2.11 の dynamo エクスポータ相当のスタックトレースを付与する。

    torch のバージョンによってはこの小さなモデルにデバッグ情報が付かないため、
    finalize_onnx が除去することを確実に検証できるよう明示的に埋め込む。
    """
    model = onnx.load(str(path), load_external_data=False)
    for node in model.graph.node:
        entry = node.metadata_props.add()
        entry.key = "pkg.torch.onnx.stack_trace"
        entry.value = LEAKED_TRACE
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, np.ndarray, np.ndarray]:
    pytest.importorskip("onnxscript", reason="torch.onnx の dynamo エクスポータに onnxscript が必要")
    torch.manual_seed(0)
    model = _Tiny().eval()
    out_path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    torch.onnx.export(
        model, (torch.zeros(1, 4),), str(out_path),
        input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    _inject_stack_trace(out_path)
    x = np.random.RandomState(0).rand(3, 4).astype(np.float32)
    before = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    finalize_onnx(out_path)
    return out_path, x, before


class TestFinalizeOnnx:
    def test_no_debug_metadata(self, exported) -> None:
        out_path, _, _ = exported
        assert find_debug_metadata(onnx.load(str(out_path))) == []

    def test_no_source_paths_in_bytes(self, exported) -> None:
        out_path, _, _ = exported
        data = out_path.read_bytes()
        assert b"site-packages" not in data
        assert Path(__file__).name.encode() not in data

    def test_single_file(self, exported) -> None:
        out_path, _, _ = exported
        assert not out_path.with_name(out_path.name + ".data").exists()

    def test_inference_unchanged(self, exported) -> None:
        out_path, x, before = exported
        after = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        np.testing.assert_array_equal(after, before)

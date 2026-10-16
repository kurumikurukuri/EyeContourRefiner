"""eye_contour.models.onnx_sanitize のテスト。

torch.onnx（dynamo エクスポータ）はノード・グラフ・入出力に metadata_props として
スタックトレース（エクスポート環境の絶対パスを含む）等のデバッグ情報を埋め込む。
配布前にそれらを除去できること、推論に必要な情報は保たれることを確認する。
"""
from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from eye_contour.models.onnx_sanitize import find_debug_metadata, strip_debug_metadata

LEAKED_PATH = r'File "F:\somewhere\project\.venv\Lib\site-packages\torch\nn\modules\linear.py"'


def _add_meta(proto, key: str, value: str) -> None:
    entry = proto.metadata_props.add()
    entry.key = key
    entry.value = value


def _make_model_with_debug_info() -> onnx.ModelProto:
    """y = x + w の最小モデルに、dynamo エクスポータ相当のデバッグ情報を付ける。"""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [None, 2])
    w = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="w")
    add = helper.make_node("Add", ["x", "w"], ["mid"], doc_string=LEAKED_PATH)
    ident = helper.make_node("Identity", ["mid"], ["y"])
    graph = helper.make_graph(
        [add, ident], "g", [x], [y], initializer=[w], value_info=[mid], doc_string=LEAKED_PATH,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    model.doc_string = LEAKED_PATH

    for node in model.graph.node:
        _add_meta(node, "pkg.torch.onnx.stack_trace", LEAKED_PATH)
    _add_meta(model.graph, "pkg.torch.export.ExportedProgram.graph_signature", "sig")
    _add_meta(model.graph.input[0], "pkg.torch.onnx.original_node_name", "x")
    _add_meta(model.graph.output[0], "pkg.torch.onnx.original_node_name", "y")
    _add_meta(model.graph.value_info[0], "pkg.torch.onnx.original_node_name", "mid")
    _add_meta(model.graph.initializer[0], "pkg.torch.onnx.original_node_name", "w")
    _add_meta(model, "exporter", LEAKED_PATH)
    return model


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


class TestFindDebugMetadata:
    def test_detects_all_locations(self) -> None:
        found = find_debug_metadata(_make_model_with_debug_info())
        assert any("node[0].doc_string" in f for f in found)
        assert any("node[0].metadata_props" in f for f in found)
        assert any("graph.metadata_props" in f for f in found)
        assert any("graph.doc_string" in f for f in found)
        assert any("input[0].metadata_props" in f for f in found)
        assert any("output[0].metadata_props" in f for f in found)
        assert any("value_info[0].metadata_props" in f for f in found)
        assert any("initializer[0].metadata_props" in f for f in found)
        assert any(f.startswith("model.") for f in found)

    def test_clean_model_has_none(self) -> None:
        assert find_debug_metadata(strip_debug_metadata(_make_model_with_debug_info())) == []


class TestStripDebugMetadata:
    def test_removes_leaked_path_from_serialized_bytes(self) -> None:
        stripped = strip_debug_metadata(_make_model_with_debug_info())
        assert b"site-packages" not in stripped.SerializeToString()

    def test_does_not_mutate_input(self) -> None:
        original = _make_model_with_debug_info()
        before = original.SerializeToString()
        strip_debug_metadata(original)
        assert original.SerializeToString() == before

    def test_inference_unchanged_and_model_valid(self) -> None:
        original = _make_model_with_debug_info()
        stripped = strip_debug_metadata(original)
        onnx.checker.check_model(stripped)
        x = np.array([[0.5, -1.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(stripped, x), _run(original, x))

    def test_keeps_graph_structure(self) -> None:
        original = _make_model_with_debug_info()
        stripped = strip_debug_metadata(original)
        assert [n.op_type for n in stripped.graph.node] == ["Add", "Identity"]
        assert [i.name for i in stripped.graph.input] == ["x"]
        assert [o.name for o in stripped.graph.output] == ["y"]
        assert stripped.opset_import == original.opset_import

    def test_strips_subgraphs(self) -> None:
        inner = _make_model_with_debug_info().graph
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])
        if_node = helper.make_node("If", ["cond"], ["y"], then_branch=inner, else_branch=inner)
        outer = helper.make_graph([if_node], "outer", [cond], [out])
        model = helper.make_model(outer, opset_imports=[helper.make_opsetid("", 18)])
        stripped = strip_debug_metadata(model)
        assert find_debug_metadata(stripped) == []
        assert b"site-packages" not in stripped.SerializeToString()

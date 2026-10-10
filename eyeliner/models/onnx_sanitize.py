"""配布用 ONNX からエクスポート時のデバッグ情報を取り除く。

torch.onnx（dynamo エクスポータ）は各ノード・グラフ・入出力・initializer に
metadata_props（スタックトレース、FX ノード表現、モジュール階層など）や doc_string を
埋め込む。スタックトレースにはエクスポート環境の絶対パスが含まれ、推論には一切使われないため、
配布前に除去する。演算・重み・入出力名・opset には触れない。
"""
from __future__ import annotations

from collections.abc import Iterator

import onnx

_DEBUG_FIELDS = ("doc_string", "metadata_props")


def _has_field(proto, field: str) -> bool:
    return field in proto.DESCRIPTOR.fields_by_name


def _is_set(proto, field: str) -> bool:
    if not _has_field(proto, field):
        return False
    value = getattr(proto, field)
    return len(value) > 0


def _graph_children(graph: onnx.GraphProto, prefix: str) -> Iterator[tuple[str, object]]:
    """graph 配下でデバッグ情報を持ちうる proto を (位置ラベル, proto) で列挙する（サブグラフ含む）。"""
    yield prefix, graph
    for coll in ("input", "output", "value_info", "initializer", "sparse_initializer"):
        for i, item in enumerate(getattr(graph, coll)):
            yield f"{prefix}.{coll}[{i}]", item
    for i, node in enumerate(graph.node):
        node_label = f"{prefix}.node[{i}]"
        yield node_label, node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _graph_children(attr.g, f"{node_label}.{attr.name}")
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for j, sub in enumerate(attr.graphs):
                    yield from _graph_children(sub, f"{node_label}.{attr.name}[{j}]")


def _model_children(model: onnx.ModelProto) -> Iterator[tuple[str, object]]:
    yield "model", model
    yield from _graph_children(model.graph, "graph")
    for i, func in enumerate(model.functions):
        func_label = f"function[{i}]"
        yield func_label, func
        for j, node in enumerate(func.node):
            yield f"{func_label}.node[{j}]", node


def find_debug_metadata(model: onnx.ModelProto) -> list[str]:
    """デバッグ情報（doc_string / metadata_props）が残っている箇所のラベル一覧を返す。"""
    return [
        f"{label}.{field}"
        for label, proto in _model_children(model)
        for field in _DEBUG_FIELDS
        if _is_set(proto, field)
    ]


def strip_debug_metadata(model: onnx.ModelProto) -> onnx.ModelProto:
    """デバッグ情報を除去したモデルのコピーを返す（引数のモデルは変更しない）。"""
    cleaned = onnx.ModelProto()
    cleaned.CopyFrom(model)
    for _, proto in _model_children(cleaned):
        for field in _DEBUG_FIELDS:
            if _has_field(proto, field):
                proto.ClearField(field)
    return cleaned

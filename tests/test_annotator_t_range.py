"""annotator.js の物理 t 座標系定数が patch_utils の T_MIN / T_MAX と一致することのテスト。

サーバーは patch_utils の T_MIN / T_MAX で t_norm と物理 t を相互変換する。annotator.js の
定数がずれると、DB の保存値は正しいまま画面の t 表示だけが誤る。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from eyeliner.data.patch_utils import T_MAX, T_MIN

_ANNOTATOR_JS = Path(__file__).parent.parent / "annotator" / "static" / "annotator.js"


def _js_constant(name: str) -> float:
    source = _ANNOTATOR_JS.read_text(encoding="utf-8")
    match = re.search(rf"const {name}\s*=\s*(-?\d+(?:\.\d+)?)\s*;", source)
    assert match, f"annotator.js に定数 {name} が見つからない"
    return float(match.group(1))


@pytest.mark.parametrize(
    "js_name, expected",
    [("ANNOTATOR_T_MIN", T_MIN), ("ANNOTATOR_T_MAX", T_MAX)],
)
def test_annotator_js_t_range_matches_patch_utils(js_name: str, expected: float) -> None:
    assert _js_constant(js_name) == pytest.approx(expected)

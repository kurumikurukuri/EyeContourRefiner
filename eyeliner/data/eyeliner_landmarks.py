"""アイライン抽出用のランドマークインデックスマップ。

MediaPipe FaceMesh の478点から P/Q 候補点を選ぶための、目の部位（left/right）×
upper/lower ごとのランドマークインデックス一覧。`annotator/preprocessor.py` の
サンプル生成時に参照する。
"""
from __future__ import annotations

EYELINER_CORRECTION_MAP: dict[str, dict[str, list[int]]] = {
    "eyeliner_left": {
        "upper": [1507, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 130, 226],
        "lower": [130, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 133],
    },
    "eyeliner_right": {
        "upper": [1707, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 359, 446],
        "lower": [359, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 362],
    },
}

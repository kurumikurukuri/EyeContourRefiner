"""アイライン抽出用のランドマークインデックスマップ。

MediaPipe FaceMesh の478点から P/Q 候補点を選ぶための、目の部位（left/right）×
upper/lower ごとのランドマーク一覧。要素は int（アンカー点）か
('edge', p_idx, q_idx) タプル（アノテーション対象の P-Q ペア）で、
('edge', ...) の並び順が point_index（0〜6）に対応する。
`annotator/preprocessor.py` のサンプル生成時に参照する。
"""
from __future__ import annotations

EYELINER_CORRECTION_MAP: dict[str, dict[str, tuple[int | tuple[str, int, int], ...]]] = {
    "eyeliner_left": {
        "upper": (
            ("edge", 173, 190), ("edge", 157, 56), ("edge", 158, 28), ("edge", 159, 27),
            ("edge", 160, 29), ("edge", 161, 30), ("edge", 246, 247),
            130, 226,
        ),
        "lower": (
            130,
            ("edge", 7, 25), ("edge", 163, 110), ("edge", 144, 24), ("edge", 145, 23),
            ("edge", 153, 22), ("edge", 154, 26), ("edge", 155, 112),
            133,
        ),
    },
    "eyeliner_right": {
        "upper": (
            ("edge", 398, 414), ("edge", 384, 286), ("edge", 385, 258), ("edge", 386, 257),
            ("edge", 387, 259), ("edge", 388, 260), ("edge", 466, 467),
            359, 446,
        ),
        "lower": (
            359,
            ("edge", 249, 255), ("edge", 390, 339), ("edge", 373, 254), ("edge", 374, 253),
            ("edge", 380, 252), ("edge", 381, 256), ("edge", 382, 341),
            362,
        ),
    },
}

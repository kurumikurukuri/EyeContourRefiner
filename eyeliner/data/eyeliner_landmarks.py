"""アイライン P/Q ランドマークの対応表。

MediaPipe FaceMesh（refine_landmarks=True, 478 点）から、アノテーション用の P/Q 2 点を
取り出すための定義。`annotator/preprocessor.py` と
`tools/imagesdb_maintenance/preprocess_new_images.py` がサンプル生成時に参照する。

- `EDGE_LANDMARK_DIRECTIONS`: edge ID → (P, Q) の MediaPipe インデックス。face_makeup
  （https://github.com/kurumikurukuri/face_makeup）の `src/face_makeup/custom_landmark.py`
  （commit 09034270）の同名テーブルを移植したもの。edge ID（1401〜1707）は 478 点の外に
  ある論理 ID で、1 つが 1 本の P→Q エッジを表す（MediaPipe のインデックス 0〜477 ではない）。
- `EYELINER_SAMPLE_EDGE_IDS`: samples.db の (part_name, sub_key, point_index) に対応する
  edge ID。point_index 0〜6 の並び。mediapipe 0.10.21 を 400 画像に実行し、全 28 点が
  DB の px/py/qx/qy と誤差 0.0000 px で一致することを確認した。
  face_makeup の `EYELINER_*_INDICES` は描画用ポリラインで、上まぶたに 1408/1608 と
  下まぶた末尾のエッジ（1507/1707）の重複を含むが、アノテーション対象は upper/lower とも
  7 本なので、そのリストからは導出せずこの表で明示する。
- `EYELINER_SAMPLE_PQ`: 上記を `(edge_id, p_idx, q_idx)` に解決したもの。
"""
from __future__ import annotations

EDGE_LANDMARK_DIRECTIONS: dict[int, tuple[int, int]] = {
    # 上まぶた（左）。1408 はアノテーション対象外。
    1401: (173, 190), 1402: (157, 56), 1403: (158, 28), 1404: (159, 27),
    1405: (160, 29), 1406: (161, 30), 1407: (246, 247), 1408: (33, 113),
    # 下まぶた（左）
    1501: (7, 25), 1502: (163, 110), 1503: (144, 24), 1504: (145, 23),
    1505: (153, 22), 1506: (154, 26), 1507: (155, 112),
    # 上まぶた（右）。1608 はアノテーション対象外。
    1601: (398, 414), 1602: (384, 286), 1603: (385, 258), 1604: (386, 257),
    1605: (387, 259), 1606: (388, 260), 1607: (466, 467), 1608: (263, 342),
    # 下まぶた（右）
    1701: (249, 255), 1702: (390, 339), 1703: (373, 254), 1704: (374, 253),
    1705: (380, 252), 1706: (381, 256), 1707: (382, 341),
}

EYELINER_SAMPLE_EDGE_IDS: dict[str, dict[str, tuple[int, ...]]] = {
    "eyeliner_left": {
        "upper": (1401, 1402, 1403, 1404, 1405, 1406, 1407),
        "lower": (1501, 1502, 1503, 1504, 1505, 1506, 1507),
    },
    "eyeliner_right": {
        "upper": (1601, 1602, 1603, 1604, 1605, 1606, 1607),
        "lower": (1701, 1702, 1703, 1704, 1705, 1706, 1707),
    },
}


def _resolve_pq(edge_ids: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple((edge_id, *EDGE_LANDMARK_DIRECTIONS[edge_id]) for edge_id in edge_ids)


EYELINER_SAMPLE_PQ: dict[str, dict[str, tuple[tuple[int, int, int], ...]]] = {
    part_name: {sub_key: _resolve_pq(edge_ids) for sub_key, edge_ids in sub_map.items()}
    for part_name, sub_map in EYELINER_SAMPLE_EDGE_IDS.items()
}

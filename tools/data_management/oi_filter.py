"""Open Images の画像取得対象を、画像を落とす前に注釈 CSV だけで絞り込む。

`FilterConfig` に条件を集約し、条件ごとのハッシュでフィルタ結果（ID 一覧）をキャッシュする。
画像本体の転送が始まる前に全条件を適用するので、不要な画像は一切ダウンロードされない。

有効な顔 bbox とは、`face_class` のボックスのうち以下を満たすもの:
    - IsDepiction / IsGroupOf / IsInside がいずれも 0（`excluded_flags`）
    - 面積（正規化座標）が `min_face_area` 以上
    - アスペクト比（w/h）が `min_face_aspect`〜`max_face_aspect`
画像は、必須クラスを全て含み・除外クラスを含まず・有効な顔数が `min_faces`〜`max_faces` のとき残る。
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Collection, Dict, FrozenSet, Iterator, Optional, Set, Tuple

from tools.data_management import _fileio
from tools.data_management.oi_metadata import PERMISSIVE_LICENSE_URLS, load_label_maps

logger = logging.getLogger(__name__)

# `exclude_children=True` のとき除外に加えるクラス（Open Images V7 に "Baby" クラスは無い）。
CHILD_CLASSES: tuple[str, ...] = ("Boy", "Girl")

_BBOX_COLUMNS: tuple[str, ...] = ("ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax")


@dataclass(frozen=True)
class FilterConfig:
    """画像取得フィルタの条件。既定値が標準の取得条件。"""

    required_classes: tuple[str, ...] = ("Human face",)
    excluded_classes: tuple[str, ...] = (
        "Doll", "Toy", "Teddy bear", "Poster", "Sculpture", "Infant bed",
    )
    face_class: str = "Human face"
    min_face_area: float = 0.01
    min_face_aspect: float = 0.2
    max_face_aspect: float = 2.0
    min_faces: int = 1
    max_faces: int = 4
    excluded_flags: tuple[str, ...] = ("IsDepiction", "IsGroupOf", "IsInside")
    allowed_licenses: tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(PERMISSIVE_LICENSE_URLS))
    )
    exclude_children: bool = False

    def __post_init__(self) -> None:
        if not self.required_classes:
            raise ValueError("required_classes は 1 つ以上必要です")
        if not 0.0 <= self.min_face_area <= 1.0:
            raise ValueError(f"min_face_area は 0〜1 が必要です: {self.min_face_area}")
        if not 0.0 < self.min_face_aspect <= self.max_face_aspect:
            raise ValueError(
                "顔アスペクト比は 0 < min <= max が必要です: "
                f"{self.min_face_aspect}〜{self.max_face_aspect}"
            )
        if not 1 <= self.min_faces <= self.max_faces:
            raise ValueError(f"顔数は 1 <= min <= max が必要です: {self.min_faces}〜{self.max_faces}")

    def effective_excluded_classes(self) -> frozenset[str]:
        extra = CHILD_CLASSES if self.exclude_children else ()
        return frozenset(self.excluded_classes) | frozenset(extra)

    def cache_key(self) -> str:
        """条件から決まる 12 桁のハッシュ。条件が同じなら同じキャッシュを共有する。"""
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _is_valid_face(record: Dict[str, str], config: FilterConfig) -> bool:
    """bbox 1 行が有効な顔か（属性フラグ・面積・アスペクト比）。不正な値の行は無効扱い。"""
    try:
        for flag in config.excluded_flags:
            if int(record.get(flag) or 0):
                return False
        width = float(record["XMax"]) - float(record["XMin"])
        height = float(record["YMax"]) - float(record["YMin"])
    except (KeyError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    if width * height < config.min_face_area:
        return False
    return config.min_face_aspect <= width / height <= config.max_face_aspect


def _iter_bbox_records(
    annotations_csv: Path,
    extra_columns: Collection[str],
) -> Iterator[Optional[Dict[str, str]]]:
    """注釈 CSV を dict で返す。列位置を先に解決し、DictReader より高速に読む。"""
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        missing = [c for c in _BBOX_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{annotations_csv.name}: 必要な列がありません: {', '.join(missing)}")
        wanted = list(_BBOX_COLUMNS) + [c for c in extra_columns if c in header]
        positions = {name: header.index(name) for name in wanted}
        width = max(positions.values()) + 1
        for row in reader:
            if len(row) < width:
                yield None  # 壊れた行（呼び出し側で無効扱い）
                continue
            yield {name: row[pos] for name, pos in positions.items()}


def _resolve_class_ids(
    config: FilterConfig,
    name_to_id: Dict[str, str],
) -> Tuple[FrozenSet[str], FrozenSet[str], str]:
    """クラス名を Open Images の ID に解決する。戻り値は (必須 ID, 除外 ID, 顔クラス ID)。"""
    missing_required = [c for c in config.required_classes if c not in name_to_id]
    if missing_required:
        raise ValueError(f"必須クラスが classes.csv にありません: {missing_required}")
    face_label = name_to_id.get(config.face_class)
    if face_label is None:
        raise ValueError(f"顔クラスが classes.csv にありません: {config.face_class!r}")
    excluded_names = config.effective_excluded_classes()
    missing_excluded = sorted(excluded_names - set(name_to_id))
    if missing_excluded:
        logger.warning("除外クラスが classes.csv にありません（スキップ）: %s", missing_excluded)
    return (
        frozenset(name_to_id[c] for c in config.required_classes),
        frozenset(name_to_id[c] for c in excluded_names if c in name_to_id),
        face_label,
    )


def filter_image_ids(
    candidate_ids: Collection[str],
    annotations_csv: Path,
    classes_csv: Path,
    config: FilterConfig,
) -> Set[str]:
    """`candidate_ids`（ライセンス適合済み）を注釈 CSV の条件で絞り込んで返す。

    Raises:
        ValueError: 必須クラスが classes.csv に無い / 注釈 CSV の列が不足。
    """
    name_to_id, _ = load_label_maps(classes_csv)
    required_ids, excluded_ids, face_label = _resolve_class_ids(config, name_to_id)

    candidates = candidate_ids if isinstance(candidate_ids, (set, frozenset)) else set(candidate_ids)
    labels: Dict[str, Set[str]] = defaultdict(set)
    valid_faces: Dict[str, int] = defaultdict(int)
    skipped_rows = 0

    logger.info("注釈 CSV を解析中: %s", annotations_csv)
    for record in _iter_bbox_records(annotations_csv, config.excluded_flags):
        if record is None:
            skipped_rows += 1
            continue
        image_id = record["ImageID"]
        if image_id not in candidates:
            continue
        label = record["LabelName"]
        labels[image_id].add(label)
        if label == face_label and _is_valid_face(record, config):
            valid_faces[image_id] += 1
    if skipped_rows:
        logger.warning("列数が足りない注釈行をスキップ: %d 行", skipped_rows)

    kept = {
        image_id
        for image_id, image_labels in labels.items()
        if required_ids <= image_labels
        and not image_labels & excluded_ids
        and config.min_faces <= valid_faces[image_id] <= config.max_faces
    }
    logger.info("フィルタ結果: %d / %d 枚", len(kept), len(candidates))
    return kept


def filtered_ids_cache_path(split: str, cache_dir: Path, config: FilterConfig) -> Path:
    return cache_dir / f"{split}_filtered_{config.cache_key()}.txt"


def save_ids(ids: Collection[str], path: Path) -> None:
    """ID 一覧を昇順 1 行 1 件で保存する（一時ファイル経由で置換）。"""
    with _fileio.atomic_replace(path) as part:
        part.write_text("".join(f"{image_id}\n" for image_id in sorted(ids)), encoding="utf-8", newline="")


def load_ids(path: Path) -> Set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_or_compute_filtered_ids(
    split: str,
    cache_dir: Path,
    config: FilterConfig,
    *,
    compute: Callable[[], Set[str]],
    refresh: bool = False,
) -> Set[str]:
    """条件ハッシュ付きキャッシュがあれば読み、無ければ `compute()` して保存する。"""
    path = filtered_ids_cache_path(split, cache_dir, config)
    if path.exists() and not refresh:
        logger.info("フィルタ済み ID キャッシュを使用: %s", path)
        return load_ids(path)
    ids = compute()
    save_ids(ids, path)
    logger.info("フィルタ済み ID キャッシュを保存: %s (%d 件)", path, len(ids))
    return ids

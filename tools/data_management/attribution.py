"""Open Images の attribution（帰属情報）CSV の管理。

取得した画像の著者・出典・ライセンスを `data/open_images_attribution.csv` に追記する
（CC BY 2.0 の表示義務に対応）。複数プロセス（`--shard` 並列実行）から同時に追記されても
壊れないよう、サイドカーのロックファイルで排他制御し、ロック下で既存 ID を読み直して
同一 Image ID の二重登録を防ぐ。
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Dict, Iterable, Iterator, List, Set

from tools.data_management.oi_metadata import ImageMeta

ATTRIBUTION_FIELDS: tuple[str, ...] = (
    "image_id", "license_url", "original_url", "author", "title",
)
DEFAULT_ATTRIBUTION_CSV = Path("data/open_images_attribution.csv")

_LOCK_TIMEOUT_SEC = 120.0
_LOCK_POLL_SEC = 0.05


class AttributionError(ValueError):
    """attribution CSV が想定形式でない。"""


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """`path` のサイドカー（`<name>.lock`）を使うプロセス間排他ロック。"""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
        while True:
            try:
                _try_lock(lock_file)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"attribution CSV のロック取得がタイムアウトしました: {lock_path}") from None
                time.sleep(_LOCK_POLL_SEC)
        try:
            yield
        finally:
            _unlock(lock_file)


if sys.platform == "win32":
    import msvcrt

    def _try_lock(f: IO[bytes]) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f: IO[bytes]) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(f: IO[bytes]) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f: IO[bytes]) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_id" not in (reader.fieldnames or []):
            raise AttributionError(f"{path}: image_id 列がありません（attribution CSV ではない可能性）")
        return list(reader)


def read_attribution(path: Path) -> Dict[str, ImageMeta]:
    """attribution CSV を {image_id: ImageMeta} で返す。ファイルが無ければ空。"""
    return {
        row["image_id"]: ImageMeta(
            image_id=row["image_id"],
            license_url=row.get("license_url") or "",
            original_url=row.get("original_url") or "",
            author=row.get("author") or "",
            title=row.get("title") or "",
        )
        for row in _rows(path)
        if row.get("image_id")
    }


def known_image_ids(path: Path) -> Set[str]:
    """attribution CSV に登録済みの Image ID 集合を返す。"""
    return set(read_attribution(path))


def _line_terminator(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4096)
    return "\r\n" if b"\r\n" in head else "\n"


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) in (b"\n", b"\r")


def append_attribution(path: Path, metas: Iterable[ImageMeta]) -> int:
    """未登録の画像だけを attribution CSV に追記し、追記した件数を返す。

    追記は排他ロック下で行い、既存 ID・入力内の重複はスキップする。既存ファイルの
    改行コードは維持し、末尾改行が無ければ補ってから追記する。
    """
    candidates = list(metas)
    if not candidates:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        existing = {row["image_id"] for row in _rows(path) if row.get("image_id")}
        fresh: List[ImageMeta] = []
        for meta in candidates:
            if meta.image_id and meta.image_id not in existing:
                existing.add(meta.image_id)
                fresh.append(meta)
        if not fresh:
            return 0

        has_content = path.exists() and path.stat().st_size > 0
        terminator = _line_terminator(path) if has_content else "\n"
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator=terminator)
        if has_content and not _ends_with_newline(path):
            buffer.write(terminator)
        if not has_content:
            writer.writerow(ATTRIBUTION_FIELDS)
        for m in fresh:
            writer.writerow([m.image_id, m.license_url, m.original_url, m.author, m.title])

        with open(path, "ab") as f:
            f.write(buffer.getvalue().encode("utf-8"))
        return len(fresh)

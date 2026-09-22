"""Open Images V7 のメタデータ・注釈 CSV の取得とキャッシュ。

- 画像メタデータ（ImageID / License / OriginalURL / Author / Title）から、再配布可能な
  ライセンス（CC0 / CC BY 2.0）の画像だけを split ごとに取り出してキャッシュする。
- 注釈 CSV（bbox）・classes.csv は、既存の fiftyone キャッシュ（~/fiftyone/open-images-v7）が
  あれば **読み取り専用で再利用** し、無ければ `data/.open_images_cache/` に取得する。
  数 GB ある注釈 CSV の再取得を避けるための措置で、fiftyone 自体には依存しない。

ダウンロードは `.part` に書いてから rename するため、中断してもキャッシュを壊さない。
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Iterator, Mapping

import requests

from tools.data_management import _fileio
from tools.data_management.manifest_source import is_valid_image_id

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "validation", "test")

# 許可するライセンス（正規化済み完全一致）。startswith ではなく完全一致で判定し、
# 管轄移転版（/ca/, /de/ 等）や BY-SA / BY-NC / BY-ND の混入を防ぐ。
PERMISSIVE_LICENSE_URLS: frozenset[str] = frozenset({
    "https://creativecommons.org/publicdomain/zero/1.0",  # CC0
    "https://creativecommons.org/licenses/by/2.0",        # CC BY 2.0
})

METADATA_URLS: Mapping[str, str] = MappingProxyType({
    "validation": (
        "https://storage.googleapis.com/openimages/2018_04/"
        "validation/validation-images-with-rotation.csv"
    ),
    "test": (
        "https://storage.googleapis.com/openimages/2018_04/"
        "test/test-images-with-rotation.csv"
    ),
    "train": (
        "https://storage.googleapis.com/openimages/2018_04/"
        "train/train-images-boxable-with-rotation.csv"
    ),
})

ANNOTATION_URLS: Mapping[str, str] = MappingProxyType({
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
})

CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

DEFAULT_CACHE_DIR = Path("data/.open_images_cache")
FIFTYONE_CACHE_ROOT = Path.home() / "fiftyone" / "open-images-v7"

_META_FIELDS = ("image_id", "license_url", "original_url", "author", "title")
_DOWNLOAD_CHUNK = 1 << 16
_DOWNLOAD_TIMEOUT = 600
_STREAM_TIMEOUT = 120

Downloader = Callable[[str, Path], None]


@dataclass(frozen=True)
class ImageMeta:
    """1 枚の画像に紐づくライセンス・帰属情報。"""

    image_id: str
    license_url: str
    original_url: str
    author: str
    title: str


def normalize_license(url: str) -> str:
    """ライセンス URL の前後空白と末尾スラッシュを除いて比較用に正規化する。"""
    return url.strip().rstrip("/")


def is_permissive_license(license_url: str) -> bool:
    """再配布可能なライセンス（CC0 / CC BY 2.0）かを完全一致で判定する。"""
    return normalize_license(license_url) in PERMISSIVE_LICENSE_URLS


def _safe_decode(line: bytes) -> str:
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        return line.decode("latin-1", errors="replace")


def _stream_csv_rows(url: str) -> Iterator[Dict[str, str]]:
    """URL から CSV をストリーミングで行ごとに返す（空行はスキップ）。"""
    with requests.get(url, stream=True, timeout=_STREAM_TIMEOUT) as resp:
        resp.raise_for_status()
        lines = (_safe_decode(raw) for raw in resp.iter_lines() if raw)
        yield from csv.DictReader(lines)


def download_file(url: str, dest: Path, *, get=requests.get) -> None:
    """URL を `dest` に取得する。一時ファイルに書いて長さを検証してから rename し、失敗時は残骸を消す。

    Raises:
        OSError: 受信サイズが Content-Length と一致しない（途中で切れた）。
    """
    logger.info("Downloading %s", url)
    with _fileio.atomic_replace(dest) as part:
        with get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            written = 0
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            expected = _fileio.expected_length(getattr(resp, "headers", {}))
        if expected is not None and written != expected:
            raise OSError(f"ダウンロードのサイズが一致しません: {written} / {expected} bytes ({url})")


def _save_meta_cache(metas: Dict[str, ImageMeta], path: Path) -> None:
    with _fileio.atomic_replace(path) as part:
        with open(part, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_META_FIELDS)
            for meta in metas.values():
                writer.writerow([meta.image_id, meta.license_url, meta.original_url, meta.author, meta.title])


def _load_meta_cache(path: Path) -> Dict[str, ImageMeta]:
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["image_id"]: ImageMeta(
                image_id=row["image_id"],
                license_url=row["license_url"],
                original_url=row["original_url"],
                author=row["author"],
                title=row["title"],
            )
            for row in csv.DictReader(f)
            if is_valid_image_id(row["image_id"])  # 旧ツールが作ったキャッシュ内の不正 ID も持ち込まない
        }


def fetch_permissive_metadata(
    split: str,
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, ImageMeta]:
    """指定 split の再配布可能ライセンス画像のメタデータを返す（結果はキャッシュ）。

    Args:
        split: "train" / "validation" / "test"
        cache_dir: `{split}_permissive_meta.csv` の保存先
        refresh: True ならキャッシュを無視して再取得する

    Raises:
        ValueError: split が不正。
    """
    if split not in METADATA_URLS:
        raise ValueError(f"不正な split: {split!r}。有効な値: {list(METADATA_URLS)}")

    cache_file = cache_dir / f"{split}_permissive_meta.csv"
    if cache_file.exists() and not refresh:
        logger.info("メタデータキャッシュを使用: %s", cache_file)
        return _load_meta_cache(cache_file)

    logger.info("メタデータを取得中 (split=%s): %s", split, METADATA_URLS[split])
    metas: Dict[str, ImageMeta] = {}
    total = 0
    for row in _stream_csv_rows(METADATA_URLS[split]):
        total += 1
        license_url = row.get("License") or ""
        image_id = (row.get("ImageID") or "").strip()
        # リモート由来の ID は取り込み時に検証する（不正な ID がキャッシュや取得計画に残らないように）
        if not is_valid_image_id(image_id) or not is_permissive_license(license_url):
            continue
        metas[image_id] = ImageMeta(
            image_id=image_id,
            license_url=license_url.strip(),
            original_url=row.get("OriginalURL") or "",
            author=row.get("Author") or "",
            title=row.get("Title") or "",
        )
    logger.info("%d 件中 %d 件が許可ライセンス", total, len(metas))

    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_meta_cache(metas, cache_file)
    return metas


def _resolve_csv(
    fiftyone_path: Path,
    own_path: Path,
    url: str,
    refresh: bool,
    download: Downloader,
) -> Path:
    if fiftyone_path.exists() and not refresh:
        return fiftyone_path  # 読み取り専用で再利用（変更・削除しない）
    if refresh or not own_path.exists():
        download(url, own_path)
    return own_path


def resolve_classes_csv(
    split: str,
    cache_dir: Path,
    refresh: bool = False,
    *,
    fiftyone_root: Path = FIFTYONE_CACHE_ROOT,
    download: Downloader = download_file,
) -> Path:
    """classes.csv のパスを返す。fiftyone キャッシュを優先し、無ければ取得する。"""
    if split not in SPLITS:
        raise ValueError(f"不正な split: {split!r}。有効な値: {list(SPLITS)}")
    return _resolve_csv(
        fiftyone_root / split / "metadata" / "classes.csv",
        cache_dir / "classes.csv",
        CLASSES_URL,
        refresh,
        download,
    )


def resolve_annotations_csv(
    split: str,
    cache_dir: Path,
    refresh: bool = False,
    *,
    fiftyone_root: Path = FIFTYONE_CACHE_ROOT,
    download: Downloader = download_file,
) -> Path:
    """bbox 注釈 CSV のパスを返す。fiftyone キャッシュを優先し、無ければ取得する。"""
    if split not in ANNOTATION_URLS:
        raise ValueError(f"不正な split: {split!r}。有効な値: {list(ANNOTATION_URLS)}")
    return _resolve_csv(
        fiftyone_root / split / "labels" / "detections.csv",
        cache_dir / f"{split}_annotations_bbox.csv",
        ANNOTATION_URLS[split],
        refresh,
        download,
    )


def load_label_maps(classes_csv: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    """classes.csv から (name→id, id→name) を返す。"""
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    with open(classes_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                id_to_name[row[0]] = row[1]
                name_to_id[row[1]] = row[0]
    return name_to_id, id_to_name

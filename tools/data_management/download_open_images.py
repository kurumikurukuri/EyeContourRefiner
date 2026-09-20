"""
Open Images V7 から商用利用・再配布可能な顔画像を取得するスクリプト。

ライセンスフィルタ:
    CC0 / CC BY 2.0 のみ取得（MIT ライセンスで再配布可能）
    CC BY-SA / CC BY-NC / CC BY-ND は除外

高精度フィルタ（アノテーション CSV を使って画像 DL 前に適用）:
    必須:
        - Person
        - Human face

    除外:
        - Boy / Girl（Child/Baby/Infant の代替）
        - Doll / Toy / Teddy bear
        - Poster / Sculpture

    属性:
        - IsDepiction == 0
        - IsGroupOf == 0
        - IsInside == 0

    品質:
        - 最大顔 bbox 面積 >= 5%
        - 顔数 <= 4
        - 顔アスペクト比 0.6〜1.8

これにより:
    - 子供画像
    - 人形・玩具
    - ポスター・彫刻
    - 群衆・遠景
を大幅に削減。

使い方:
    python tools/data_management/download_open_images.py --output data/images/
    python tools/data_management/download_open_images.py --split train --max-images 2000 --output data/images/

    # train 分割を3回に分けてダウンロード（アノテーション前処理は初回のみ）
    python tools/data_management/download_open_images.py --split train --shard 0 --shard-count 3
    python tools/data_management/download_open_images.py --split train --shard 1 --shard-count 3
    python tools/data_management/download_open_images.py --split train --shard 2 --shard-count 3
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

PERMISSIVE_LICENSE_URLS: frozenset[str] = frozenset({
    "https://creativecommons.org/publicdomain/zero/1.0",
    "https://creativecommons.org/licenses/by/2.0",
})

METADATA_URLS: Dict[str, str] = {
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
}

ANNOTATION_URLS: Dict[str, str] = {
    "validation": (
        "https://storage.googleapis.com/openimages/v5/"
        "validation-annotations-bbox.csv"
    ),
    "test": (
        "https://storage.googleapis.com/openimages/v5/"
        "test-annotations-bbox.csv"
    ),
    "train": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-train-annotations-bbox.csv"
    ),
}

CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

# fiftyone が保存するキャッシュのデフォルト位置
_FO_CACHE_ROOT = Path.home() / "fiftyone" / "open-images-v7"

DEFAULT_CACHE_DIR = Path("data/.open_images_cache")
ATTRIBUTION_CSV = Path("data/open_images_attribution.csv")

REQUIRED_CLASSES: frozenset[str] = frozenset({"Human face"})

EXCLUDED_CLASSES: frozenset[str] = frozenset({
    "Doll", "Toy", "Teddy bear",
    "Poster", "Sculpture",
    "Infant bed",
})

FACE_CLASS = "Human face"

MIN_FACE_AREA_RATIO = 0.01
MAX_FACES = 4
MIN_FACE_ASPECT = 0.2
MAX_FACE_ASPECT = 2


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

class ImageMeta:
    __slots__ = ("image_id", "license_url", "original_url", "author", "title")

    def __init__(
        self,
        image_id: str,
        license_url: str,
        original_url: str,
        author: str,
        title: str,
    ) -> None:
        self.image_id = image_id
        self.license_url = license_url
        self.original_url = original_url
        self.author = author
        self.title = title


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _normalize_license(url: str) -> str:
    return url.strip().rstrip("/")


def _is_permissive(license_url: str) -> bool:
    return _normalize_license(license_url) in PERMISSIVE_LICENSE_URLS


def _safe_decode(line: bytes) -> Optional[str]:
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        return line.decode("latin-1", errors="replace")


def _stream_csv_rows(url: str) -> Iterator[Dict[str, str]]:
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        lines_iter = (
            decoded
            for raw in resp.iter_lines()
            if raw
            for decoded in (_safe_decode(raw),)
            if decoded is not None
        )
        reader = csv.DictReader(lines_iter)
        yield from reader


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)


# ---------------------------------------------------------------------------
# ライセンスメタデータ取得
# ---------------------------------------------------------------------------

def fetch_permissive_metadata(
    split: str,
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, ImageMeta]:
    if split not in METADATA_URLS:
        raise ValueError(f"不正な split: {split}")

    cache_file = cache_dir / f"{split}_permissive_meta.csv"

    if cache_file.exists() and not refresh:
        return _load_meta_cache(cache_file)

    metas: Dict[str, ImageMeta] = {}

    for row in _stream_csv_rows(METADATA_URLS[split]):
        license_url = row.get("License") or ""
        if not _is_permissive(license_url):
            continue

        image_id = (row.get("ImageID") or "").strip()
        if not image_id:
            continue

        metas[image_id] = ImageMeta(
            image_id=image_id,
            license_url=license_url.strip(),
            original_url=row.get("OriginalURL") or "",
            author=row.get("Author") or "",
            title=row.get("Title") or "",
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_meta_cache(metas, cache_file)

    return metas


def _save_meta_cache(metas: Dict[str, ImageMeta], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "license_url", "original_url", "author", "title"])
        for m in metas.values():
            writer.writerow([m.image_id, m.license_url, m.original_url, m.author, m.title])


def _load_meta_cache(path: Path) -> Dict[str, ImageMeta]:
    metas: Dict[str, ImageMeta] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = ImageMeta(
                image_id=row["image_id"],
                license_url=row["license_url"],
                original_url=row["original_url"],
                author=row["author"],
                title=row["title"],
            )
            metas[m.image_id] = m

    return metas


# ---------------------------------------------------------------------------
# フィルタ済みIDキャッシュ
# ---------------------------------------------------------------------------

def _save_filtered_ids(ids: Set[str], path: Path) -> None:
    """フィルタ済みの画像ID一覧をキャッシュファイルに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for image_id in sorted(ids):
            f.write(image_id + "\n")


def _load_filtered_ids(path: Path) -> Set[str]:
    """キャッシュファイルからフィルタ済み画像ID一覧を読み込む。"""
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                ids.add(stripped)
    return ids


def _filtered_ids_cache_path(split: str, cache_dir: Path) -> Path:
    return cache_dir / f"{split}_filtered_ids.txt"


# ---------------------------------------------------------------------------
# アノテーション CSV 事前フィルタリング
# ---------------------------------------------------------------------------

def _resolve_classes_csv(split: str, cache_dir: Path, refresh: bool) -> Path:
    """fiftyone キャッシュを優先し、なければ独自にダウンロード"""
    fo_cache = _FO_CACHE_ROOT / split / "metadata" / "classes.csv"
    if fo_cache.exists() and not refresh:
        return fo_cache
    our_cache = cache_dir / "classes.csv"
    if not our_cache.exists() or refresh:
        _download_file(CLASSES_URL, our_cache)
    return our_cache


def _resolve_annotations_csv(split: str, cache_dir: Path, refresh: bool) -> Path:
    """fiftyone キャッシュを優先し、なければ独自にダウンロード"""
    fo_cache = _FO_CACHE_ROOT / split / "labels" / "detections.csv"
    if fo_cache.exists() and not refresh:
        return fo_cache
    our_cache = cache_dir / f"{split}_annotations_bbox.csv"
    if not our_cache.exists() or refresh:
        _download_file(ANNOTATION_URLS[split], our_cache)
    return our_cache


def _load_label_maps(classes_csv: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    """classes.csv から (name→id, id→name) マップを返す"""
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    with open(classes_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                id_to_name[row[0]] = row[1]
                name_to_id[row[1]] = row[0]
    return name_to_id, id_to_name


def _valid_face_row(row: Dict[str, str]) -> bool:
    """bbox CSV 行から顔の品質チェック（属性・面積・アスペクト比）"""
    for flag in ("IsDepiction", "IsGroupOf", "IsInside"):
        if int(row.get(flag, 0)):
            return False

    x_min, x_max = float(row["XMin"]), float(row["XMax"])
    y_min, y_max = float(row["YMin"]), float(row["YMax"])
    w = x_max - x_min
    h = y_max - y_min

    if w * h < MIN_FACE_AREA_RATIO:
        return False

    aspect = w / h if h > 0 else 999.0
    return MIN_FACE_ASPECT <= aspect <= MAX_FACE_ASPECT


def prefilter_image_ids(
    permissive_ids: Set[str],
    split: str,
    cache_dir: Path,
    refresh: bool = False,
) -> Set[str]:
    """アノテーション CSV だけで画像 DL 前に ID を絞り込む。

    画像本体のダウンロードが始まる前に全フィルタを適用するため、
    不要な画像転送が発生しない。
    """
    classes_csv = _resolve_classes_csv(split, cache_dir, refresh)
    annotations_csv = _resolve_annotations_csv(split, cache_dir, refresh)

    name_to_id, _ = _load_label_maps(classes_csv)

    required_ids = frozenset(name_to_id[c] for c in REQUIRED_CLASSES if c in name_to_id)
    excluded_ids = frozenset(name_to_id[c] for c in EXCLUDED_CLASSES if c in name_to_id)
    face_label_id = name_to_id.get(FACE_CLASS)

    missing_required = REQUIRED_CLASSES - set(name_to_id)
    if missing_required:
        raise ValueError(f"Required classes not in Open Images classes.csv: {missing_required}")

    missing_excluded = EXCLUDED_CLASSES - set(name_to_id)
    if missing_excluded:
        logger.warning("EXCLUDED_CLASSES に不明クラスあり（スキップ）: %s", missing_excluded)

    image_labels: Dict[str, Set[str]] = defaultdict(set)
    image_faces: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    logger.info("アノテーション CSV を解析中: %s", annotations_csv)
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ImageID"]
            if img_id not in permissive_ids:
                continue
            label = row["LabelName"]
            image_labels[img_id].add(label)
            if label == face_label_id:
                image_faces[img_id].append(row)

    valid_ids: Set[str] = set()
    for img_id, labels in image_labels.items():
        if not required_ids.issubset(labels):
            continue
        if labels & excluded_ids:
            continue
        valid_faces = [r for r in image_faces[img_id] if _valid_face_row(r)]
        if not valid_faces or len(valid_faces) > MAX_FACES:
            continue
        valid_ids.add(img_id)

    logger.info("事前フィルタ結果: %d / %d 枚", len(valid_ids), len(permissive_ids))
    return valid_ids


# ---------------------------------------------------------------------------
# fiftyone ダウンロード
# ---------------------------------------------------------------------------

def download_face_images(
    filtered_ids: Set[str],
    split: str,
    output_dir: Path,
    max_images: int,
) -> Dict[str, Path]:
    """事前フィルタ済みの ID のみ画像ダウンロードする。

    classes フィルタは使わない（事前フィルタ済みのため不要）。
    """
    try:
        import fiftyone.zoo as foz
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError("fiftyone が必要です: pip install fiftyone") from exc

    target_ids = sorted(filtered_ids)[:max_images]
    if not target_ids:
        logger.warning("ダウンロード対象 ID がありません")
        return {}

    logger.info("fiftyone で %d 枚をダウンロード中...", len(target_ids))

    dataset_name = f"open_images_faces_{split}"

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        image_ids=target_ids,
        max_samples=len(target_ids),
        dataset_name=dataset_name,
        overwrite=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}

    for sample in dataset:
        src = Path(sample.filepath)
        if not src.exists():
            continue
        image_id = src.stem
        dst = output_dir / src.name
        if not dst.exists():
            shutil.move(str(src), str(dst))
        else:
            src.unlink(missing_ok=True)
        downloaded[image_id] = dst

    fo.delete_dataset(dataset_name, verbose=False)
    logger.info("保存枚数: %d", len(downloaded))
    return downloaded


# ---------------------------------------------------------------------------
# 帰属情報
# ---------------------------------------------------------------------------

def save_attribution(
    metas: Dict[str, ImageMeta],
    downloaded: Dict[str, Path],
    output_path: Path,
) -> None:
    rows = [metas[iid] for iid in downloaded if iid in metas]
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_id", "license_url", "original_url", "author", "title"])
        for m in rows:
            writer.writerow([m.image_id, m.license_url, m.original_url, m.author, m.title])


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def run(
    split: str,
    output_dir: Path,
    max_images: int,
    cache_dir: Path,
    refresh_cache: bool,
    attribution_path: Path,
    shard: int | None = None,
    shard_count: int | None = None,
) -> None:
    if (shard is None) != (shard_count is None):
        raise ValueError("--shard と --shard-count は両方指定してください")
    if shard is not None and (shard < 0 or shard >= shard_count):
        raise ValueError(f"--shard は 0 以上 {shard_count - 1} 以下である必要があります")

    metas = fetch_permissive_metadata(split, cache_dir, refresh=refresh_cache)

    ids_cache_path = _filtered_ids_cache_path(split, cache_dir)

    if ids_cache_path.exists() and not refresh_cache:
        logger.info("フィルタ済みIDキャッシュを使用: %s", ids_cache_path)
        filtered_ids = _load_filtered_ids(ids_cache_path)
    else:
        filtered_ids = prefilter_image_ids(
            permissive_ids=set(metas.keys()),
            split=split,
            cache_dir=cache_dir,
            refresh=refresh_cache,
        )
        _save_filtered_ids(filtered_ids, ids_cache_path)
        logger.info("フィルタ済みIDキャッシュを保存: %s (%d 件)", ids_cache_path, len(filtered_ids))

    sorted_ids = sorted(filtered_ids)

    if shard is not None:
        chunk_size = (len(sorted_ids) + shard_count - 1) // shard_count
        start = shard * chunk_size
        end = min(start + chunk_size, len(sorted_ids))
        target_ids = set(sorted_ids[start:end])
        logger.info(
            "シャード %d/%d: 全 %d 件中 %d-%d 件目 (%d 件) をダウンロード",
            shard + 1, shard_count, len(sorted_ids), start + 1, end, len(target_ids),
        )
    else:
        target_ids = filtered_ids

    downloaded = download_face_images(
        filtered_ids=target_ids,
        split=split,
        output_dir=output_dir,
        max_images=max_images,
    )

    save_attribution(metas, downloaded, attribution_path)
    logger.info("完了: %d 枚", len(downloaded))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Open Images V7 顔画像取得",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--output", type=Path, default="data/images")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--attribution-csv", type=Path, default=ATTRIBUTION_CSV)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--shard", type=int, default=None,
        help="シャード番号 (0始まり, --shard-count と併用)",
    )
    parser.add_argument(
        "--shard-count", type=int, default=None,
        help="総シャード数 (--shard と併用)",
    )

    args = parser.parse_args()

    run(
        split=args.split,
        output_dir=args.output,
        max_images=args.max_images,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        attribution_path=args.attribution_csv,
        shard=args.shard,
        shard_count=args.shard_count,
    )


if __name__ == "__main__":
    main()

"""
Open Images V7 から商用利用・再配布可能な顔画像を取得するスクリプト。

ライセンスフィルタ:
    CC0 / CC BY 2.0 のみ取得（MIT ライセンスで再配布可能）
    CC BY-SA / CC BY-NC / CC BY-ND は除外

使い方:
    # validation split（~41K 枚の中から顔画像を取得、高速）
    python scripts/download_open_images.py --output data/images/

    # train split からより多く取得
    python scripts/download_open_images.py --split train --max-images 2000 --output data/images/

    # キャッシュをクリアして再取得
    python scripts/download_open_images.py --refresh-cache --output data/images/

取得後の流れ:
    python annotator/preprocessor.py --images data/images/ --output data/samples.json
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# 許可するライセンス（正規化済み完全一致 URL）
# startswith ではなく完全一致で判定し、管轄移転版（/ca/, /de/ 等）の混入を防ぐ
PERMISSIVE_LICENSE_URLS: frozenset[str] = frozenset({
    "https://creativecommons.org/publicdomain/zero/1.0",  # CC0
    "https://creativecommons.org/licenses/by/2.0",        # CC BY 2.0
})

# Open Images V7 画像メタデータ URL（ImageID + License + 帰属情報）
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

# fiftyone で使用する顔クラス名
FACE_CLASS = "Human face"

# これらのラベルを含む画像はダウンロード対象から除外（子ども）
# Open Images V7 の 601 クラスに "Baby" は存在しないため含まない
CHILD_CLASSES: frozenset[str] = frozenset({
    "Boy",
    "Girl",
})

# キャッシュディレクトリ（大きなファイルの再ダウンロードを防ぐ）
DEFAULT_CACHE_DIR = Path("data/.open_images_cache")

# 帰属情報 CSV の保存先
ATTRIBUTION_CSV = Path("data/open_images_attribution.csv")


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

class ImageMeta:
    """1 枚の画像に紐づくライセンス・帰属情報。"""

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
# Step 1: 許可ライセンス画像メタデータの取得
# ---------------------------------------------------------------------------

def _normalize_license(url: str) -> str:
    """末尾スラッシュを除去して正規化する。"""
    return url.strip().rstrip("/")


def _is_permissive(license_url: str) -> bool:
    """完全一致でライセンスを判定する（管轄移転版 /ca/, /de/ 等を除外）。"""
    return _normalize_license(license_url) in PERMISSIVE_LICENSE_URLS


def _safe_decode(line: bytes) -> Optional[str]:
    """バイト列を UTF-8 でデコードし、失敗時は Latin-1 にフォールバックする。"""
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        return line.decode("latin-1", errors="replace")


def _stream_csv_rows(url: str) -> Iterator[Dict[str, str]]:
    """URL から CSV をストリーミングで行ごとに返す。空行・デコード不能行はスキップする。"""
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        lines_iter = (
            decoded
            for raw in resp.iter_lines()
            if raw  # 空行スキップ
            for decoded in (_safe_decode(raw),)
            if decoded is not None
        )
        reader = csv.DictReader(lines_iter)
        yield from reader


def fetch_permissive_metadata(
    split: str,
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, ImageMeta]:
    """
    指定 split のメタデータを取得し、許可ライセンス画像の辞書を返す。

    Args:
        split: "train" / "validation" / "test"
        cache_dir: キャッシュ保存ディレクトリ
        refresh: True の場合キャッシュを無視して再取得

    Returns:
        {ImageID: ImageMeta} の辞書（許可ライセンスのみ）
    """
    if split not in METADATA_URLS:
        raise ValueError(
            f"不正な split: {split!r}。"
            f"有効な値: {list(METADATA_URLS)}"
        )

    cache_file = cache_dir / f"{split}_permissive_meta.csv"

    if cache_file.exists() and not refresh:
        logger.info("キャッシュからメタデータを読み込みます: %s", cache_file)
        return _load_meta_cache(cache_file)

    url = METADATA_URLS[split]
    logger.info("メタデータをストリーミング取得中 (split=%s): %s", split, url)

    metas: Dict[str, ImageMeta] = {}
    total = 0
    skipped = 0

    for row in _stream_csv_rows(url):
        total += 1
        try:
            license_url = row.get("License") or ""
            if not _is_permissive(license_url):
                continue

            # DictReader は列数不一致時に None を返すため明示的にチェック
            image_id_raw = row.get("ImageID")
            if image_id_raw is None:
                skipped += 1
                continue
            image_id = image_id_raw.strip()
            if not image_id:
                skipped += 1
                continue

            metas[image_id] = ImageMeta(
                image_id=image_id,
                license_url=license_url.strip(),
                original_url=row.get("OriginalURL") or "",
                author=row.get("Author") or "",
                title=row.get("Title") or "",
            )
        except Exception as exc:
            skipped += 1
            logger.debug("行をスキップ (row=%d): %s", total, exc)
            continue

        if total % 100_000 == 0:
            logger.info(
                "  %d 件処理済み（許可: %d 件, スキップ: %d 件）...",
                total, len(metas), skipped,
            )

    logger.info(
        "完了: %d 件中 %d 件が許可ライセンス (%.1f%%)",
        total, len(metas),
        100.0 * len(metas) / max(total, 1),
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_meta_cache(metas, cache_file)
    logger.info("キャッシュ保存: %s", cache_file)

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
    logger.info("  %d 件", len(metas))
    return metas


# ---------------------------------------------------------------------------
# Step 2: fiftyone で顔画像をダウンロード
# ---------------------------------------------------------------------------

def download_face_images(
    permissive_ids: Set[str],
    split: str,
    output_dir: Path,
    max_images: int,
) -> Dict[str, Path]:
    """
    fiftyone を使って顔画像をダウンロードし、output_dir にコピーする。

    Open Images のファイル名は常に ImageID.jpg 形式（fiftyone zoo dataset の仕様）。
    stem から ImageID を取得し、permissive_ids との一致を確認して返す。

    Args:
        permissive_ids: 許可ライセンスの ImageID セット
        split: "train" / "validation" / "test"
        output_dir: 画像の保存先ディレクトリ
        max_images: 最大取得枚数

    Returns:
        {ImageID: ローカルパス} の辞書
    """
    try:
        import fiftyone.zoo as foz
        import fiftyone as fo
        from fiftyone import ViewField as F
    except ImportError as exc:
        raise ImportError(
            "fiftyone が見つかりません。`pip install fiftyone` でインストールしてください。"
        ) from exc

    logger.info(
        "fiftyone で顔画像をダウンロード中 (split=%s, max=%d, 許可ID=%d 件)...",
        split, max_images, len(permissive_ids),
    )

    # 許可 ID を候補として渡す（十分な候補を渡して max_images を満たす）
    sampled_ids = sorted(permissive_ids)[:max_images * 10]

    dataset_name = f"open_images_faces_{split}"
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=[FACE_CLASS],
        image_ids=sampled_ids,
        max_samples=max_images,
        dataset_name=dataset_name,
        overwrite=True,
    )

    # 子ども・赤ん坊ラベルを含む画像を除外
    child_ids = dataset.match_labels(
        filter=F("label").is_in(list(CHILD_CLASSES)),
        fields="detections",
    ).values("id")
    if child_ids:
        logger.info(
            "子ども・赤ん坊ラベルを含む画像を除外: %d 件 (%d → %d)",
            len(child_ids), len(dataset), len(dataset) - len(child_ids),
        )
    clean_view = dataset.exclude(child_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    # {ImageID: ローカルパス} — 帰属情報の正確な照合のために ImageID をキーにする
    downloaded: Dict[str, Path] = {}

    for sample in clean_view:
        src = Path(sample.filepath)
        if not src.exists():
            logger.warning("ファイルが見つかりません: %s", src)
            continue

        # fiftyone の Open Images zoo dataset はファイル名が必ず <ImageID>.<ext> 形式
        image_id = src.stem
        if image_id not in permissive_ids:
            # 万が一 ID が一致しない場合はコピーしない
            logger.debug("許可IDリストに存在しないサンプルをスキップ: %s", image_id)
            continue

        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        downloaded[image_id] = dst

    # fiftyone のデータセットを削除（ローカルキャッシュは保持）
    fo.delete_dataset(dataset_name, verbose=False)

    logger.info("%d 枚を %s にコピーしました。", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# Step 3: 帰属情報の保存
# ---------------------------------------------------------------------------

def save_attribution(
    metas: Dict[str, ImageMeta],
    downloaded: Dict[str, Path],
    output_path: Path,
) -> None:
    """
    ダウンロードした画像の帰属情報を CSV に保存する。

    downloaded のキーは ImageID（download_face_images が確定したもの）であり、
    ファイル名のステムから推測しないため確実に一致する。

    CC BY 2.0 画像を再配布する際に README 等で参照する。
    """
    rows = [metas[iid] for iid in downloaded if iid in metas]

    if not rows:
        logger.warning(
            "帰属情報が見つかりませんでした。"
            "ダウンロード済み %d 件のうち metas に一致する ImageID がありません。",
            len(downloaded),
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_id", "license_url", "original_url", "author", "title"])
        for m in rows:
            writer.writerow([m.image_id, m.license_url, m.original_url, m.author, m.title])

    logger.info("帰属情報を保存しました: %s (%d 件)", output_path, len(rows))


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

def run(
    split: str,
    output_dir: Path,
    max_images: int,
    cache_dir: Path,
    refresh_cache: bool,
    attribution_path: Path,
) -> None:
    """ダウンロードパイプライン全体を実行する。"""
    logger.info("=== Step 1: 許可ライセンスのメタデータ取得 ===")
    metas = fetch_permissive_metadata(split, cache_dir, refresh=refresh_cache)
    permissive_ids = set(metas.keys())
    logger.info("許可ライセンス画像: %d 件", len(permissive_ids))

    logger.info("=== Step 2: 顔画像のダウンロード ===")
    downloaded = download_face_images(
        permissive_ids=permissive_ids,
        split=split,
        output_dir=output_dir,
        max_images=max_images,
    )

    logger.info("=== Step 3: 帰属情報の保存 ===")
    save_attribution(metas, downloaded, attribution_path)

    n_downloaded = len(downloaded)

    logger.info("=== 完了 ===")
    logger.info("  ダウンロード枚数: %d", n_downloaded)
    logger.info("  保存先: %s", output_dir)
    logger.info("  帰属情報: %s", attribution_path)
    logger.info("")
    logger.info("次のステップ:")
    logger.info(
        "  python annotator/preprocessor.py --images %s --output data/samples.json",
        output_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Open Images V7 から CC0 / CC BY 2.0 ライセンスの顔画像を取得します。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="使用する split（default: validation）",
    )
    parser.add_argument(
        "--max-images",
        default=500,
        type=int,
        metavar="N",
        help="最大取得枚数（default: 500）",
    )
    parser.add_argument(
        "--output",
        default="data/images",
        type=Path,
        help="画像の保存先ディレクトリ（default: data/images）",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        type=Path,
        help=f"メタデータキャッシュ先（default: {DEFAULT_CACHE_DIR}）",
    )
    parser.add_argument(
        "--attribution-csv",
        default=str(ATTRIBUTION_CSV),
        type=Path,
        help=f"帰属情報 CSV の保存先（default: {ATTRIBUTION_CSV}）",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="メタデータキャッシュを無視して再取得する",
    )

    args = parser.parse_args()

    run(
        split=args.split,
        output_dir=args.output,
        max_images=args.max_images,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        attribution_path=args.attribution_csv,
    )


if __name__ == "__main__":
    main()

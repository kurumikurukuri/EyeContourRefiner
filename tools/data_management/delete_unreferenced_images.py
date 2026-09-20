"""全実験の manifest で一度も参照されていない画像ファイルを削除するスクリプト。

data/images/ の画像のうち、experiments/EXP*/manifests/*.csv のどれからも
一度も参照されていない（過去に参照されたが現在は外れた画像は保持対象）ものを
削除する。data/samples.db の行は変更しない（ファイルのみ削除）。

使い方:
    # 確認のみ（削除しない）
    python tools/data_management/delete_unreferenced_images.py --dry-run

    # 実際に削除する
    python tools/data_management/delete_unreferenced_images.py --execute
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = ROOT / "data" / "images"
MANIFEST_GLOB = "experiments/*/manifests/*.csv"


def load_referenced_image_paths(root: Path) -> tuple[set[str], int]:
    """manifest で参照されている画像パス集合と、読んだ manifest 数を返す。"""
    referenced: set[str] = set()
    manifest_paths = sorted(root.glob(MANIFEST_GLOB))
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                continue
            for row in reader:
                referenced.add(row["image_path"].replace("\\", "/"))
    return referenced, len(manifest_paths)


def main(execute: bool) -> None:
    if not IMAGES_DIR.exists():
        print(f"images ディレクトリが見つかりません: {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    referenced, manifest_count = load_referenced_image_paths(ROOT)
    if manifest_count == 0 or not referenced:
        print(
            f"参照元 manifest が見つかりません（manifest {manifest_count} 件、参照画像 {len(referenced)} 件）。"
            "削除を実行せず中断します。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"参照元 manifest 数: {manifest_count}")
    print(f"全 manifest で一度でも参照された画像数: {len(referenced)}")

    image_files = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    print(f"data/images/ 内の画像数: {len(image_files)}")

    to_delete = [
        img for img in image_files
        if f"data/images/{img.name}" not in referenced
    ]
    print(f"削除対象（一度も参照されていない画像）: {len(to_delete)}")
    print(f"保持対象（一度でも参照された画像）: {len(image_files) - len(to_delete)}")

    if not execute:
        print("\n[dry-run] 削除は実行されていません。--execute を付けると実際に削除します。")
        return

    deleted_count = 0
    for img in to_delete:
        img.unlink()
        deleted_count += 1

    print(f"\n完了: {deleted_count} 件の画像ファイルを削除しました。")
    print("data/samples.db の行は変更していません。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="どの manifest からも一度も参照されていない画像ファイルを削除する"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="削除対象を表示するだけ")
    group.add_argument("--execute", action="store_true", help="実際に削除を実行する")
    args = parser.parse_args()
    main(execute=args.execute)

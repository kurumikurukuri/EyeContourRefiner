"""vlm_skip_hints.db の判定結果を samples.db の skip_flag に反映するスクリプト。

適用ロジック:
  vlm_only (デフォルト): vlm_skip=1 AND vlm_confidence >= min_confidence のみ適用
  human_only           : human_skip=1 のみ適用

使い方:
    python tools/annotation_support/VLM_skip/vlm_skip_apply_hints.py --dry-run
    python tools/annotation_support/VLM_skip/vlm_skip_apply_hints.py --min-confidence 0.7
    python tools/annotation_support/VLM_skip/vlm_skip_apply_hints.py --use human_only --yes
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tools.annotation_support.VLM_skip.vlm_skip_hints_db import (
    DEFAULT_HINTS_DB,
    count_stats,
    get_skip_candidates,
)


_CHUNK = 500  # SQLite の変数数上限対策


def _chunked(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _update_skip_flags(db_path: Path, image_paths: List[str]) -> int:
    """samples.db の指定 image_path の skip_flag を 1 に更新する。更新件数を返す。"""
    if not image_paths:
        return 0
    total = 0
    with sqlite3.connect(str(db_path)) as conn:
        for chunk in _chunked(image_paths, _CHUNK):
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"UPDATE samples SET skip_flag = 1 WHERE image_path IN ({placeholders})",
                chunk,
            )
            total += cur.rowcount
    return total


def _get_already_annotated(db_path: Path, image_paths: List[str]) -> List[str]:
    """指定 image_path のうち、1件以上アノテーション済み (t IS NOT NULL) のものを返す。"""
    if not image_paths:
        return []
    result: List[str] = []
    with sqlite3.connect(str(db_path)) as conn:
        for chunk in _chunked(image_paths, _CHUNK):
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""
                SELECT DISTINCT image_path FROM samples
                WHERE image_path IN ({placeholders})
                  AND t IS NOT NULL
                """,
                chunk,
            ).fetchall()
            result.extend(r[0] for r in rows)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vlm_skip_hints.db の判定を samples.db の skip_flag に反映する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="data/samples.db", help="samples.db パス")
    parser.add_argument("--hints-db", default=str(DEFAULT_HINTS_DB), help="vlm_skip_hints.db パス")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="VLM 判定を適用する信頼度の下限",
    )
    parser.add_argument(
        "--use",
        choices=["vlm_only", "human_only"],
        default="vlm_only",
        help="適用ロジック (vlm_only=VLMのみ, human_only=人間のみ)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="件数確認のみ（DB を更新しない）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="確認プロンプトをスキップして即座に適用",
    )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = _parse_args()

    db_path = Path(args.db)
    hints_db_path = Path(args.hints_db)

    if not db_path.exists():
        print(f"ERROR: samples.db が見つかりません: {db_path}", file=sys.stderr)
        sys.exit(1)

    if not hints_db_path.exists():
        print(f"ERROR: vlm_skip_hints.db が見つかりません: {hints_db_path}", file=sys.stderr)
        sys.exit(1)

    stats = count_stats(hints_db_path)
    print(f"[hints DB] 処理済み: {stats['vlm_processed']} / VLM skip: {stats['vlm_skip']} / 人間確認済み: {stats['human_reviewed']}")

    candidates = get_skip_candidates(
        hints_db_path=hints_db_path,
        min_confidence=args.min_confidence,
        use=args.use,
    )
    target_paths = [c["image_path"] for c in candidates]

    if not target_paths:
        print("適用対象が 0 件です。")
        return

    # アノテーション済み画像の警告
    annotated_overlap = _get_already_annotated(db_path, target_paths)
    if annotated_overlap:
        print(f"\n警告: {len(annotated_overlap)} 枚はアノテーション済みですが skip_flag=1 に変更されます:")
        for p in annotated_overlap[:10]:
            print(f"  {p}")
        if len(annotated_overlap) > 10:
            print(f"  ... 他 {len(annotated_overlap) - 10} 枚")

    print(f"\n適用対象: {len(target_paths)} 枚 (--use={args.use}, min_confidence={args.min_confidence})")
    print(f"  samples.db の skip_flag を 1 に更新します: {db_path}")

    if args.dry_run:
        print("\n[dry-run] DB は更新しません。")
        return

    if not args.yes:
        ans = input("\n続行しますか？ [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("キャンセルしました。")
            return

    updated_rows = _update_skip_flags(db_path, target_paths)
    print(f"\n完了: {len(target_paths)} 画像 / {updated_rows} サンプル行を更新しました。")


if __name__ == "__main__":
    main()

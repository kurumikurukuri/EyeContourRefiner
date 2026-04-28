"""data/samples.json を data/samples.db (SQLite) へ移行するスクリプト。

使い方:
    python tools/migrate_json_to_sqlite.py
    python tools/migrate_json_to_sqlite.py --json data/samples.json --db data/samples.db
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotator import db as _db

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 10_000  # INSERT のバッチサイズ


def migrate(json_path: Path, db_path: Path) -> int:
    """JSON → SQLite 移行。挿入件数を返す。"""
    if not json_path.exists():
        logger.error("samples.json が見つかりません: %s", json_path)
        sys.exit(1)

    logger.info("samples.json を読み込んでいます: %s", json_path)
    samples: list[dict] = json.loads(json_path.read_text(encoding="utf-8"))
    logger.info("  %d 件のサンプルを読み込みました", len(samples))

    _db.init_db(db_path)
    logger.info("DB を初期化しました: %s", db_path)

    total = 0
    for i in range(0, len(samples), _CHUNK_SIZE):
        chunk = samples[i : i + _CHUNK_SIZE]
        _db.upsert_samples(chunk, db_path)
        total += len(chunk)
        logger.info("  %d / %d 件挿入済み...", total, len(samples))

    logger.info("移行完了: %d 件 → %s", total, db_path)
    return total


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="samples.json を SQLite DB へ移行する"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("data/samples.json"),
        help="入力 JSON ファイルパス (default: data/samples.json)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/samples.db"),
        help="出力 SQLite DB パス (default: data/samples.db)",
    )
    args = parser.parse_args()

    n = migrate(args.json, args.db)
    print(f"[OK] {n} 件を {args.db} へ移行しました。")


if __name__ == "__main__":
    main()

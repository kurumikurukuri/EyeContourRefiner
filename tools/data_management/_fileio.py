"""ファイルの原子的な書き込み（一時ファイル → rename）の共通処理。

キャッシュや取得画像を、途中で止まっても壊れた状態で残さないために使う。一時ファイル名には
プロセス ID を含め、同じファイルを同時に書く別プロセス（`--shard` の並列実行や、fetch と restore の
同時実行）の一時ファイルと衝突させない。
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional


def part_path(dest: Path) -> Path:
    """`dest` の一時ファイルのパス（同じディレクトリ・プロセス ID 入り）。"""
    return dest.with_name(f"{dest.name}.{os.getpid()}.part")


@contextmanager
def atomic_replace(dest: Path) -> Iterator[Path]:
    """一時ファイルのパスを渡し、正常終了したときだけ `dest` に rename で置き換える。

    例外時は一時ファイルを消し、既存の `dest` には触れない。

    Raises:
        FileNotFoundError: ブロック内で一時ファイルが作られなかった（正常終了時）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dest)
    try:
        yield part
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def expected_length(headers: Mapping[str, str]) -> Optional[int]:
    """HTTP レスポンスの Content-Length（取得したバイト数の検証用）。

    圧縮転送（Content-Encoding）では展開後のサイズと一致しないため None（検証しない）。
    urllib3 1.x は本体が途中で切れても例外にしないので、呼び出し側で長さを照合する。
    """
    encoding = (headers.get("Content-Encoding") or "").lower()
    if encoding not in ("", "identity"):
        return None
    try:
        return int(headers["Content-Length"])
    except (KeyError, ValueError):
        return None

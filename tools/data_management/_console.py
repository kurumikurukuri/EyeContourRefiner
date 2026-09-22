"""CLI の標準出力・標準エラーの文字コード対策。

Windows の日本語環境（cp932）は「〜」など一部の文字を表現できず、エラーメッセージの出力自体が
`UnicodeEncodeError` で落ちることがある。文字コードは変えず（従来のコンソールで文字化けさせない）、
表現できない文字だけを置換して出力を続ける。
"""
from __future__ import annotations

import sys


def make_console_safe() -> None:
    """stdout / stderr を、エンコードできない文字を `?` に置換する設定にする。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")

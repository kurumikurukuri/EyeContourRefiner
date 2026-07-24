"""新規実験用 summary テンプレートコピーツール。

EXPn_summary_template.md を実験ディレクトリにコピーし、プレースホルダを置換する。

使用例:
  python scripts/init_exp_summary.py --exp EXP028 --title "新しい実験"
  python scripts/init_exp_summary.py --exp EXP028 --title "loss sweep v2" --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ROOT, find_exp_dir

TEMPLATE_PATH = ROOT / "EXPn_summary_template.md"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="新規実験用 summary テンプレートコピー"
    )
    ap.add_argument("--exp", type=str, required=True, help="実験番号（例: 028）")
    ap.add_argument("--title", type=str, default="", help="実験タイトル（プレースホルダ置換用）")
    ap.add_argument("--force", action="store_true", help="既存ファイルを強制上書き")
    args = ap.parse_args()

    exp_dir = find_exp_dir(args.exp)
    exp_num = exp_dir.name[:6]  # "EXP028"
    out_path = exp_dir / f"{exp_num}_summary.md"

    if out_path.exists() and not args.force:
        print(f"[ERROR] 既に存在します: {out_path}")
        print(f"  上書きする場合は --force を指定してください")
        return 1

    if not TEMPLATE_PATH.exists():
        print(f"[ERROR] テンプレートが見つかりません: {TEMPLATE_PATH}")
        return 1

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # プレースホルダ置換
    template = template.replace("EXP{n}", exp_num)
    if args.title:
        # 1行目のタイトル行のみを置換（グローバル replace を避けて安全に）
        lines = template.split("\n")
        if lines and lines[0].startswith(f"# {exp_num}:"):
            lines[0] = f"# {exp_num}: {args.title}"
            template = "\n".join(lines)

    out_path.write_text(template, encoding="utf-8")
    print(f"作成: {out_path}")
    print(f"  exp:   {exp_num}")
    if args.title:
        print(f"  title: {args.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

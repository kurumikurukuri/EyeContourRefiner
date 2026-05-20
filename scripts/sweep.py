"""sweep.py: 実験番号を受け取り、EXP側のスクリプトを実行するディスパッチャー。

使い方:
    python scripts/sweep.py --exp 001
    python scripts/sweep.py --exp 004 --configs A_bn B_bn
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def find_exp_dir(exp_num: str) -> Path:
    """EXP番号からディレクトリを探す（例: "001" → experiments/EXP001_xxx/）"""
    pattern = f"EXP{exp_num}_*"
    matches = list((ROOT / "experiments").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"実験ディレクトリが見つかりません: experiments/{pattern}")
    if len(matches) > 1:
        raise ValueError(
            f"複数の実験ディレクトリが見つかりました: {[str(m.name) for m in matches]}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="実験ディスパッチャー: experiments/EXP{n}_*/scr/train_exp{n}.py を実行する"
    )
    parser.add_argument("--exp", required=True, help="実験番号（例: 001）")
    parser.add_argument("--configs", nargs="+", help="実行する設定名（省略時は全実験）")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader の num_workers（省略時は train_exp{n}.py のデフォルト）")
    args = parser.parse_args()

    exp_dir = find_exp_dir(args.exp)
    script = exp_dir / "scr" / f"train_exp{args.exp}.py"

    if not script.exists():
        raise FileNotFoundError(f"実験スクリプトが見つかりません: {script}")

    cmd = [sys.executable, str(script)]
    if args.configs:
        cmd += ["--configs"] + args.configs
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]

    print(f"[sweep] EXP{args.exp}: {script.relative_to(ROOT)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

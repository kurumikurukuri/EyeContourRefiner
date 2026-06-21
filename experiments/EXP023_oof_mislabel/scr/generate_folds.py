"""EXP023: 固定 5-Fold manifest (fold1.csv〜fold5.csv) を生成する。

一度生成すれば、以降の run_oof.py 実行は常に同じ分割を使い回す
（GroupKFold(shuffle=True, random_state=seed) はシード固定で決定的）。

使い方:
    python experiments/EXP023_oof_mislabel/scr/generate_folds.py
    python .../generate_folds.py --k 5 --seed 42 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"

DEFAULT_MANIFEST = MANIFEST_DIR / "full_manifest.csv"

sys.path.insert(0, str(ROOT))
from eyeliner.training.oof_kfold import split_fixed_folds  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="EXP023 固定 Fold 生成")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=MANIFEST_DIR)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="既存 fold ファイルを再生成する")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] manifest 不在: {args.manifest}")
        return 1

    existing = sorted(args.out_dir.glob("fold*.csv"))
    if existing and not args.force:
        print(f"[skip] 既存の fold ファイルあり: {[p.name for p in existing]} (--force で再生成)")
        return 0

    fold_paths = split_fixed_folds(args.manifest, args.out_dir, k=args.k, seed=args.seed)
    print(f"生成: {len(fold_paths)} fold ファイル → {args.out_dir}")
    for p in fold_paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

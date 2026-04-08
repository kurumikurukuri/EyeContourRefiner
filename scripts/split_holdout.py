"""Train/Test 分割スクリプト（一度だけ実行）。

data/manifest.csv（god ファイル）から画像単位で分割し、
data/manifest_train.csv と data/manifest_test.csv を生成する。

重要:
  - data/manifest.csv は変更しない（god ファイル）
  - manifest_test.csv は訓練に使用してはならない
  - holdout_mae の計算のみに使用する（ハイパーパラメータ選定に使わないこと）
  - 既存ファイルは上書きされる

使用例:
  python scripts/split_holdout.py
  python scripts/split_holdout.py --manifest data/manifest.csv --ratio 0.1 --seed 42
  python scripts/split_holdout.py --exp 021 --method stability --ratio 0.20
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def split_holdout(
    manifest_path: Path,
    train_path: Path,
    test_path: Path,
    ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """manifest.csv（god）を画像単位で分割し、train/test CSV を生成する。

    manifest_path は変更しない。

    Args:
        manifest_path: manifest.csv のパス（god ファイル、変更しない）
        train_path:    訓練用 CSV の出力パス
        test_path:     テスト用 CSV の出力パス
        ratio:         テストの割合（デフォルト 10%）
        seed:          乱数シード（固定）

    Returns:
        (n_train, n_test) — 分割後のサンプル数
    """
    # manifest.csv（god）の読み込み
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError("manifest.csv が空です")

    # 画像単位で分割（同一画像の複数点がリークしないよう）
    unique_images = sorted({row["image_path"] for row in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_images)

    n_test_images = max(1, int(len(unique_images) * ratio))
    test_images = set(unique_images[:n_test_images])
    train_images = set(unique_images[n_test_images:])

    train_rows = [r for r in rows if r["image_path"] in train_images]
    test_rows = [r for r in rows if r["image_path"] in test_images]

    # manifest_train.csv の保存
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    # manifest_test.csv の保存
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    return len(train_rows), len(test_rows)


def split_holdout_by_stability(
    manifest_path: Path,
    train_path: Path,
    test_path: Path,
    ratio: float = 0.2,
) -> tuple[int, int]:
    """manifest.csv から t 値のばらつき（std）が小さい順に ratio の画像をテストに割り当てる。

    アノテーションが安定している（ばらつきが少ない）画像をテストセットにすることで、
    テスト評価の再現性を高める。std が NaN（1点のみの画像）は 0.0 として扱う。

    Args:
        manifest_path: manifest.csv のパス（god ファイル、変更しない）
        train_path:    訓練用 CSV の出力パス
        test_path:     テスト用 CSV の出力パス
        ratio:         テストの割合（デフォルト 20%）

    Returns:
        (n_train, n_test) — 分割後のサンプル数
    """
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError("manifest.csv が空です")

    df = pd.DataFrame(rows)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])

    stats = (
        df.groupby("image_path")["t"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats_sorted = stats.sort_values("std")

    n_test_images = max(1, int(len(stats_sorted) * ratio))
    test_images = set(stats_sorted.head(n_test_images)["image_path"].tolist())
    train_images = set(stats_sorted.iloc[n_test_images:]["image_path"].tolist())

    train_rows = [r for r in rows if r["image_path"] in train_images]
    test_rows = [r for r in rows if r["image_path"] in test_images]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    return len(train_rows), len(test_rows)


def find_exp_dir(exp_num: str) -> Path:
    """EXP番号からディレクトリを探す（例: "001" → experiments/EXP001_xxx/）"""
    root = Path(__file__).parent.parent
    pattern = f"EXP{exp_num}_*"
    matches = list((root / "experiments").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"実験ディレクトリが見つかりません: experiments/{pattern}")
    if len(matches) > 1:
        raise ValueError(
            f"複数の実験ディレクトリが見つかりました: {[str(m.name) for m in matches]}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="manifest.csv（god）から train/test を生成する"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="実験番号（例: 001）。指定時は experiments/EXP{n}_xxx/manifests/ に出力",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="god manifest CSV のパス（変更しない）（デフォルト: data/manifest.csv）",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="訓練用 CSV の出力パス（--exp 未指定時のデフォルト: data/manifest_train.csv）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="テスト用 CSV の出力パス（--exp 未指定時のデフォルト: data/manifest_test.csv）",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="テストの割合（デフォルト: 0.2 = 20%%）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42、random メソッドのみ使用）",
    )
    parser.add_argument(
        "--method",
        choices=["random", "stability"],
        default="random",
        help=(
            "分割方法: random=ランダム（デフォルト）, "
            "stability=t値のstdが小さい順に ratio をテストへ"
        ),
    )
    args = parser.parse_args()

    # --exp が指定された場合は実験ディレクトリの manifests/ に出力
    if args.exp is not None:
        exp_dir = find_exp_dir(args.exp)
        train_path = exp_dir / "manifests" / "manifest_train.csv"
        test_path  = exp_dir / "manifests" / "manifest_test.csv"
    else:
        train_path = args.train or Path("data/manifest_train.csv")
        test_path  = args.output or Path("data/manifest_test.csv")

    if not args.manifest.exists():
        print(f"[ERROR] manifest が見つかりません: {args.manifest}")
        sys.exit(1)

    print(f"god manifest: {args.manifest}")
    print(f"train_path:   {train_path}")
    print(f"test_path:    {test_path}")
    print(f"method:       {args.method}")
    print(f"ratio:        {args.ratio:.1%}")
    if args.method == "random":
        print(f"seed:         {args.seed}")
    print()

    if args.method == "stability":
        n_train, n_test = split_holdout_by_stability(
            manifest_path=args.manifest,
            train_path=train_path,
            test_path=test_path,
            ratio=args.ratio,
        )
    else:
        n_train, n_test = split_holdout(
            manifest_path=args.manifest,
            train_path=train_path,
            test_path=test_path,
            ratio=args.ratio,
            seed=args.seed,
        )

    print(f"[OK] 分割完了:")
    print(f"  訓練用  : {n_train} サンプル → {train_path}")
    print(f"  テスト用: {n_test} サンプル → {test_path}")
    print()
    print("[注意] manifest_test.csv は以降変更しないでください。")
    print("       holdout_mae の計算のみに使用し、ハイパーパラメータ選定には使わないこと。")


if __name__ == "__main__":
    main()

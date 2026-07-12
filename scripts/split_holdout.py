"""Train/Test 分割スクリプト（一度だけ実行）。

data/samples.db（god ファイル / SQLite）から t IS NOT NULL のサンプルを取得し、
画像単位で train/test に分割し experiments/EXP{n}_xxx/manifests/ に書き出す。

重要:
  - data/samples.db は変更しない（god ファイル）
  - manifest_test.csv は訓練に使用してはならない
  - holdout_mae の計算のみに使用する（ハイパーパラメータ選定に使わないこと）
  - 既存ファイルは上書きされる

使用例:
  python scripts/split_holdout.py --exp 014 --ratio 0.20 --seed 42
  python scripts/split_holdout.py --exp 015 --method stability --ratio 0.20
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotator import db as _db  # noqa: E402

FIELDNAMES = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "nose_x", "nose_y",
]


def _load_annotated_rows(db_path: Path) -> List[Dict[str, Any]]:
    """DB から t IS NOT NULL のサンプルを読み、manifest 形式の dict 列を返す。"""
    annotated = _db.get_annotated_samples(db_path)
    return [
        {
            "idx":         s["idx"],
            "image_path":  s["image_path"].replace("\\", "/"),
            "face_index":  s.get("face_index", 0),
            "part_name":   s["part_name"],
            "sub_key":     s.get("sub_key", ""),
            "point_index": s.get("point_index", 0),
            "px":          s["px"],
            "py":          s["py"],
            "qx":          s["qx"],
            "qy":          s["qy"],
            "t":           s["t"],
            "nose_x":      s.get("nose_x"),
            "nose_y":      s.get("nose_y"),
        }
        for s in annotated
    ]


def _write_manifest(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def split_holdout(
    db_path: Path,
    train_path: Path,
    test_path: Path,
    ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """DB から annotated サンプルを取り、画像単位でランダム分割して train/test を出力する。

    Args:
        db_path:    samples.db のパス（god ファイル、変更しない）
        train_path: 訓練用 CSV の出力パス
        test_path:  テスト用 CSV の出力パス
        ratio:      テストの割合（デフォルト 10%）
        seed:       乱数シード（固定）

    Returns:
        (n_train, n_test) — 分割後のサンプル数
    """
    rows = _load_annotated_rows(db_path)
    if not rows:
        raise ValueError("DB に annotated (t IS NOT NULL) サンプルがありません")

    unique_images = sorted({row["image_path"] for row in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_images)

    n_test_images = max(1, int(len(unique_images) * ratio))
    test_images = set(unique_images[:n_test_images])
    train_images = set(unique_images[n_test_images:])

    train_rows = [r for r in rows if r["image_path"] in train_images]
    test_rows  = [r for r in rows if r["image_path"] in test_images]

    _write_manifest(train_rows, train_path)
    _write_manifest(test_rows, test_path)

    return len(train_rows), len(test_rows)


def split_holdout_by_stability(
    db_path: Path,
    train_path: Path,
    test_path: Path,
    ratio: float = 0.2,
) -> tuple[int, int]:
    """DB の annotated サンプルを t 値の std が小さい順に ratio の画像をテストへ。

    アノテーションが安定している画像をテストセットにすることで、
    テスト評価の再現性を高める。std が NaN（1点のみの画像）は 0.0 として扱う。

    Args:
        db_path:    samples.db のパス（god ファイル、変更しない）
        train_path: 訓練用 CSV の出力パス
        test_path:  テスト用 CSV の出力パス
        ratio:      テストの割合（デフォルト 20%）

    Returns:
        (n_train, n_test) — 分割後のサンプル数
    """
    rows = _load_annotated_rows(db_path)
    if not rows:
        raise ValueError("DB に annotated (t IS NOT NULL) サンプルがありません")

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
    test_images  = set(stats_sorted.head(n_test_images)["image_path"].tolist())
    train_images = set(stats_sorted.iloc[n_test_images:]["image_path"].tolist())

    train_rows = [r for r in rows if r["image_path"] in train_images]
    test_rows  = [r for r in rows if r["image_path"] in test_images]

    _write_manifest(train_rows, train_path)
    _write_manifest(test_rows, test_path)

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
        description="samples.db（god）から train/test を画像単位で生成する"
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="実験番号（例: 014）。出力先は experiments/EXP{n}_xxx/manifests/",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/samples.db"),
        help="samples.db のパス（god ファイル、変更しない）",
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

    if not args.db.exists():
        print(f"[ERROR] samples.db が見つかりません: {args.db}")
        sys.exit(1)

    exp_dir    = find_exp_dir(args.exp)
    train_path = exp_dir / "manifests" / "manifest_train.csv"
    test_path  = exp_dir / "manifests" / "manifest_test.csv"

    print(f"god db:     {args.db}")
    print(f"exp_dir:    {exp_dir}")
    print(f"train_path: {train_path}")
    print(f"test_path:  {test_path}")
    print(f"method:     {args.method}")
    print(f"ratio:      {args.ratio:.1%}")
    if args.method == "random":
        print(f"seed:       {args.seed}")
    print()

    if args.method == "stability":
        n_train, n_test = split_holdout_by_stability(
            db_path=args.db,
            train_path=train_path,
            test_path=test_path,
            ratio=args.ratio,
        )
    else:
        n_train, n_test = split_holdout(
            db_path=args.db,
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

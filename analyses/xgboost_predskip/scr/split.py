"""samples.csv を train/val/test の 3 つの manifest に分割する。

分類用: 全データ（skip_flag=0/1 両方）を使用し unlabeled 分割なし。
同一 image_path のサンプルが異なる split に混在しないよう image_path 単位で分割する。
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


def _normalize_ratios(cfg: SplitConfig) -> tuple[float, float, float]:
    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if total <= 0:
        raise ValueError("train/val/test の合計比が 0 以下です。")
    return (
        cfg.train_ratio / total,
        cfg.val_ratio / total,
        cfg.test_ratio / total,
    )


def split_by_image_path(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Dict[str, pd.DataFrame]:
    """全データを image_path 単位で train/val/test に分割する。

    Returns:
        {"train": df, "val": df, "test": df}
    """
    if "skip_flag" not in df.columns:
        raise ValueError("CSV に skip_flag カラムがありません。")

    unique_images = np.array(sorted(df["image_path"].unique()))
    if len(unique_images) < 3:
        raise ValueError(
            f"画像数が少なすぎます（{len(unique_images)}）。"
            "train/val/test 分割できません。"
        )

    rng = np.random.default_rng(cfg.seed)
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    train_r, val_r, _test_r = _normalize_ratios(cfg)
    n_total = len(shuffled)
    n_train = int(round(n_total * train_r))
    n_val = int(round(n_total * val_r))
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = max(1, n_total - n_train - 1)

    train_imgs = set(shuffled[:n_train])
    val_imgs = set(shuffled[n_train : n_train + n_val])
    test_imgs = set(shuffled[n_train + n_val :])

    def _pick(imgs: set) -> pd.DataFrame:
        return df[df["image_path"].isin(imgs)].copy()

    result = {
        "train": _pick(train_imgs),
        "val": _pick(val_imgs),
        "test": _pick(test_imgs),
    }

    train_set = set(result["train"]["image_path"].unique())
    val_set = set(result["val"]["image_path"].unique())
    test_set = set(result["test"]["image_path"].unique())
    assert train_set.isdisjoint(val_set), "train/val 間で image_path 漏洩"
    assert train_set.isdisjoint(test_set), "train/test 間で image_path 漏洩"
    assert val_set.isdisjoint(test_set), "val/test 間で image_path 漏洩"

    return result


def write_manifests(splits: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = out_dir / f"manifest_{name}.csv"
        df.to_csv(path, index=False)
        logger.info(
            "wrote %s: %d rows, %d images",
            path,
            len(df),
            df["image_path"].nunique(),
        )


def _load_config(config_path: Path) -> SplitConfig:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = cfg["split"]
    return SplitConfig(
        train_ratio=float(s["train_ratio"]),
        val_ratio=float(s["val_ratio"]),
        test_ratio=float(s["test_ratio"]),
        seed=int(s.get("seed", 42)),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="xgboost_predskip の manifest を生成する"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predskip/config/baseline.yaml"),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] config が見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    csv_path = Path(full_cfg["data"]["csv_path"])
    manifests_dir = Path(full_cfg["data"]["manifests_dir"])
    split_cfg = _load_config(args.config)

    if not csv_path.exists():
        print(f"[ERROR] CSV が見つかりません: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("loaded %s: %d rows", csv_path, len(df))

    splits = split_by_image_path(df, split_cfg)
    write_manifests(splits, manifests_dir)
    print("[OK] split 完了")


if __name__ == "__main__":
    main()

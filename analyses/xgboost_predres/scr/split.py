"""EXP017 W0_baseline の予測と特徴量 manifests を結合して train/val/test を生成する。

データフロー:
  xgboost_predskip manifests (特徴量ソース) + EXP017 predictions (目的変数ソース)
    → idx で結合 → 残差 (pred_t_raw - true_t_raw) を付与
    → EXP017 test → manifest_test.csv
    → EXP017 train → image_path 単位シャッフル → manifest_train.csv / manifest_val.csv
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
    val_ratio: float
    seed: int


def _load_feature_manifests(manifests_dir: Path) -> pd.DataFrame:
    """xgboost_predskip manifests を全スプリット結合して特徴量 lookup を作る。"""
    dfs = []
    for split_name in ["train", "val", "test", "unlabeled"]:
        path = manifests_dir / f"manifest_{split_name}.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(f"manifest が見つかりません: {manifests_dir}")
    df = pd.concat(dfs, ignore_index=True)
    logger.info("feature lookup: %d rows from %s", len(df), manifests_dir)
    return df


def _compute_residuals(pred_csv: Path) -> pd.DataFrame:
    """予測 CSV を読み込み残差カラムを追加して返す。"""
    df = pd.read_csv(pred_csv)
    df["residual"] = df["pred_t_raw"] - df["true_t_raw"]
    return df


def _join_features(
    pred_df: pd.DataFrame,
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """idx を key に特徴量 lookup を left-join する。

    EXP017 predictions にない特徴量カラム（eye_scale 等）を付与する。
    pred_df 側にある列が優先される。
    """
    # 重複カラムを避けるため feat_df から idx と pred_df 既存カラムを除外
    feat_cols = [c for c in feat_df.columns if c not in pred_df.columns or c == "idx"]
    added_cols = [c for c in feat_cols if c != "idx"]
    feat_subset = feat_df[feat_cols]
    merged = pred_df.merge(feat_subset, on="idx", how="left")
    n_missing = merged[added_cols].isnull().any(axis=1).sum()
    if n_missing > 0:
        logger.warning("join 後に %d 行で特徴量欠損あり（feature lookup に idx なし）", n_missing)
    return merged


def _split_train_val(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """image_path 単位でシャッフルして train/val に分割する。"""
    unique_images = np.array(sorted(df["image_path"].unique()))
    rng = np.random.default_rng(cfg.seed)
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * cfg.val_ratio)))
    n_train = max(1, len(shuffled) - n_val)

    val_imgs = set(shuffled[:n_val])
    train_imgs = set(shuffled[n_val:])

    train_df = df[df["image_path"].isin(train_imgs)].copy()
    val_df = df[df["image_path"].isin(val_imgs)].copy()

    assert set(train_df["image_path"]).isdisjoint(set(val_df["image_path"])), \
        "train/val 間で image_path 漏洩"

    return train_df, val_df


def run_split(
    feature_manifests_dir: Path,
    exp017_train_pred: Path,
    exp017_test_pred: Path,
    manifests_out_dir: Path,
    cfg: SplitConfig,
) -> None:
    feat_df = _load_feature_manifests(feature_manifests_dir)

    train_pred = _compute_residuals(exp017_train_pred)
    test_pred = _compute_residuals(exp017_test_pred)
    logger.info(
        "EXP017 predictions: train=%d test=%d",
        len(train_pred),
        len(test_pred),
    )

    train_full = _join_features(train_pred, feat_df)
    test_df = _join_features(test_pred, feat_df)

    train_df, val_df = _split_train_val(train_full, cfg)
    logger.info(
        "split: train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    manifests_out_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, pd.DataFrame] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for name, df in splits.items():
        path = manifests_out_dir / f"manifest_{name}.csv"
        df.to_csv(path, index=False)
        logger.info(
            "wrote %s: %d rows, %d images",
            path,
            len(df),
            df["image_path"].nunique(),
        )

    residual_stats = {
        "train": train_df["residual"].describe().to_dict(),
        "val": val_df["residual"].describe().to_dict(),
        "test": test_df["residual"].describe().to_dict(),
    }
    for split_name, stats in residual_stats.items():
        logger.info(
            "residual [%s] mean=%.4f std=%.4f",
            split_name,
            stats["mean"],
            stats["std"],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predres manifest 生成")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predres/config/baseline.yaml"),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] config が見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    data = cfg_dict["data"]
    split_cfg = SplitConfig(
        val_ratio=float(cfg_dict["split"]["val_ratio"]),
        seed=int(cfg_dict["split"]["seed"]),
    )

    run_split(
        feature_manifests_dir=Path(data["feature_manifests_dir"]),
        exp017_train_pred=Path(data["exp017_train_predictions"]),
        exp017_test_pred=Path(data["exp017_test_predictions"]),
        manifests_out_dir=Path(data["manifests_dir"]),
        cfg=split_cfg,
    )
    print("[OK] split 完了")


if __name__ == "__main__":
    main()

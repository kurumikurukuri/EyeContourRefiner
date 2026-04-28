"""xgboost_baseline: split.py のテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyses.xgboost_baseline.scr.split import (
    SplitConfig,
    split_by_image_path,
    write_manifests,
)


def _make_df(
    n_images: int = 20,
    rows_per_image: int = 3,
    unlabeled_images: int = 5,
    skip_flag_images: int = 0,
) -> pd.DataFrame:
    rows = []
    idx = 0
    for i in range(n_images):
        img = f"img_{i:03d}.jpg"
        has_label = i >= unlabeled_images
        skip = 1 if (has_label and i < unlabeled_images + skip_flag_images) else 0
        for k in range(rows_per_image):
            rows.append(
                {
                    "idx": idx,
                    "image_path": img,
                    "point_index": k,
                    "true_t_raw": 0.5 if has_label else np.nan,
                    "skip_flag": skip,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _default_cfg(**overrides) -> SplitConfig:
    base = dict(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        exclude_skip_flag=True,
        seed=42,
    )
    base.update(overrides)
    return SplitConfig(**base)


class TestSplitByImagePath:
    def test_returns_four_keys(self):
        df = _make_df()
        out = split_by_image_path(df, _default_cfg())
        assert set(out.keys()) == {"train", "val", "test", "unlabeled"}

    def test_no_image_path_leakage_between_splits(self):
        df = _make_df(n_images=30, unlabeled_images=5)
        out = split_by_image_path(df, _default_cfg())
        train_imgs = set(out["train"]["image_path"].unique())
        val_imgs = set(out["val"]["image_path"].unique())
        test_imgs = set(out["test"]["image_path"].unique())
        assert train_imgs.isdisjoint(val_imgs)
        assert train_imgs.isdisjoint(test_imgs)
        assert val_imgs.isdisjoint(test_imgs)

    def test_unlabeled_contains_all_nan_rows(self):
        df = _make_df(n_images=20, unlabeled_images=7, rows_per_image=2)
        out = split_by_image_path(df, _default_cfg())
        expected_unlabeled = df[df["true_t_raw"].isna()]
        assert len(out["unlabeled"]) == len(expected_unlabeled)
        assert out["unlabeled"]["true_t_raw"].isna().all()

    def test_skip_flag_excluded_when_configured(self):
        df = _make_df(n_images=20, unlabeled_images=3, skip_flag_images=2)
        cfg = _default_cfg(exclude_skip_flag=True)
        out = split_by_image_path(df, cfg)
        combined = pd.concat([out["train"], out["val"], out["test"]])
        assert (combined["skip_flag"] == 0).all()

    def test_skip_flag_included_when_configured(self):
        df = _make_df(n_images=20, unlabeled_images=3, skip_flag_images=2)
        cfg = _default_cfg(exclude_skip_flag=False)
        out = split_by_image_path(df, cfg)
        combined = pd.concat([out["train"], out["val"], out["test"]])
        # skip_flag=1 の画像が 2 個あり、それも annotated 扱いで含まれる
        assert (combined["skip_flag"] == 1).any()

    def test_seed_reproducibility(self):
        df = _make_df(n_images=30)
        out1 = split_by_image_path(df, _default_cfg(seed=123))
        out2 = split_by_image_path(df, _default_cfg(seed=123))
        assert sorted(out1["train"]["image_path"].unique().tolist()) == sorted(
            out2["train"]["image_path"].unique().tolist()
        )

    def test_different_seeds_produce_different_splits(self):
        df = _make_df(n_images=30)
        out1 = split_by_image_path(df, _default_cfg(seed=1))
        out2 = split_by_image_path(df, _default_cfg(seed=999))
        imgs1 = sorted(out1["train"]["image_path"].unique().tolist())
        imgs2 = sorted(out2["train"]["image_path"].unique().tolist())
        assert imgs1 != imgs2

    def test_ratio_normalization(self):
        # 合計 2.0 でも内部で正規化されて同じ比率の結果になる
        df = _make_df(n_images=40)
        cfg_a = _default_cfg(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        cfg_b = _default_cfg(train_ratio=1.40, val_ratio=0.30, test_ratio=0.30)
        out_a = split_by_image_path(df, cfg_a)
        out_b = split_by_image_path(df, cfg_b)
        assert sorted(out_a["train"]["image_path"].unique().tolist()) == sorted(
            out_b["train"]["image_path"].unique().tolist()
        )

    def test_raises_when_too_few_annotated_images(self):
        df = _make_df(n_images=5, unlabeled_images=3)  # annotated=2
        with pytest.raises(ValueError, match="annotated 画像数"):
            split_by_image_path(df, _default_cfg())

    def test_raises_when_true_t_raw_column_missing(self):
        df = pd.DataFrame([{"image_path": "a.jpg", "skip_flag": 0}])
        with pytest.raises(ValueError, match="true_t_raw"):
            split_by_image_path(df, _default_cfg())

    def test_raises_when_all_ratios_zero(self):
        df = _make_df(n_images=20)
        with pytest.raises(ValueError, match="合計比"):
            split_by_image_path(df, _default_cfg(train_ratio=0, val_ratio=0, test_ratio=0))

    def test_split_proportions_approx(self):
        df = _make_df(n_images=100, rows_per_image=1, unlabeled_images=0)
        out = split_by_image_path(df, _default_cfg())
        n_train = out["train"]["image_path"].nunique()
        n_val = out["val"]["image_path"].nunique()
        n_test = out["test"]["image_path"].nunique()
        total = n_train + n_val + n_test
        assert total == 100
        assert abs(n_train / total - 0.70) < 0.05
        assert abs(n_val / total - 0.15) < 0.05
        assert abs(n_test / total - 0.15) < 0.05

    def test_each_split_has_at_least_one_image(self):
        # 少ない画像数でも各 split に最低1枚入ることを保証
        df = _make_df(n_images=5, unlabeled_images=0)
        out = split_by_image_path(df, _default_cfg())
        assert out["train"]["image_path"].nunique() >= 1
        assert out["val"]["image_path"].nunique() >= 1
        assert out["test"]["image_path"].nunique() >= 1


class TestWriteManifests:
    def test_writes_four_files(self, tmp_path):
        df = _make_df(n_images=20)
        out = split_by_image_path(df, _default_cfg())
        write_manifests(out, tmp_path)
        for name in ["train", "val", "test", "unlabeled"]:
            assert (tmp_path / f"manifest_{name}.csv").exists()

    def test_written_csv_roundtrips(self, tmp_path):
        df = _make_df(n_images=20)
        out = split_by_image_path(df, _default_cfg())
        write_manifests(out, tmp_path)
        back = pd.read_csv(tmp_path / "manifest_train.csv")
        assert len(back) == len(out["train"])
        assert set(back["image_path"].unique()) == set(
            out["train"]["image_path"].unique()
        )

    def test_creates_parent_directory(self, tmp_path):
        df = _make_df(n_images=20)
        out = split_by_image_path(df, _default_cfg())
        target = tmp_path / "nested" / "dir"
        write_manifests(out, target)
        assert target.exists()
        assert (target / "manifest_train.csv").exists()

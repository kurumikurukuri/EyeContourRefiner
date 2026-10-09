"""eyeliner.training.oof_kfold の単体テスト。

split_fixed_folds() の網羅性・重複なし・再現性、run_oof_kfold() の OOF 一意性
（訓練全サンプルが重複なく 1 回ずつ予測を持つこと）を検証する。
run_dual_oof_kfold() / run_nested_dual_oof_kfold() でデュアルブランチ版も同様に検証する。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyeliner.training.dual_kfold import DualTrainConfig
from eyeliner.training.oof_kfold import (
    run_dual_oof_kfold,
    run_nested_dual_oof_kfold,
    run_nested_oof_kfold,
    run_oof_kfold,
    split_fixed_folds,
)
from eyeliner.training.trainer import TrainConfig

MANIFEST_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
]


def _write_image(path: Path, size: int = 200) -> None:
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_manifest(tmp_path: Path, n_images: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n_images):
        img_path = image_dir / f"img_{i}.jpg"
        _write_image(img_path)
        rows.append({
            "idx": str(i),
            "image_path": str(img_path),
            "part_name": "eyeliner_right",
            "sub_key": "upper",
            "point_index": 3,
            "px": 60.0, "py": 100.0, "qx": 140.0, "qy": 100.0,
            "t": 0.5,
        })
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def _fast_config(**overrides) -> TrainConfig:
    base = dict(
        backbone="custom",
        loss_fn="t",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=2,
        patience=1,
        seed=42,
        num_conv_layers=2,
        use_sigmoid=True,
        use_batchnorm=False,
        dropout_rate=0.0,
        aug_flags=0,
        num_workers=0,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _fast_dual_config(**overrides) -> DualTrainConfig:
    base = dict(
        fusion_mode="independent",
        pq_rescale_mode_b="eye_scale",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=2,
        patience=1,
        seed=42,
        num_conv_layers=2,
        use_sigmoid=True,
        use_batchnorm=False,
        dropout_rate=0.0,
        aug_flags=0,
        endpoint_blend_range=0.25,
        loss_fn="t",
        num_workers=0,
        base_channel=16,
        pool_width=1,
    )
    base.update(overrides)
    return DualTrainConfig(**base)


class TestSplitFixedFolds:

    def test_covers_all_rows_exactly_once(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=20)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)
        assert len(fold_paths) == 5

        all_idx: list[str] = []
        for p in fold_paths:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            all_idx.extend(r["idx"] for r in rows)

        assert sorted(all_idx, key=int) == [str(i) for i in range(20)]
        assert len(all_idx) == len(set(all_idx))

    def test_is_reproducible_with_same_seed(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=20)
        paths_a = split_fixed_folds(manifest, tmp_path / "folds_a", k=5, seed=42)
        paths_b = split_fixed_folds(manifest, tmp_path / "folds_b", k=5, seed=42)

        def fold_idx_sets(paths):
            sets = []
            for p in paths:
                with open(p, newline="", encoding="utf-8") as f:
                    sets.append({r["idx"] for r in csv.DictReader(f)})
            return sets

        assert fold_idx_sets(paths_a) == fold_idx_sets(paths_b)

    def test_no_records_raises(self, tmp_path: Path):
        manifest = tmp_path / "empty.csv"
        _write_manifest(manifest, [])
        with pytest.raises(ValueError):
            split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)


class TestRunOofKfold:

    def test_oof_predictions_cover_all_samples_exactly_once(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        result = run_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert result.n_total == 8
        oof_idx = [p.idx for p in result.oof_predictions]
        assert sorted(oof_idx, key=int) == [str(i) for i in range(8)]
        assert len(oof_idx) == len(set(oof_idx))

    def test_fold_idx_within_range(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        result = run_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert {p.fold_idx for p in result.oof_predictions} <= {1, 2}

    def test_returns_one_fold_result_per_fold_file(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        result = run_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert len(result.fold_results) == 2

    def test_saves_fold_model_files(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)
        exp_dir = tmp_path / "outputs"

        run_oof_kfold(fold_paths, exp_dir, _fast_config())

        saved = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2


class TestRunNestedOofKfold:

    def test_requires_at_least_three_folds(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        with pytest.raises(ValueError):
            run_nested_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

    def test_oof_predictions_cover_all_samples_exactly_once(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=15)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)

        result = run_nested_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert result.n_total == 15
        oof_idx = [p.idx for p in result.oof_predictions]
        assert sorted(oof_idx, key=int) == [str(i) for i in range(15)]
        assert len(oof_idx) == len(set(oof_idx))

    def test_returns_one_fold_result_per_round(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=15)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)

        result = run_nested_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert len(result.fold_results) == 5

    def test_test_records_are_disjoint_from_their_round_val_records(self, tmp_path: Path):
        """各 round の Test (= fold[i]) は Val (= fold[(i+1)%k]) と重複しないこと。"""
        manifest = _make_manifest(tmp_path, n_images=15)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)

        fold_idx_sets = []
        for p in fold_paths:
            with open(p, newline="", encoding="utf-8") as f:
                fold_idx_sets.append({r["idx"] for r in csv.DictReader(f)})

        k = len(fold_idx_sets)
        for i in range(k):
            test_set = fold_idx_sets[i]
            val_set = fold_idx_sets[(i + 1) % k]
            assert test_set.isdisjoint(val_set)

    def test_minimum_k_of_three_runs(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=9)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=3, seed=42)

        result = run_nested_oof_kfold(fold_paths, tmp_path / "outputs", _fast_config())

        assert result.n_total == 9
        assert len(result.oof_predictions) == 9


class TestRunDualOofKfold:

    def test_oof_predictions_cover_all_samples_exactly_once(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        result = run_dual_oof_kfold(fold_paths, tmp_path / "outputs", _fast_dual_config())

        assert result.n_total == 8
        oof_idx = [p.idx for p in result.oof_predictions]
        assert sorted(oof_idx, key=int) == [str(i) for i in range(8)]
        assert len(oof_idx) == len(set(oof_idx))

    def test_returns_one_fold_result_per_fold_file(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        result = run_dual_oof_kfold(fold_paths, tmp_path / "outputs", _fast_dual_config())

        assert len(result.fold_results) == 2

    def test_saves_fold_model_files(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)
        exp_dir = tmp_path / "outputs"

        run_dual_oof_kfold(fold_paths, exp_dir, _fast_dual_config())

        saved = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2


class TestRunNestedDualOofKfold:

    def test_requires_at_least_three_folds(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=2, seed=42)

        with pytest.raises(ValueError):
            run_nested_dual_oof_kfold(fold_paths, tmp_path / "outputs", _fast_dual_config())

    def test_oof_predictions_cover_all_samples_exactly_once(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=15)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)

        result = run_nested_dual_oof_kfold(fold_paths, tmp_path / "outputs", _fast_dual_config())

        assert result.n_total == 15
        oof_idx = [p.idx for p in result.oof_predictions]
        assert sorted(oof_idx, key=int) == [str(i) for i in range(15)]
        assert len(oof_idx) == len(set(oof_idx))

    def test_returns_one_fold_result_per_round(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=15)
        fold_paths = split_fixed_folds(manifest, tmp_path / "folds", k=5, seed=42)

        result = run_nested_dual_oof_kfold(fold_paths, tmp_path / "outputs", _fast_dual_config())

        assert len(result.fold_results) == 5

"""eyeliner.training.dual_kfold の単体テスト。

旧 experiments/EXP020_dual_branch/scr/train_exp020.py の train_dual() を
eyeliner/training/dual_kfold.py に抽出した run_dual_kfold() を検証する。
EXP020/EXP021 双方から import される共有実装になるため、モジュール定数
（旧 MANIFEST_TRAIN/MANIFEST_TEST/OUTPUT_DIR）に依存せず、すべて明示的な
引数で完全に制御できることをテストする
（旧実装は manifest_train/manifest_test の override 引数を計算しても
  実際の load_manifest() 呼び出しでは使っていないバグがあった — 本テストの
  test_run_dual_kfold_uses_explicit_manifest_paths がその回帰防止）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyeliner.training.dual_kfold import (
    DualKfoldResult,
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)


def _write_image(path: Path, size: int = 200) -> None:
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy", "t",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_records(image_dir: Path, n: int, prefix: str = "img") -> list[dict]:
    rows = []
    for i in range(n):
        img_path = image_dir / f"{prefix}_{i}.jpg"
        _write_image(img_path)
        rows.append({
            "image_path": str(img_path),
            "part_name": "eyeliner_right",
            "sub_key": "upper",
            "point_index": 3,
            "px": 60.0, "py": 100.0, "qx": 140.0, "qy": 100.0,
            "t": 0.5,
        })
    return rows


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    train_rows = _make_records(image_dir, 4, prefix="train")
    test_rows = _make_records(image_dir, 2, prefix="test")
    train_path = tmp_path / "manifest_train.csv"
    test_path = tmp_path / "manifest_test.csv"
    _write_manifest(train_path, train_rows)
    _write_manifest(test_path, test_rows)
    return train_path, test_path


def _fast_config(**overrides) -> DualTrainConfig:
    base = dict(
        fusion_mode="independent",
        pq_rescale_mode_b="eye_scale",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=2,
        k=2,
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
    )
    base.update(overrides)
    return DualTrainConfig(**base)


class TestRunDualKfold:

    def test_returns_one_fold_result_per_k(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "cond_a"
        result = run_dual_kfold(manifest_train, manifest_test, exp_dir, _fast_config())
        assert isinstance(result, DualKfoldResult)
        assert len(result.fold_results) == 2

    def test_saves_fold_model_files_to_exp_dir(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "cond_b"
        run_dual_kfold(manifest_train, manifest_test, exp_dir, _fast_config())
        saved = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2

    def test_computes_holdout_mae_when_holdout_given(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "cond_c"
        result = run_dual_kfold(manifest_train, manifest_test, exp_dir, _fast_config())
        assert result.holdout_mae is not None
        assert result.holdout_mae_raw is not None
        assert result.holdout_mae >= 0.0
        assert result.n_holdout == 2

    def test_holdout_is_none_when_manifest_test_is_none(self, tmp_path, manifests):
        manifest_train, _ = manifests
        exp_dir = tmp_path / "outputs" / "cond_d"
        result = run_dual_kfold(manifest_train, None, exp_dir, _fast_config())
        assert result.holdout_mae is None
        assert result.holdout_mae_raw is None
        assert result.n_holdout == 0

    def test_uses_explicit_manifest_paths_not_module_globals(self, tmp_path):
        """manifest_train/manifest_test は呼び出し時の引数がそのまま使われること。

        旧 train_dual() は output_dir/manifest_train/manifest_test を引数化した際、
        existence チェックにのみ override 変数を使い、実際の load_manifest() 呼び出しは
        モジュール定数 MANIFEST_TRAIN/MANIFEST_TEST を参照していた（サイレントなバグ）。
        run_dual_kfold() にはそもそもモジュール定数が存在しないため、
        異なる manifest を渡せば異なる n_train になることを確認する。
        """
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        small_path = tmp_path / "manifest_small.csv"
        large_path = tmp_path / "manifest_large.csv"
        _write_manifest(small_path, _make_records(image_dir, 4, prefix="small"))
        _write_manifest(large_path, _make_records(image_dir, 6, prefix="large"))

        result_small = run_dual_kfold(
            small_path, None, tmp_path / "outputs" / "small", _fast_config(),
        )
        result_large = run_dual_kfold(
            large_path, None, tmp_path / "outputs" / "large", _fast_config(),
        )
        assert result_small.n_train == 4
        assert result_large.n_train == 6

    def test_respects_explicit_exp_dir(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        custom_dir = tmp_path / "somewhere_else"
        run_dual_kfold(manifest_train, manifest_test, custom_dir, _fast_config())
        assert len(list(custom_dir.glob("eyeliner_detection_fold*.pt"))) == 2

    def test_runs_with_custom_base_channel_and_pool_width(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "cond_arch"
        config = _fast_config(base_channel=16, pool_width=1)
        result = run_dual_kfold(manifest_train, manifest_test, exp_dir, config)
        assert len(result.fold_results) == 2
        assert len(list(exp_dir.glob("eyeliner_detection_fold*.pt"))) == 2


class TestBuildDualExperimentRecord:

    def test_record_contains_expected_keys(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "cond_e"
        config = _fast_config()
        result = run_dual_kfold(manifest_train, manifest_test, exp_dir, config)
        record = build_dual_experiment_record("cond_e", config, result)

        assert record["exp_name"] == "cond_e"
        hp = record["hyperparameters"]
        assert hp["fusion_mode"] == "independent"
        assert hp["aug_flags"] == 0
        assert hp["endpoint_blend_range"] == pytest.approx(0.25)
        assert hp["loss_fn"] == "t"
        assert hp["branch_a"] is not None
        assert hp["branch_b"] is not None
        assert hp["base_channel"] == 32
        assert hp["pool_width"] == 32

        data = record["data"]
        assert data["n_train_samples"] == 4
        assert data["n_holdout_samples"] == 2

        results = record["results"]
        assert len(results["cv_folds"]) == 2
        assert results["holdout_mae"] is not None

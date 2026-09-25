"""eyeliner.training.kfold.run_kfold() の回帰テスト。

_train_single_fold() への抽出リファクタ後も既存の入出力・挙動が不変であることを確認する。
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

import eyeliner.training.kfold as kfold_module
from eyeliner.training.kfold import FoldResult, run_kfold
from eyeliner.training.trainer import TrainConfig

MANIFEST_FIELDNAMES = [
    "image_path", "part_name", "sub_key", "point_index", "px", "py", "qx", "qy", "t",
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


class TestRunKfold:

    def test_returns_one_fold_result_per_k(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        result = run_kfold(manifest, tmp_path / "outputs", _fast_config(), k_folds=2)
        assert all(isinstance(r, FoldResult) for r in result)
        assert len(result) == 2

    def test_saves_fold_model_files(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        run_kfold(manifest, exp_dir, _fast_config(), k_folds=2)
        saved = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2

    def test_k_equals_one_uses_group_shuffle_split(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        result = run_kfold(manifest, tmp_path / "outputs", _fast_config(), k_folds=1)
        assert len(result) == 1


class TestRunKfoldResume:
    """中断からのリジューム機能。dual_kfold.py と実装・挙動を揃える。"""

    def test_no_checkpoint_files_remain_after_successful_run(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        run_kfold(manifest, exp_dir, _fast_config(epochs=2), k_folds=2)

        assert list(exp_dir.glob("checkpoint_fold*.pt")) == []

    def test_resume_does_not_retrain_already_saved_epochs(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        config = _fast_config(epochs=4, patience=10)

        recorded: list[tuple[str, int]] = []
        real_save_checkpoint = kfold_module.save_checkpoint
        state = {"raised": False}

        def flaky_save_checkpoint(path, *, epoch, **kwargs):
            real_save_checkpoint(path, epoch=epoch, **kwargs)
            recorded.append((Path(path).name, epoch))
            if not state["raised"] and Path(path).name == "checkpoint_fold1.pt" and epoch == 2:
                state["raised"] = True
                raise RuntimeError("simulated crash")

        with mock.patch.object(kfold_module, "save_checkpoint", side_effect=flaky_save_checkpoint):
            with pytest.raises(RuntimeError, match="simulated crash"):
                run_kfold(manifest, exp_dir, config, k_folds=2)

            fold1_ckpt = kfold_module.load_checkpoint(exp_dir / "checkpoint_fold1.pt")
            assert fold1_ckpt is not None
            assert fold1_ckpt["epoch"] == 2

            result = run_kfold(manifest, exp_dir, config, k_folds=2)

        fold1_epochs = [e for name, e in recorded if name == "checkpoint_fold1.pt"]
        assert fold1_epochs == [1, 2, 3, 4]
        assert len(result) == 2
        assert not (exp_dir / "checkpoint_fold1.pt").exists()
        assert not (exp_dir / "checkpoint_fold2.pt").exists()

    def test_rerun_skips_fully_completed_folds(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        config = _fast_config(epochs=1)

        first = run_kfold(manifest, exp_dir, config, k_folds=2)

        call_log: list[int] = []
        real_train_fold = kfold_module._train_single_fold

        def spy_train_fold(*args, **kwargs):
            call_log.append(args[2])
            return real_train_fold(*args, **kwargs)

        with mock.patch.object(kfold_module, "_train_single_fold", side_effect=spy_train_fold):
            second = run_kfold(manifest, exp_dir, config, k_folds=2)

        assert call_log == []
        assert [r.best_val_mae for r in second] == [r.best_val_mae for r in first]

    def test_resume_false_forces_full_retrain(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        config = _fast_config(epochs=1)

        run_kfold(manifest, exp_dir, config, k_folds=2)

        call_log: list[int] = []
        real_train_fold = kfold_module._train_single_fold

        def spy_train_fold(*args, **kwargs):
            call_log.append(args[2])
            return real_train_fold(*args, **kwargs)

        with mock.patch.object(kfold_module, "_train_single_fold", side_effect=spy_train_fold):
            run_kfold(manifest, exp_dir, config, k_folds=2, resume=False)

        assert call_log == [1, 2]

    def test_progress_file_records_completed_folds(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        run_kfold(manifest, exp_dir, _fast_config(epochs=1), k_folds=2)

        assert (exp_dir / "kfold_progress.json").exists()


class TestGroupFoldCompleteness:
    """GroupKFold が同一画像のサンプルを train/val に分割しないこと。

    テスト用 manifest が 1 画像 = 1 サンプルだと、group key を取り違えても検出できない。
    同一 image_path を複数サンプルで共有する manifest を使い、画像が fold を跨がないことを保証する。
    """

    def test_same_image_samples_stay_together(self, tmp_path: Path):
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        rows = []
        for i in range(6):
            img_path = image_dir / f"img_{i}.jpg"
            _write_image(img_path)
            for point_index in range(3):
                rows.append({
                    "image_path": str(img_path),
                    "part_name": "eyeliner_right",
                    "sub_key": "upper",
                    "point_index": point_index,
                    "px": 60.0, "py": 100.0, "qx": 140.0, "qy": 100.0,
                    "t": 0.5,
                })
        manifest = tmp_path / "manifest.csv"
        _write_manifest(manifest, rows)

        captured: list[tuple[int, list[dict], list[dict]]] = []

        def spy_train_fold(train_records, val_records, fold_idx, *args, **kwargs):
            captured.append((fold_idx, train_records, val_records))
            return FoldResult(fold_idx=fold_idx, best_val_mae=0.0, best_epoch=0, model_path="")

        with mock.patch.object(kfold_module, "_train_single_fold", side_effect=spy_train_fold):
            run_kfold(manifest, tmp_path / "outputs", _fast_config(), k_folds=3)

        assert len(captured) == 3
        for fold_idx, train_records, val_records in captured:
            train_images = {r["image_path"] for r in train_records}
            val_images = {r["image_path"] for r in val_records}
            assert train_images.isdisjoint(val_images), (
                f"fold {fold_idx} で同一画像が train/val に分割された"
            )


# ---------------------------------------------------------------------------
# _build_oof_predictions: pq_rescale 時は t' 予測を生 t に戻してから t_true と並べる
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pq_rescale_mode", ["eye_scale", "none"])
def test_build_oof_predictions_returns_raw_t(pq_rescale_mode):
    from eyeliner.data.patch_utils import compute_pq_rescale

    rec = {
        "idx": "7", "image_path": "a.jpg", "part_name": "eyeliner_right",
        "sub_key": "upper", "point_index": "3",
        "px": "100.0", "py": "50.0", "qx": "130.0", "qy": "60.0",
        "eye_scale": "80.0", "t": "0.8",
    }
    config = TrainConfig(pq_rescale_mode=pq_rescale_mode, pq_rescale_factor=0.5, use_t_norm=True)
    label_t_min, label_range = -0.5, 2.0
    *_, t_prime, _ = compute_pq_rescale(
        100.0, 50.0, 130.0, 60.0, 0.8, rec, pq_rescale_mode, 0.5,
    )
    pred_norm = torch.tensor([[(t_prime - label_t_min) / label_range]])

    oof = kfold_module._build_oof_predictions(
        [rec], pred_norm, config, label_range, label_t_min, fold_idx=0,
    )

    assert oof[0].t_pred == pytest.approx(0.8, abs=1e-5)
    assert oof[0].t_true == pytest.approx(0.8)

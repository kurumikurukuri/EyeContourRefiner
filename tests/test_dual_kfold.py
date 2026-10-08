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
import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

import eyeliner.training.dual_averaging as dual_averaging_module
import eyeliner.training.dual_kfold as dual_kfold_module
from eyeliner.data.augmentation import (
    AUG_ALL,
    AUG_PQ_JITTER,
    AUG_SOURCE_JPEG,
    AUG_SOURCE_RES_JITTER,
    augment,
    check_aug_flags,
)
from eyeliner.data.dataset_dual import BranchConfig
from eyeliner.training.dual_kfold import (
    DualKfoldResult,
    DualTrainConfig,
    build_dual_experiment_record,
    build_source_degrade_config,
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


class TestDualKfoldResume:
    """中断からのリジューム機能。EXP027 が実際に使う経路なので優先的に検証する。"""

    def test_no_checkpoint_files_remain_after_successful_run(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "resume_clean"
        run_dual_kfold(manifest_train, manifest_test, exp_dir, _fast_config(epochs=2))

        assert list(exp_dir.glob("checkpoint_fold*.pt")) == []

    def test_resume_does_not_retrain_already_saved_epochs(self, tmp_path, manifests):
        """途中エポックのチェックポイント保存直後にクラッシュしても、
        再実行時に完了済みエポックを再学習しないことを確認する。
        """
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "resume_epoch"
        config = _fast_config(epochs=4, patience=10)

        recorded: list[tuple[str, int]] = []
        real_save_checkpoint = dual_kfold_module.save_checkpoint
        state = {"raised": False}

        def flaky_save_checkpoint(path, *, epoch, **kwargs):
            real_save_checkpoint(path, epoch=epoch, **kwargs)
            recorded.append((Path(path).name, epoch))
            if not state["raised"] and Path(path).name == "checkpoint_fold1.pt" and epoch == 2:
                state["raised"] = True
                raise RuntimeError("simulated crash")

        with mock.patch.object(dual_kfold_module, "save_checkpoint", side_effect=flaky_save_checkpoint):
            with pytest.raises(RuntimeError, match="simulated crash"):
                run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

            fold1_ckpt = dual_kfold_module.load_checkpoint(exp_dir / "checkpoint_fold1.pt")
            assert fold1_ckpt is not None
            assert fold1_ckpt["epoch"] == 2

            result = run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        fold1_epochs = [e for name, e in recorded if name == "checkpoint_fold1.pt"]
        assert fold1_epochs == [1, 2, 3, 4]
        assert len(result.fold_results) == config.k
        assert not (exp_dir / "checkpoint_fold1.pt").exists()
        assert not (exp_dir / "checkpoint_fold2.pt").exists()

    def test_rerun_skips_fully_completed_folds(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "resume_skip"
        config = _fast_config(epochs=1)

        first = run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        call_log: list[int] = []
        real_train_fold = dual_kfold_module._train_single_dual_fold

        def spy_train_fold(*args, **kwargs):
            call_log.append(args[2])
            return real_train_fold(*args, **kwargs)

        with mock.patch.object(dual_kfold_module, "_train_single_dual_fold", side_effect=spy_train_fold):
            second = run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        assert call_log == []
        assert [r.best_val_mae for r in second.fold_results] == [r.best_val_mae for r in first.fold_results]

    def test_resume_false_forces_full_retrain(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "resume_disabled"
        config = _fast_config(epochs=1)

        run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        call_log: list[int] = []
        real_train_fold = dual_kfold_module._train_single_dual_fold

        def spy_train_fold(*args, **kwargs):
            call_log.append(args[2])
            return real_train_fold(*args, **kwargs)

        with mock.patch.object(dual_kfold_module, "_train_single_dual_fold", side_effect=spy_train_fold):
            run_dual_kfold(manifest_train, manifest_test, exp_dir, config, resume=False)

        assert call_log == [1, 2]

    def test_progress_file_records_completed_folds(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "resume_progress"
        run_dual_kfold(manifest_train, manifest_test, exp_dir, _fast_config(epochs=1))

        progress_path = exp_dir / "dual_kfold_progress.json"
        assert progress_path.exists()


class TestSourceDegradeFlags:
    """EXP031 の元画像空間劣化は aug_flags のビットで ON/OFF する。"""

    def test_no_source_bits_returns_none(self):
        config = _fast_config(aug_flags=63)
        assert build_source_degrade_config(config) is None

    def test_jpeg_bit_enables_jpeg_only(self):
        config = _fast_config(aug_flags=63 | AUG_SOURCE_JPEG)
        cfg = build_source_degrade_config(config)
        assert cfg is not None
        assert cfg.jpeg is True
        assert cfg.res_jitter is False
        assert cfg.enabled is True

    def test_res_jitter_bit_enables_res_jitter_only(self):
        config = _fast_config(aug_flags=63 | AUG_SOURCE_RES_JITTER)
        cfg = build_source_degrade_config(config)
        assert cfg is not None
        assert cfg.jpeg is False
        assert cfg.res_jitter is True

    def test_both_bits_enable_both(self):
        config = _fast_config(aug_flags=63 | AUG_SOURCE_JPEG | AUG_SOURCE_RES_JITTER)
        cfg = build_source_degrade_config(config)
        assert cfg is not None
        assert (cfg.jpeg, cfg.res_jitter) == (True, True)

    def test_source_bits_do_not_disturb_patch_bits(self):
        """source ビットを立ててもパッチ空間 aug の解釈は変わらない。"""
        patch = np.full((8, 8, 3), 0.5, dtype=np.float32)
        with mock.patch("random.random", return_value=0.99),              mock.patch("random.uniform", return_value=0.0):
            out, t = augment(patch, 0.5, aug_flags=AUG_SOURCE_JPEG | AUG_SOURCE_RES_JITTER)
        assert t == 0.5
        assert np.allclose(out, patch)

    def test_intensity_params_are_passed_through(self):
        config = _fast_config(
            aug_flags=63 | AUG_SOURCE_JPEG | AUG_SOURCE_RES_JITTER,
            source_jpeg_prob=0.25,
            source_jpeg_quality_min=40,
            source_jpeg_quality_max=80,
            source_res_jitter_prob=0.75,
            source_res_scale_min=1.5,
            source_res_scale_max=2.0,
        )
        cfg = build_source_degrade_config(config)
        assert cfg.jpeg_prob == 0.25
        assert (cfg.jpeg_quality_min, cfg.jpeg_quality_max) == (40, 80)
        assert cfg.res_jitter_prob == 0.75
        assert (cfg.res_scale_min, cfg.res_scale_max) == (1.5, 2.0)

    def test_invalid_intensity_params_raise(self):
        config = _fast_config(
            aug_flags=63 | AUG_SOURCE_JPEG,
            source_jpeg_quality_min=90,
            source_jpeg_quality_max=30,
        )
        with pytest.raises(ValueError):
            build_source_degrade_config(config)

    def test_experiment_record_reflects_bits(self):
        config = _fast_config(aug_flags=63 | AUG_SOURCE_JPEG)
        branch = BranchConfig(
            patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        )
        result = DualKfoldResult(
            fold_results=[], n_train=0, n_holdout=0,
            train_images=[], holdout_images=[], cfg_a=branch, cfg_b=branch,
        )
        record = build_dual_experiment_record("t", config, result)
        hp = record["hyperparameters"]
        assert hp["aug_flags"] == 63 | AUG_SOURCE_JPEG
        assert hp["source_jpeg"] is True
        assert hp["source_res_jitter"] is False

    def test_unknown_bits_are_warned(self, caplog):
        with caplog.at_level("WARNING"):
            check_aug_flags(0x1FF | 0x800)
        assert "未定義ビット" in caplog.text

    def test_known_bits_do_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            check_aug_flags(AUG_ALL)
        assert caplog.text == ""


class TestWeightAveragingVariants:
    """EXP034: EMA / SWA の派生モデルを base と同一 run 内で作る。"""

    def test_default_config_creates_no_variant_dirs(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        outputs = tmp_path / "outputs"
        run_dual_kfold(manifest_train, manifest_test, outputs / "base", _fast_config())
        assert sorted(p.name for p in outputs.iterdir()) == ["base"]

    def test_ema_saves_fold_models_to_sibling_dir(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        outputs = tmp_path / "outputs"
        config = _fast_config(epochs=2, patience=5, ema_decays=[0.9])
        run_dual_kfold(manifest_train, manifest_test, outputs / "base", config)

        ema_dir = outputs / "base_ema09"
        saved = sorted(ema_dir.glob("eyeliner_detection_fold*.pt"))
        assert [p.name for p in saved] == ["eyeliner_detection_fold1.pt", "eyeliner_detection_fold2.pt"]
        for path in saved:
            assert dual_kfold_module.load_dual_model(path) is not None
        folds = json.loads((ema_dir / "variant_folds.json").read_text(encoding="utf-8"))
        assert sorted(folds) == ["1", "2"]
        assert all(v["val_mae"] >= 0.0 and v["epoch"] >= 1 for v in folds.values())

    def test_swa_saves_bn_model_to_sibling_dir(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        outputs = tmp_path / "outputs"
        config = _fast_config(
            epochs=1, use_batchnorm=True, swa_epochs=2, swa_lr=2.5e-4,
        )
        run_dual_kfold(manifest_train, manifest_test, outputs / "base", config)

        swa_dir = outputs / "base_swa"
        saved = sorted(swa_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2
        folds = json.loads((swa_dir / "variant_folds.json").read_text(encoding="utf-8"))
        assert len(folds["1"]["swa_epoch_val_maes"]) == 2
        assert not list((outputs / "base").glob("checkpoint_fold*.pt"))

    def test_variants_do_not_change_base_models(self, tmp_path, manifests):
        """EMA 評価・SWA の乱数消費が base（次 fold を含む）の学習に影響しない。"""
        manifest_train, manifest_test = manifests
        plain = tmp_path / "plain" / "base"
        averaged = tmp_path / "averaged" / "base"
        run_dual_kfold(manifest_train, manifest_test, plain, _fast_config(epochs=2, patience=5))
        run_dual_kfold(
            manifest_train, manifest_test, averaged,
            _fast_config(epochs=2, patience=5, ema_decays=[0.9], swa_epochs=1),
        )
        for fold in (1, 2):
            name = f"eyeliner_detection_fold{fold}.pt"
            a = torch.load(plain / name, map_location="cpu", weights_only=True)["state_dict"]
            b = torch.load(averaged / name, map_location="cpu", weights_only=True)["state_dict"]
            for key in a:
                assert torch.equal(a[key], b[key]), (fold, key)

    def test_crash_during_swa_resumes_without_retraining_main_epochs(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "base"
        config = _fast_config(epochs=3, patience=10, ema_decays=[0.9], swa_epochs=1)

        recorded: list[tuple[str, int]] = []
        real_save_checkpoint = dual_kfold_module.save_checkpoint
        real_run_swa = dual_averaging_module.run_swa
        state = {"raised": False}

        def recording_save_checkpoint(path, *, epoch, **kwargs):
            real_save_checkpoint(path, epoch=epoch, **kwargs)
            recorded.append((Path(path).name, epoch))

        def flaky_run_swa(*args, **kwargs):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("simulated crash in swa")
            return real_run_swa(*args, **kwargs)

        with mock.patch.object(dual_kfold_module, "save_checkpoint", side_effect=recording_save_checkpoint), \
                mock.patch.object(dual_averaging_module, "run_swa", side_effect=flaky_run_swa):
            with pytest.raises(RuntimeError, match="simulated crash in swa"):
                run_dual_kfold(manifest_train, manifest_test, exp_dir, config)
            run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        fold1_epochs = [e for name, e in recorded if name == "checkpoint_fold1.pt"]
        assert fold1_epochs == [1, 2, 3]
        assert len(list((tmp_path / "outputs" / "base_swa").glob("eyeliner_detection_fold*.pt"))) == 2
        assert len(list((tmp_path / "outputs" / "base_ema09").glob("eyeliner_detection_fold*.pt"))) == 2

    def test_early_stopped_fold_does_not_train_extra_epoch_on_resume(self, tmp_path, manifests):
        """早期終了済みの fold は、SWA 中のクラッシュから再開しても本学習を再開しない。"""
        manifest_train, manifest_test = manifests
        exp_dir = tmp_path / "outputs" / "base"
        config = _fast_config(epochs=20, patience=1, swa_epochs=1)
        real_run_swa = dual_averaging_module.run_swa
        state = {"raised": False}
        counts = {"first": 0, "second": 0}
        real_save_checkpoint = dual_kfold_module.save_checkpoint
        phase = {"name": "first"}

        def counting_save_checkpoint(path, *, epoch, **kwargs):
            real_save_checkpoint(path, epoch=epoch, **kwargs)
            if Path(path).name == "checkpoint_fold1.pt":
                counts[phase["name"]] += 1

        def flaky_run_swa(*args, **kwargs):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("simulated crash in swa")
            return real_run_swa(*args, **kwargs)

        with mock.patch.object(dual_kfold_module, "save_checkpoint", side_effect=counting_save_checkpoint), \
                mock.patch.object(dual_averaging_module, "run_swa", side_effect=flaky_run_swa):
            with pytest.raises(RuntimeError):
                run_dual_kfold(manifest_train, manifest_test, exp_dir, config)
            phase["name"] = "second"
            run_dual_kfold(manifest_train, manifest_test, exp_dir, config)

        assert counts["first"] < config.epochs
        assert counts["second"] == 0

    def test_record_contains_averaging_settings(self, tmp_path, manifests):
        config = _fast_config(ema_decays=[0.999, 0.9998], swa_epochs=30, swa_lr=2.5e-4)
        branch = BranchConfig(
            patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        )
        result = DualKfoldResult(
            fold_results=[], n_train=0, n_holdout=0,
            train_images=[], holdout_images=[], cfg_a=branch, cfg_b=branch,
        )
        hp = build_dual_experiment_record("t", config, result)["hyperparameters"]
        assert hp["ema_decays"] == [0.999, 0.9998]
        assert hp["swa_epochs"] == 30
        assert hp["swa_lr"] == 2.5e-4


class TestPqJitterConfig:
    """EXP035: PQ 回転摂動の強度パラメータが config → dataset → record に流れる。"""

    def test_default_is_disabled(self):
        config = _fast_config()
        assert config.pq_jitter_rot_deg == 0.0
        assert config.pq_jitter_prob == 1.0
        assert not (config.aug_flags & AUG_PQ_JITTER)

    def test_only_train_dataset_receives_jitter_params(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        config = _fast_config(
            aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, pq_jitter_prob=0.5,
        )
        real_cls = dual_kfold_module.EyelinerPatchDualDataset
        calls: list[dict] = []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_cls(*args, **kwargs)

        with mock.patch.object(dual_kfold_module, "EyelinerPatchDualDataset", side_effect=spy):
            run_dual_kfold(manifest_train, manifest_test, tmp_path / "base", config)

        train_calls = [c for c in calls if c.get("augment_data")]
        assert train_calls
        for c in train_calls:
            assert c["pq_jitter_rot_deg"] == 5.0
            assert c["pq_jitter_prob"] == 0.5
        # fold ごとに異なる seed（train の乱数列が fold 間で同一にならない）
        assert len({c["pq_jitter_seed"] for c in train_calls}) == len(train_calls)
        for c in calls:
            if not c.get("augment_data"):
                assert c.get("pq_jitter_rot_deg", 0.0) == 0.0

    def test_experiment_record_contains_pq_jitter(self):
        config = _fast_config(
            aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=10.0, pq_jitter_prob=0.5,
        )
        branch = BranchConfig(
            patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        )
        result = DualKfoldResult(
            fold_results=[], n_train=0, n_holdout=0,
            train_images=[], holdout_images=[], cfg_a=branch, cfg_b=branch,
        )
        hp = build_dual_experiment_record("t", config, result)["hyperparameters"]
        assert hp["pq_jitter"] is True
        assert hp["pq_jitter_rot_deg"] == 10.0
        assert hp["pq_jitter_prob"] == 0.5

    def test_record_flags_disabled_when_bit_off(self):
        config = _fast_config(aug_flags=63, pq_jitter_rot_deg=10.0)
        branch = BranchConfig(
            patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        )
        result = DualKfoldResult(
            fold_results=[], n_train=0, n_holdout=0,
            train_images=[], holdout_images=[], cfg_a=branch, cfg_b=branch,
        )
        hp = build_dual_experiment_record("t", config, result)["hyperparameters"]
        assert hp["pq_jitter"] is False

    def test_invalid_params_rejected(self, tmp_path, manifests):
        manifest_train, manifest_test = manifests
        for bad in (dict(pq_jitter_rot_deg=-1.0), dict(pq_jitter_prob=1.5), dict(pq_jitter_prob=-0.1)):
            with pytest.raises(ValueError):
                run_dual_kfold(manifest_train, manifest_test, tmp_path / "x", _fast_config(**bad))


class TestPqJitterValidation:
    """EXP035 レビュー指摘: 無音の no-op・NaN・worker 複製・aux 角度の不整合を弾く。"""

    @pytest.mark.parametrize("bad", [
        dict(aug_flags=63, pq_jitter_rot_deg=5.0),                       # ビット忘れ
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=0.0),       # 強度忘れ
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=float("nan")),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=float("inf")),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, pq_jitter_prob=float("nan")),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, num_workers=2),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, use_aux_face_shape=True),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, use_aux_face_shape=True,
             aux_features=["pq_angle_cos"]),
    ])
    def test_inconsistent_config_rejected(self, bad):
        with pytest.raises(ValueError):
            dual_kfold_module.validate_pq_jitter_config(_fast_config(**bad))

    @pytest.mark.parametrize("ok", [
        dict(),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, pq_jitter_prob=0.5),
        dict(aug_flags=63 | AUG_PQ_JITTER, pq_jitter_rot_deg=5.0, use_aux_face_shape=True,
             aux_features=["face_yaw_sin"]),
    ])
    def test_consistent_config_accepted(self, ok):
        dual_kfold_module.validate_pq_jitter_config(_fast_config(**ok))

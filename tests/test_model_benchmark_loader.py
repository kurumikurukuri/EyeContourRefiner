"""tools/model_benchmark/loader.py のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eyeliner.models.eyeliner_cnn import EyelinerCNN  # noqa: E402
from eyeliner.models.eyeliner_cnn_dual import AUX_INPUT_DIM, EyelinerCNNDual  # noqa: E402
from eyeliner.models.ensemble import EnsembleModel, save_ensemble, save_model  # noqa: E402
from eyeliner.models.dual_ensemble import save_dual_model  # noqa: E402
from tools.model_benchmark import loader  # noqa: E402


def _tiny_single() -> EyelinerCNN:
    return EyelinerCNN(num_conv_layers=2)


def _tiny_dual(use_aux_face_shape: bool = False) -> EyelinerCNNDual:
    return EyelinerCNNDual(num_conv_layers=2, use_aux_face_shape=use_aux_face_shape)


def _save_single(tmp_path: Path, name: str = "single.pt") -> Path:
    path = tmp_path / name
    save_model(_tiny_single(), "custom", path)
    return path


def _save_dual(tmp_path: Path, name: str = "dual.pt") -> Path:
    path = tmp_path / name
    save_dual_model(_tiny_dual(), "independent", {"num_conv_layers": 2}, path)
    return path


# ---------------------------------------------------------------------------
# detect_checkpoint_kind
# ---------------------------------------------------------------------------

class TestDetectCheckpointKind:
    def test_single_new(self, tmp_path: Path) -> None:
        ckpt = loader.load_checkpoint(_save_single(tmp_path))
        assert loader.detect_checkpoint_kind(ckpt) == "single_new"

    def test_ensemble(self, tmp_path: Path) -> None:
        path = tmp_path / "ens.pt"
        save_ensemble(EnsembleModel([_tiny_single()]), "custom", path)
        ckpt = loader.load_checkpoint(path)
        assert loader.detect_checkpoint_kind(ckpt) == "ensemble"

    def test_legacy(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        torch.save(_tiny_single().state_dict(), path)
        ckpt = loader.load_checkpoint(path)
        assert loader.detect_checkpoint_kind(ckpt) == "legacy"

    def test_dual(self, tmp_path: Path) -> None:
        ckpt = loader.load_checkpoint(_save_dual(tmp_path))
        assert loader.detect_checkpoint_kind(ckpt) == "dual"


# ---------------------------------------------------------------------------
# load_benchmark_model
# ---------------------------------------------------------------------------

class TestLoadBenchmarkModel:
    def test_roundtrip_single(self, tmp_path: Path) -> None:
        loaded = loader.load_benchmark_model(_save_single(tmp_path))
        assert loaded.kind == "single_new"
        assert loaded.backbone == "custom"
        assert isinstance(loaded.model, EyelinerCNN)
        assert not loaded.model.training

    def test_roundtrip_legacy(self, tmp_path: Path) -> None:
        # legacy 形式はデフォルト構成の EyelinerCNN（num_conv_layers=3）として復元される
        path = tmp_path / "legacy.pt"
        torch.save(EyelinerCNN().state_dict(), path)
        loaded = loader.load_benchmark_model(path)
        assert loaded.kind == "legacy"
        assert not loaded.model.training

    def test_roundtrip_dual_forward_smoke(self, tmp_path: Path) -> None:
        loaded = loader.load_benchmark_model(_save_dual(tmp_path))
        assert loaded.kind == "dual"
        assert loader.family_of(loaded.kind) == "dual"
        out = loaded.model(
            torch.zeros(1, 3, 64, 128), torch.zeros(1, 3, 64, 128), torch.zeros(1, 1)
        )
        assert len(out) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_benchmark_model(tmp_path / "absent.pt")

    def test_load_failure_wraps_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"backbone": "custom", "state_dict": {}}, path)
        with pytest.raises(RuntimeError, match="判定種別"):
            loader.load_benchmark_model(path)


# ---------------------------------------------------------------------------
# discover_fold_models
# ---------------------------------------------------------------------------

class TestDiscoverFoldModels:
    def test_finds_folds_and_ignores_checkpoints(self, tmp_path: Path) -> None:
        for name in (
            "eyeliner_detection_fold1.pt",
            "eyeliner_detection_fold2.pt",
            "checkpoint_fold0.pt",
        ):
            (tmp_path / name).write_bytes(b"x")
        paths = loader.discover_fold_models(tmp_path)
        assert len(paths) == 2
        assert all(p.name.startswith("eyeliner_detection_fold") for p in paths)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="fold モデルが見つかりません"):
            loader.discover_fold_models(tmp_path)

    def test_interrupted_training_hint(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint_fold0.pt").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="中断/未完"):
            loader.discover_fold_models(tmp_path)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ディレクトリが見つかりません"):
            loader.discover_fold_models(tmp_path / "absent")


# ---------------------------------------------------------------------------
# build_ensemble_module
# ---------------------------------------------------------------------------

class TestBuildEnsembleModule:
    def test_single_ensemble_forward(self) -> None:
        models = [_tiny_single(), _tiny_single()]
        ens = loader.build_ensemble_module(models, ["single_new", "legacy"], ["custom", "custom"])
        assert isinstance(ens, EnsembleModel)
        out = ens(torch.zeros(2, 3, 64, 128), torch.zeros(2, 1))
        assert out.shape == (2, 1)

    def test_dual_ensemble_forward(self) -> None:
        models = [_tiny_dual(), _tiny_dual()]
        ens = loader.build_ensemble_module(models, ["dual", "dual"], ["dual", "dual"])
        assert isinstance(ens, loader.DualBenchmarkEnsemble)
        out = ens(torch.zeros(2, 3, 64, 128), torch.zeros(2, 3, 64, 128), torch.zeros(2, 1))
        assert out.shape == (2, 1)

    def test_dual_ensemble_with_aux_forward(self) -> None:
        models = [_tiny_dual(use_aux_face_shape=True), _tiny_dual(use_aux_face_shape=True)]
        ens = loader.build_ensemble_module(models, ["dual", "dual"], ["dual", "dual"])
        out = ens(
            torch.zeros(2, 3, 64, 128),
            torch.zeros(2, 3, 64, 128),
            torch.zeros(2, 1),
            torch.zeros(2, AUX_INPUT_DIM),
        )
        assert out.shape == (2, 1)

    def test_mixed_families_raises(self) -> None:
        models = [_tiny_single(), _tiny_dual()]
        with pytest.raises(ValueError, match="ファミリが混在"):
            loader.build_ensemble_module(models, ["single_new", "dual"], ["custom", "dual"])

    def test_too_few_models_raises(self) -> None:
        with pytest.raises(ValueError, match="2 個以上"):
            loader.build_ensemble_module([_tiny_single()], ["single_new"], ["custom"])

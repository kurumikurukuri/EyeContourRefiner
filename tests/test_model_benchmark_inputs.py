"""tools/model_benchmark/inputs.py のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eyeliner.models.eyeliner_cnn import EyelinerCNN  # noqa: E402
from eyeliner.models.eyeliner_cnn_dual import AUX_INPUT_DIM, EyelinerCNNDual  # noqa: E402
from tools.model_benchmark.inputs import (  # noqa: E402
    aux_dim_of,
    build_inputs,
    build_feed_for_session,
)


def _tiny_single() -> EyelinerCNN:
    return EyelinerCNN(num_conv_layers=2)


def _tiny_dual(use_aux_face_shape: bool = False) -> EyelinerCNNDual:
    return EyelinerCNNDual(num_conv_layers=2, use_aux_face_shape=use_aux_face_shape)


class TestAuxDimOf:
    def test_single_has_no_aux(self) -> None:
        assert aux_dim_of(_tiny_single()) == 0

    def test_dual_with_aux(self) -> None:
        assert aux_dim_of(_tiny_dual(use_aux_face_shape=True)) == AUX_INPUT_DIM

    def test_dual_ensemble_recurses_members(self) -> None:
        from tools.model_benchmark.loader import DualBenchmarkEnsemble

        ens = DualBenchmarkEnsemble([
            _tiny_dual(use_aux_face_shape=True),
            _tiny_dual(use_aux_face_shape=True),
        ])
        assert aux_dim_of(ens) == AUX_INPUT_DIM


class TestBuildInputs:
    def test_single_shapes_and_feed(self) -> None:
        inputs = build_inputs(_tiny_single(), "single", 4, torch.device("cpu"), seed=0)
        assert inputs.torch_args[0].shape == (4, 3, 64, 64)
        assert inputs.torch_args[1].shape == (4, 1)
        assert set(inputs.onnx_feed) == {"patch", "is_upper"}
        assert inputs.onnx_feed["patch"].shape == (4, 3, 64, 64)

    def test_custom_patch_size(self) -> None:
        inputs = build_inputs(
            _tiny_single(), "single", 2, torch.device("cpu"), seed=0, patch_size=(64, 128)
        )
        assert inputs.torch_args[0].shape == (2, 3, 64, 128)
        assert inputs.onnx_feed["patch"].shape == (2, 3, 64, 128)

    def test_dual_without_aux(self) -> None:
        inputs = build_inputs(_tiny_dual(), "dual", 2, torch.device("cpu"), seed=0)
        assert len(inputs.torch_args) == 3
        assert set(inputs.onnx_feed) == {"patch_a", "patch_b", "is_upper"}

    def test_dual_with_aux(self) -> None:
        inputs = build_inputs(_tiny_dual(use_aux_face_shape=True), "dual", 2, torch.device("cpu"), seed=0)
        assert len(inputs.torch_args) == 4
        assert inputs.torch_args[3].shape == (2, AUX_INPUT_DIM)
        assert inputs.onnx_feed["aux_features"].shape == (2, AUX_INPUT_DIM)

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            build_inputs(_tiny_single(), "single", 0, torch.device("cpu"), seed=0)

    def test_seed_reproducibility(self) -> None:
        a = build_inputs(_tiny_single(), "single", 2, torch.device("cpu"), seed=7)
        b = build_inputs(_tiny_single(), "single", 2, torch.device("cpu"), seed=7)
        c = build_inputs(_tiny_single(), "single", 2, torch.device("cpu"), seed=8)
        assert torch.equal(a.torch_args[0], b.torch_args[0])
        assert not torch.equal(a.torch_args[0], c.torch_args[0])


class _FakeInput:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, inputs: list[_FakeInput]) -> None:
        self._inputs = inputs

    def get_inputs(self) -> list[_FakeInput]:
        return self._inputs


class TestBuildFeedForSession:
    def test_names_and_shapes(self) -> None:
        session = _FakeSession([
            _FakeInput("patch", ["batch", 3, 64, 128]),
            _FakeInput("is_upper", ["batch", 1]),
        ])
        feed = build_feed_for_session(session, 8, seed=0)
        assert set(feed) == {"patch", "is_upper"}
        assert feed["patch"].shape == (8, 3, 64, 128)
        assert feed["is_upper"].shape == (8, 1)
        assert feed["is_upper"].dtype == np.float32
        assert float(feed["is_upper"].sum()) == 0.0

    def test_static_dims_preserved(self) -> None:
        session = _FakeSession([_FakeInput("patch_a", [1, 3, 64, 128])])
        feed = build_feed_for_session(session, 4, seed=0)
        assert feed["patch_a"].shape == (1, 3, 64, 128)

    def test_empty_session_raises(self) -> None:
        with pytest.raises(ValueError, match="入力が定義されていません"):
            build_feed_for_session(_FakeSession([]), 1, seed=0)

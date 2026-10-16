"""eye_contour.training.weight_averaging（EMA / SWA / BN 統計再計算）の単体テスト。"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from eye_contour.training.weight_averaging import (
    EmaTracker,
    ema_variant_name,
    preserve_rng_state,
    recompute_bn_stats,
    run_swa,
    variant_dir,
)


def _linear(value: float) -> nn.Linear:
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(value)
    return layer


def _set_weight(layer: nn.Linear, value: float) -> None:
    with torch.no_grad():
        layer.weight.fill_(value)


class _BnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


def _forward_first(model: nn.Module, batch: tuple) -> None:
    model(batch[0])


class TestEmaVariantName:

    @pytest.mark.parametrize("decay, expected", [
        (0.999, "ema0999"),
        (0.9998, "ema09998"),
        (0.9, "ema09"),
    ])
    def test_name_drops_decimal_point(self, decay, expected):
        assert ema_variant_name(decay) == expected

    @pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_decay_outside_open_unit_interval(self, decay):
        with pytest.raises(ValueError):
            ema_variant_name(decay)


class TestVariantDir:

    def test_is_sibling_of_exp_dir_with_suffix(self, tmp_path):
        exp_dir = tmp_path / "outputs" / "base"
        assert variant_dir(exp_dir, "swa") == tmp_path / "outputs" / "base_swa"


class TestEmaTracker:

    def test_first_update_copies_then_exponential_average(self):
        model = _linear(0.0)
        tracker = EmaTracker(model, [0.9])
        _set_weight(model, 1.0)
        tracker.update(model)
        _set_weight(model, 2.0)
        tracker.update(model)

        averaged = dict(tracker.items())["ema09"]
        expected = 0.9 * 1.0 + 0.1 * 2.0
        assert torch.allclose(averaged.module.weight, torch.full((1, 2), expected))

    def test_tracks_multiple_decays_independently(self):
        model = _linear(0.0)
        tracker = EmaTracker(model, [0.5, 0.9])
        for value in (1.0, 3.0):
            _set_weight(model, value)
            tracker.update(model)

        items = dict(tracker.items())
        assert torch.allclose(items["ema05"].module.weight, torch.full((1, 2), 2.0))
        assert torch.allclose(items["ema09"].module.weight, torch.full((1, 2), 1.2))

    def test_does_not_modify_source_model(self):
        model = _linear(1.0)
        tracker = EmaTracker(model, [0.9])
        tracker.update(model)
        assert torch.allclose(model.weight, torch.full((1, 2), 1.0))

    def test_averages_bn_buffers(self):
        model = _BnNet()
        tracker = EmaTracker(model, [0.5])
        model.bn.running_mean.fill_(2.0)
        tracker.update(model)
        model.bn.running_mean.fill_(4.0)
        tracker.update(model)

        averaged = dict(tracker.items())["ema05"]
        assert torch.allclose(averaged.module.bn.running_mean, torch.full((2,), 3.0))

    def test_state_dict_round_trip_restores_average(self):
        model = _linear(0.0)
        tracker = EmaTracker(model, [0.9])
        for value in (1.0, 2.0):
            _set_weight(model, value)
            tracker.update(model)

        restored = EmaTracker(_linear(0.0), [0.9])
        restored.load_state_dict(tracker.state_dict())
        _set_weight(model, 5.0)
        tracker.update(model)
        restored.update(model)

        original = dict(tracker.items())["ema09"].module.weight
        loaded = dict(restored.items())["ema09"].module.weight
        assert torch.allclose(original, loaded)

    def test_rejects_empty_decays(self):
        with pytest.raises(ValueError):
            EmaTracker(_linear(0.0), [])


class TestRecomputeBnStats:

    def test_running_stats_match_data_statistics(self):
        model = _BnNet()
        model.bn.running_mean.fill_(100.0)
        model.bn.running_var.fill_(100.0)
        data = torch.randn(64, 2) * 3.0 + 5.0
        loader = [(data[i:i + 16],) for i in range(0, 64, 16)]

        recompute_bn_stats(model, loader, _forward_first)

        # 累積平均（momentum=None）: 等サイズバッチの平均の平均 = 全体平均
        assert torch.allclose(model.bn.running_mean, data.mean(dim=0), atol=1e-5)

    def test_running_var_matches_single_batch_variance(self):
        model = _BnNet()
        model.bn.running_var.fill_(100.0)
        data = torch.randn(64, 2) * 3.0 + 5.0

        recompute_bn_stats(model, [(data,)], _forward_first)

        assert torch.allclose(model.bn.running_var, data.var(dim=0, unbiased=True), atol=1e-4)

    def test_restores_momentum_and_eval_mode(self):
        model = _BnNet()
        model.bn.momentum = 0.3
        model.eval()
        recompute_bn_stats(model, [(torch.randn(4, 2),)], _forward_first)
        assert model.bn.momentum == 0.3
        assert not model.training

    def test_no_bn_layers_is_noop(self):
        model = _linear(1.0)
        recompute_bn_stats(model, [(torch.randn(4, 2),)], _forward_first)
        assert torch.allclose(model.weight, torch.full((1, 2), 1.0))


class TestRunSwa:

    def test_returns_equal_weight_average_of_epoch_end_weights(self):
        model = _linear(0.0)
        schedule = iter([1.0, 2.0, 6.0])

        def train_epoch(m: nn.Module, optimizer: torch.optim.Optimizer) -> None:
            _set_weight(m, next(schedule))

        result = run_swa(
            model, train_epoch, swa_epochs=3, swa_lr=1e-3, weight_decay=0.0,
            bn_loader=[], bn_forward=_forward_first,
        )
        assert torch.allclose(result.model.weight, torch.full((1, 2), 3.0))
        assert result.model is not model

    def test_optimizer_uses_constant_swa_lr(self):
        seen: list[float] = []

        def train_epoch(m: nn.Module, optimizer: torch.optim.Optimizer) -> None:
            seen.append(optimizer.param_groups[0]["lr"])

        run_swa(
            _linear(0.0), train_epoch, swa_epochs=2, swa_lr=2.5e-4, weight_decay=1e-4,
            bn_loader=[], bn_forward=_forward_first,
        )
        assert seen == [2.5e-4, 2.5e-4]

    def test_records_eval_of_raw_weights_each_epoch(self):
        model = _linear(0.0)
        schedule = iter([1.0, 2.0])

        def train_epoch(m: nn.Module, optimizer: torch.optim.Optimizer) -> None:
            _set_weight(m, next(schedule))

        result = run_swa(
            model, train_epoch, swa_epochs=2, swa_lr=1e-3, weight_decay=0.0,
            bn_loader=[], bn_forward=_forward_first,
            eval_fn=lambda m: float(m.weight.detach().mean()),
        )
        assert result.epoch_val_maes == [1.0, 2.0]

    def test_rejects_non_positive_epochs(self):
        with pytest.raises(ValueError):
            run_swa(
                _linear(0.0), lambda m, o: None, swa_epochs=0, swa_lr=1e-3,
                weight_decay=0.0, bn_loader=[], bn_forward=_forward_first,
            )


class TestPreserveRngState:

    def test_restores_python_numpy_and_torch_rng(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        with preserve_rng_state():
            random.random()
            np.random.rand()
            torch.rand(3)
        actual = (random.random(), np.random.rand(), torch.rand(1).item())

        assert actual == expected


def test_module_has_no_dependency_on_experiments():
    source = Path("eye_contour/training/weight_averaging.py").read_text(encoding="utf-8")
    assert "experiments" not in source

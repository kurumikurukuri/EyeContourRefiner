"""重み平均（EMA / SWA）と BatchNorm 統計の再計算。

いずれも学習時のみの処理で、最終成果物は元と同一構造の 1 モデル（推論コスト不変）。

- EMA: 学習ステップごとに重み（BN の running stats を含む）の指数移動平均を取る。
- SWA: 学習済みモデルから一定 lr で追加学習し、各 epoch 末の重みを等重み平均する。
  平均後の重みは BN の running stats と整合しないため、学習データで再計算する。

dual-branch モデルの forward は複数入力を取るため、torch.optim.swa_utils.update_bn
（単一入力前提）の代わりに、batch → forward を呼び出し側が渡す recompute_bn_stats を使う。
"""
from __future__ import annotations

import copy
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

BnForward = Callable[[nn.Module, Sequence], None]


def ema_variant_name(decay: float) -> str:
    """EMA decay から出力 config 名の接尾辞を作る（0.999 → "ema0999"）。"""
    if not 0.0 < decay < 1.0:
        raise ValueError(f"EMA decay は (0, 1) の範囲で指定してください: {decay}")
    return "ema" + repr(float(decay)).replace(".", "")


def variant_dir(exp_dir: Path, variant: str) -> Path:
    """派生モデル（EMA/SWA）の保存先。exp_dir と同階層の `{exp_dir.name}_{variant}`。"""
    return exp_dir.parent / f"{exp_dir.name}_{variant}"


class EmaTracker:
    """複数 decay の EMA モデルを同時に保持する。BN の running stats も平均する。"""

    def __init__(self, model: nn.Module, decays: Sequence[float]) -> None:
        if not decays:
            raise ValueError("EMA decay を 1 つ以上指定してください")
        names = [ema_variant_name(d) for d in decays]
        if len(set(names)) != len(names):
            raise ValueError(f"EMA decay が重複しています: {list(decays)}")
        self._models: dict[str, AveragedModel] = {
            name: AveragedModel(
                model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True,
            )
            for name, decay in zip(names, decays)
        }

    def update(self, model: nn.Module) -> None:
        for averaged in self._models.values():
            averaged.update_parameters(model)

    def items(self) -> Iterator[tuple[str, AveragedModel]]:
        return iter(self._models.items())

    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {name: averaged.state_dict() for name, averaged in self._models.items()}

    def load_state_dict(self, state: dict[str, dict[str, Any]]) -> None:
        missing = set(self._models) - set(state)
        if missing:
            raise KeyError(f"EMA の状態が見つかりません: {sorted(missing)}")
        for name, averaged in self._models.items():
            averaged.load_state_dict(state[name])


def _bn_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recompute_bn_stats(model: nn.Module, loader: Iterable[Sequence], forward: BnForward) -> None:
    """BN の running stats を loader 全体の累積平均で計算し直す（学習モード/momentum は復元）。"""
    layers = _bn_layers(model)
    if not layers:
        return
    momenta = [layer.momentum for layer in layers]
    was_training = model.training
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
    model.train()
    for batch in loader:
        forward(model, batch)
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
    model.train(was_training)


@dataclass(frozen=True)
class SwaResult:
    model: nn.Module
    epoch_val_maes: list[float] = field(default_factory=list)


def run_swa(
    model: nn.Module,
    train_epoch: Callable[[nn.Module, torch.optim.Optimizer], None],
    *,
    swa_epochs: int,
    swa_lr: float,
    weight_decay: float,
    bn_loader: Iterable[Sequence],
    bn_forward: BnForward,
    eval_fn: Callable[[nn.Module], float] | None = None,
) -> SwaResult:
    """model のコピーを一定 lr で swa_epochs 学習し、各 epoch 末の重みを等重み平均する。

    渡された model は変更しない。eval_fn を渡すと、平均前（その epoch 末の生の重み）の
    評価値を epoch ごとに記録する（一定 lr の追加学習自体が悪化させていないかの切り分け用）。
    """
    if swa_epochs <= 0:
        raise ValueError(f"swa_epochs は正の整数で指定してください: {swa_epochs}")
    working = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(working.parameters(), lr=swa_lr, weight_decay=weight_decay)
    averaged = AveragedModel(working)
    epoch_val_maes: list[float] = []
    for _ in range(swa_epochs):
        train_epoch(working, optimizer)
        averaged.update_parameters(working)
        if eval_fn is not None:
            epoch_val_maes.append(eval_fn(working))
    swa_model = averaged.module
    recompute_bn_stats(swa_model, bn_loader, bn_forward)
    swa_model.eval()
    return SwaResult(model=swa_model, epoch_val_maes=epoch_val_maes)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """ブロック内の乱数消費が、後続の学習（次 fold 等）の乱数列に影響しないようにする。"""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

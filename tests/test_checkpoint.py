"""eyeliner.training.checkpoint の単体テスト。

学習リジューム用のエポック単位チェックポイント保存・復元を検証する。
モデル本体は checkpoint.py の関心事ではないため、軽量な nn.Linear で代替する。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from eyeliner.training.checkpoint import (
    load_checkpoint,
    remove_checkpoint,
    save_checkpoint,
)


def _make_model_optimizer_scheduler():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=3, epochs=5,
    )
    return model, optimizer, scheduler


class TestSaveAndLoadCheckpoint:

    def test_roundtrip_preserves_scalar_fields(self, tmp_path: Path):
        model, optimizer, scheduler = _make_model_optimizer_scheduler()
        path = tmp_path / "checkpoint_fold1.pt"

        save_checkpoint(
            path, epoch=3, model=model, optimizer=optimizer, scheduler=scheduler,
            best_mae=0.1234, best_epoch=2, no_improve=1, extra={"phase_switched": True},
        )
        ckpt = load_checkpoint(path)

        assert ckpt is not None
        assert ckpt["epoch"] == 3
        assert ckpt["best_mae"] == 0.1234
        assert ckpt["best_epoch"] == 2
        assert ckpt["no_improve"] == 1
        assert ckpt["extra"] == {"phase_switched": True}

    def test_load_returns_none_when_missing(self, tmp_path: Path):
        assert load_checkpoint(tmp_path / "does_not_exist.pt") is None

    def test_extra_defaults_to_empty_dict(self, tmp_path: Path):
        model, optimizer, scheduler = _make_model_optimizer_scheduler()
        path = tmp_path / "checkpoint_fold1.pt"
        save_checkpoint(
            path, epoch=0, model=model, optimizer=optimizer, scheduler=scheduler,
            best_mae=float("inf"), best_epoch=0, no_improve=0,
        )
        ckpt = load_checkpoint(path)
        assert ckpt["extra"] == {}

    def test_model_state_dict_restores_trained_weights(self, tmp_path: Path):
        model, optimizer, scheduler = _make_model_optimizer_scheduler()
        with torch.no_grad():
            model.weight.fill_(0.5)
            model.bias.fill_(0.25)
        path = tmp_path / "checkpoint_fold1.pt"
        save_checkpoint(
            path, epoch=1, model=model, optimizer=optimizer, scheduler=scheduler,
            best_mae=0.5, best_epoch=1, no_improve=0,
        )

        fresh_model, _, _ = _make_model_optimizer_scheduler()
        ckpt = load_checkpoint(path)
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        assert torch.allclose(fresh_model.weight, model.weight)
        assert torch.allclose(fresh_model.bias, model.bias)

    def test_optimizer_state_dict_restores_step_state(self, tmp_path: Path):
        model, optimizer, scheduler = _make_model_optimizer_scheduler()
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        path = tmp_path / "checkpoint_fold1.pt"
        save_checkpoint(
            path, epoch=1, model=model, optimizer=optimizer, scheduler=scheduler,
            best_mae=0.5, best_epoch=1, no_improve=0,
        )

        fresh_model, fresh_optimizer, fresh_scheduler = _make_model_optimizer_scheduler()
        ckpt = load_checkpoint(path)
        fresh_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        fresh_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        assert fresh_scheduler.last_epoch == scheduler.last_epoch
        assert len(fresh_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"])


class TestRemoveCheckpoint:

    def test_deletes_existing_file(self, tmp_path: Path):
        model, optimizer, scheduler = _make_model_optimizer_scheduler()
        path = tmp_path / "checkpoint_fold1.pt"
        save_checkpoint(
            path, epoch=1, model=model, optimizer=optimizer, scheduler=scheduler,
            best_mae=0.5, best_epoch=1, no_improve=0,
        )
        assert path.exists()
        remove_checkpoint(path)
        assert not path.exists()

    def test_noop_when_file_missing(self, tmp_path: Path):
        remove_checkpoint(tmp_path / "does_not_exist.pt")  # 例外を投げないこと

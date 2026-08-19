"""パラメータ数・MACs 推定・ファイルサイズの計測。"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """学習可能パラメータの総数を返す。"""
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, torch_args: tuple[torch.Tensor, ...]) -> int | None:
    """thop で MACs を推定する。未インストール時や失敗時は None を返す（測定は継続）。

    計測対象モデルの deepcopy を CPU 上で profile する。thop は profile が
    途中で失敗すると hook を残留させるため、本物のモデルを汚染しないようにする。
    """
    try:
        from thop import profile
    except ImportError:
        logger.warning(
            "thop が未インストールのため MACs 推定をスキップします"
            "（`pip install thop` で有効化）"
        )
        return None
    try:
        model_copy = copy.deepcopy(model).to("cpu")
        cpu_args = tuple(t.detach().to("cpu") for t in torch_args)
        macs, _params = profile(model_copy, inputs=cpu_args, verbose=False)
        return int(macs)
    except Exception as exc:
        logger.warning("MACs 推定に失敗しました: %s", exc)
        return None


def file_size_bytes(path: Path) -> int:
    """ファイルのディスクサイズ（バイト）を返す。"""
    return path.stat().st_size


def format_bytes(size: float) -> str:
    """バイト数を人間可読形式に整形する。"""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"

"""ベンチマーク用の合成入力生成。

純粋な計算性能の測定が目的のため、実画像の読み込みや正規化は行わない
（合成テンソルで計算量は等価）。seed 固定で再現性を保つ。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tools.model_benchmark.loader import ModelFamily

# 現行dual標準・EXP032モデルの実際の学習パッチサイズ。
# eyeliner.data.patch_utils.PATCH_SIZE=(64,128) は EXP019以前の旧世代の値で、
# EXP020でdual-branch化して以降は eyeliner/training/dual_branch.py の
# branch_a_x0()/branch_b_eye_scale() プリセットが (64, 64) を使っている
# （こちらが現行の source of truth）。
DEFAULT_PATCH_SIZE: tuple[int, int] = (64, 64)


@dataclass(frozen=True)
class BenchmarkInputs:
    """torch 用位置引数と ONNX Runtime 用 feed の組。"""

    torch_args: tuple[torch.Tensor, ...]
    onnx_feed: dict[str, Any]


def aux_dim_of(model: nn.Module) -> int:
    """モデル（またはアンサンブル）の aux 入力次元を返す（無ければ 0）。"""
    members = getattr(model, "_models", None)
    if members is not None and len(members) > 0:
        return aux_dim_of(members[0])
    return int(getattr(model, "aux_dim", 0) or 0)


def build_inputs(
    model: nn.Module,
    family: ModelFamily,
    batch_size: int,
    device: torch.device,
    seed: int,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
) -> BenchmarkInputs:
    """合成入力を生成する。

    single: (patch, is_upper)
    dual:   (patch_a, patch_b, is_upper) + aux_dim>0 のとき (aux_features)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size は 1 以上である必要があります: {batch_size}")
    h, w = patch_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    patch_a = torch.randn(batch_size, 3, h, w, generator=generator)
    patch_b = torch.randn(batch_size, 3, h, w, generator=generator)
    is_upper = torch.zeros(batch_size, 1)
    aux_dim = aux_dim_of(model)

    if family == "dual":
        args: tuple[torch.Tensor, ...] = (patch_a, patch_b, is_upper)
        feed: dict[str, Any] = {
            "patch_a": patch_a.numpy(),
            "patch_b": patch_b.numpy(),
            "is_upper": is_upper.numpy(),
        }
        if aux_dim > 0:
            aux = torch.zeros(batch_size, aux_dim)
            args = args + (aux,)
            feed["aux_features"] = aux.numpy()
    else:
        args = (patch_a, is_upper)
        feed = {"patch": patch_a.numpy(), "is_upper": is_upper.numpy()}

    return BenchmarkInputs(
        torch_args=tuple(t.to(device) for t in args),
        onnx_feed=feed,
    )


def build_feed_for_session(session: Any, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    """ONNX Runtime セッションの入力定義から feed を生成する。

    入力名から意味を推定する: is_upper / aux* はゼロ埋め、それ以外は乱数。
    動的次元（batch）は batch_size で具体化する。
    """
    rng = np.random.default_rng(seed)
    feed: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = tuple(
            batch_size if (not isinstance(dim, int) or dim <= 0) else dim
            for dim in inp.shape
        )
        name = inp.name.lower()
        if "is_upper" in name or "aux" in name:
            feed[inp.name] = np.zeros(shape, dtype=np.float32)
        else:
            feed[inp.name] = rng.standard_normal(shape).astype(np.float32)
    if not feed:
        raise ValueError("セッションに入力が定義されていません")
    return feed

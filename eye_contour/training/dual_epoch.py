"""デュアルブランチ学習の 1 epoch 学習・val 評価と、派生モデル（EMA/SWA）の保存。

dual_kfold._train_single_dual_fold から抽出した処理。本学習・EMA 評価・SWA 追加学習・
BN 統計再計算が同じ forward / 損失 / 評価を共有するために切り出している。
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from eye_contour.data.dataset_dual import BranchConfig
from eye_contour.models.dual_ensemble import save_dual_model
from eye_contour.training.dual_branch import (
    branch_pred_to_raw_t,
    ensemble_raw_t,
    model_pred_raw_t,
    normalize_to_branch_a,
)

_VARIANT_FOLDS_FILENAME = "variant_folds.json"


def _inputs(batch: Sequence, aux_features: list[str], device: torch.device) -> tuple:
    """batch から (x_a, x_b, is_upper, pq_dist, pq_prime, aux) を device 上に取り出す。"""
    aux = batch[14].to(device) if aux_features else None
    return (
        batch[0].to(device), batch[1].to(device), batch[2].to(device),
        batch[12].to(device), batch[13].to(device), aux,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Sequence,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fusion_mode: str,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    aux_features: list[str],
    device: torch.device,
    on_step: Callable[[nn.Module], None] | None = None,
) -> float:
    """1 epoch 学習して平均 loss を返す。scheduler=None なら lr は一定。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x_a, x_b, is_upper, pq_dist, pq_prime, aux = _inputs(batch, aux_features, device)
        y_a = batch[4].to(device)
        y_b = batch[5].to(device)

        if fusion_mode == "attention":
            pred_a, pred_b, alpha = model(x_a, x_b, is_upper, aux)
            t_a = branch_pred_to_raw_t(pred_a, cfg_a, pq_dist, pq_prime)
            t_b = branch_pred_to_raw_t(pred_b, cfg_b, pq_dist, pq_prime)
            ens_norm = normalize_to_branch_a(ensemble_raw_t(t_a, t_b, alpha), cfg_a)
            loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b) + loss_fn(ens_norm, y_a)
        else:
            pred_a, pred_b = model(x_a, x_b, is_upper, aux)
            loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if on_step is not None:
            on_step(model)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_val_mae(
    model: nn.Module,
    loader: Sequence,
    fusion_mode: str,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    aux_features: list[str],
    device: torch.device,
) -> float:
    """val の MAE（枝 A 正規化空間、早期終了判定と同じ尺度）を返す。"""
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        x_a, x_b, is_upper, pq_dist, pq_prime, aux = _inputs(batch, aux_features, device)
        t_ens = model_pred_raw_t(
            model, x_a, x_b, is_upper, fusion_mode, cfg_a, cfg_b, pq_dist, pq_prime, aux,
        )
        all_preds.append(normalize_to_branch_a(t_ens, cfg_a).cpu())
        all_labels.append(batch[4])
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return (preds - labels).abs().mean().item()


def make_bn_forward(
    aux_features: list[str], device: torch.device,
) -> Callable[[nn.Module, Sequence], None]:
    """recompute_bn_stats 用の forward（dual モデルの複数入力を渡す）。"""
    def forward(model: nn.Module, batch: Sequence) -> None:
        x_a, x_b, is_upper, _, _, aux = _inputs(batch, aux_features, device)
        model(x_a, x_b, is_upper, aux)
    return forward


def save_variant_fold(
    out_dir: Path,
    fold_idx: int,
    model: nn.Module,
    fusion_mode: str,
    config: dict,
    info: dict,
) -> None:
    """派生モデルの fold モデルを保存し、variant_folds.json の該当 fold を更新する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dual_model(model, fusion_mode, config, out_dir / f"eyeliner_detection_fold{fold_idx}.pt")
    path = out_dir / _VARIANT_FOLDS_FILENAME
    folds = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updated = {**folds, str(fold_idx): info}
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

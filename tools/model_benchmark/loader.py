"""チェックポイントの種別判定・ロード・fold 発見・アンサンブル構築。

対応するチェックポイント形式（scripts/export_onnx.py と同じ判定ポリシー）:

    single_new: {"backbone": str, "state_dict": dict}  — ensemble.save_model
    ensemble:   {"type": "ensemble", ...}              — ensemble.save_ensemble
    legacy:     plain state_dict                       — 旧実験
    dual:       {"type": "dual_model", ...}            — dual_ensemble.save_dual_model（現行 EXP031）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CheckpointKind = Literal["single_new", "ensemble", "legacy", "dual"]
ModelFamily = Literal["single", "dual"]

FOLD_GLOB = "eyeliner_detection_fold*.pt"


class DualBenchmarkEnsemble(nn.Module):
    """dual fold 群の予測平均アンサンブル（性能測定専用）。

    実運用のアンサンブル合成（eyeliner.training.dual_branch.fold_ensemble_raw_t）は
    枝ごとの label_space を考慮した生 t 空間で行うが、その変換はモデル外のスカラー
    演算のため計算グラフ（MACs / レイテンシ）は等価。正規化空間平均で代用する。
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        is_upper: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds: list[torch.Tensor] = []
        for m in self._models:
            if aux_features is not None:
                out = m(x_a, x_b, is_upper, aux_features)
            else:
                out = m(x_a, x_b, is_upper)
            if len(out) == 3:
                pred_a, pred_b, alpha = out
                preds.append(alpha * pred_a + (1.0 - alpha) * pred_b)
            else:
                pred_a, pred_b = out
                preds.append((pred_a + pred_b) / 2.0)
        return torch.stack(preds).mean(0)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """ロード済みチェックポイントのメタ情報。"""

    model: nn.Module
    kind: CheckpointKind
    backbone: str


def load_checkpoint(path: Path) -> object:
    """torch.load する（weights_only 厳格、旧 pickle は警告付きフォールバック）。

    対象は repo 内で生成された .pt のみを想定しているため、
    weights_only=True で読めない場合は警告の上で通常読み込みにフォールバックする。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(
            "weights_only=True で読み込めませんでした。旧形式として再読み込みします: %s (%s)",
            path, exc,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def detect_checkpoint_kind(ckpt: object) -> CheckpointKind:
    """チェックポイント辞書から形式を判定する。"""
    if isinstance(ckpt, dict):
        if ckpt.get("type") == "dual_model":
            return "dual"
        if ckpt.get("type") == "ensemble":
            return "ensemble"
        if "backbone" in ckpt:
            return "single_new"
    return "legacy"


def _backbone_of(ckpt: object, kind: CheckpointKind) -> str:
    if kind == "dual":
        return "dual"
    if isinstance(ckpt, dict):
        return str(ckpt.get("backbone", "custom"))
    return "custom"


def load_benchmark_model(path: Path) -> LoadedCheckpoint:
    """チェックポイントを種別判定してモデルを復元する（eval 済みで返す）。"""
    if not path.is_file():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {path}")
    ckpt = load_checkpoint(path)
    kind = detect_checkpoint_kind(ckpt)
    try:
        if kind == "dual":
            from eyeliner.models.dual_ensemble import load_dual_model

            model = load_dual_model(path)  # eval 済み
        else:
            from eyeliner.models.ensemble import load_model

            model = load_model(path, _preloaded_ckpt=ckpt)
            model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"モデルのロードに失敗しました: {path} (判定種別: {kind})"
        ) from exc
    return LoadedCheckpoint(model=model, kind=kind, backbone=_backbone_of(ckpt, kind))


def family_of(kind: CheckpointKind) -> ModelFamily:
    """チェックポイント種別からモデルファミリを返す。"""
    return "dual" if kind == "dual" else "single"


def discover_fold_models(models_dir: Path) -> list[Path]:
    """ディレクトリ内の fold モデルを発見する（学習中断用 checkpoint_fold* は除外）。"""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {models_dir}")
    fold_paths = sorted(models_dir.glob(FOLD_GLOB))
    if fold_paths:
        return fold_paths
    hint = ""
    if list(models_dir.glob("checkpoint_fold*.pt")):
        hint = "（checkpoint_fold*.pt のみ見つかりました — 学習が中断/未完の可能性があります）"
    raise FileNotFoundError(f"fold モデルが見つかりません: {models_dir}/{FOLD_GLOB}{hint}")


def build_ensemble_module(
    models: list[nn.Module],
    kinds: list[CheckpointKind],
    backbones: list[str],
) -> nn.Module:
    """fold モデル群からベンチマーク用アンサンブルを構築する。

    single: EnsembleModel（正規化空間平均 — 実運用と同一構成）
    dual:   DualBenchmarkEnsemble（正規化空間平均 — 計算グラフは実運用と等価）
    """
    if len(models) < 2:
        raise ValueError("アンサンブル構築には 2 個以上の fold モデルが必要です")
    families = {family_of(k) for k in kinds}
    if len(families) > 1:
        raise ValueError(f"fold 間でモデルファミリが混在しています: {sorted(families)}")
    if len(set(backbones)) > 1:
        logger.warning("fold モデルのバックボーンが混在しています: %s", sorted(set(backbones)))

    if "dual" in families:
        return DualBenchmarkEnsemble(models)
    from eyeliner.models.ensemble import EnsembleModel

    return EnsembleModel(models)

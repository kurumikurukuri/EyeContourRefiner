"""Dual モデルの保存・読み込み・アンサンブルユーティリティ。

eyeliner.models.ensemble（単一ブランチ）に対応するデュアルブランチ版。
旧 experiments/EXP020_dual_branch/scr/dual_model_io.py から移植
（EXP020/EXP021 双方・推論スクリプト・WebUI から再利用するため共有化）。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from eyeliner.models.eyeliner_cnn_dual import EyelinerCNNDual
from eyeliner.models.eyeliner_cnn_dual_branch import CrossFuseDualModel


class DualEnsembleModel(nn.Module):
    """複数の dual fold モデルの予測を平均するアンサンブル（推論専用）。"""

    def __init__(
        self, models: list[nn.Module], fusion_mode: str = "independent",
    ) -> None:
        super().__init__()
        self._models = nn.ModuleList(models)
        self.fusion_mode = fusion_mode

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        is_upper: torch.Tensor,
    ) -> torch.Tensor:
        ensemble_preds: list[torch.Tensor] = []
        for m in self._models:
            if self.fusion_mode == "attention":
                pred_a, pred_b, alpha = m(x_a, x_b, is_upper)
                pred = alpha * pred_a + (1.0 - alpha) * pred_b
            else:
                pred_a, pred_b = m(x_a, x_b, is_upper)
                pred = (pred_a + pred_b) / 2.0
            ensemble_preds.append(pred)
        return torch.stack(ensemble_preds).mean(0)


def get_dual_model(fusion_mode: str, **model_kwargs) -> nn.Module:
    """fusion_mode に応じたモデルインスタンスを生成する。"""
    if fusion_mode == "crossfuse":
        return CrossFuseDualModel(**model_kwargs)
    return EyelinerCNNDual(fusion_mode=fusion_mode, **model_kwargs)


def save_dual_model(
    model: nn.Module, fusion_mode: str, config: dict, path: Path,
) -> None:
    """Dual モデルを fusion_mode 情報付きで保存する。"""
    torch.save({
        "type": "dual_model",
        "fusion_mode": fusion_mode,
        "num_conv_layers": config.get("num_conv_layers", 4),
        "use_sigmoid": config.get("use_sigmoid", True),
        "use_batchnorm": config.get("use_batchnorm", True),
        "dropout_rate": config.get("dropout_rate", 0.3),
        "base_channel": config.get("base_channel", 32),
        "pool_width": config.get("pool_width", 32),
        "use_aux_face_shape": config.get("use_aux_face_shape", False),
        "aux_dim": getattr(model, "aux_dim", 0),
        "aux_features": list(config.get("aux_features") or []),
        "use_is_upper": config.get("use_is_upper", True),
        "state_dict": model.state_dict(),
    }, path)


def load_dual_model(path: Path) -> nn.Module:
    """チェックポイントから dual モデルを復元する。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    fusion_mode = ckpt.get("fusion_mode", "independent")
    model_kwargs = dict(
        num_conv_layers=ckpt.get("num_conv_layers", 4),
        use_sigmoid=ckpt.get("use_sigmoid", True),
        use_batchnorm=ckpt.get("use_batchnorm", True),
        dropout_rate=ckpt.get("dropout_rate", 0.3),
    )
    use_aux_face_shape = bool(ckpt.get("use_aux_face_shape", False))
    if fusion_mode != "crossfuse":
        # CrossFuseDualModel は base_channel/pool_width/aux を受け付けない（未対応）。
        model_kwargs["base_channel"] = ckpt.get("base_channel", 32)
        model_kwargs["pool_width"] = ckpt.get("pool_width", 32)
        model_kwargs["use_aux_face_shape"] = use_aux_face_shape
        model_kwargs["use_is_upper"] = ckpt.get("use_is_upper", True)
        aux_dim = ckpt.get("aux_dim")
        if aux_dim is not None:
            model_kwargs["aux_dim"] = aux_dim
    model = get_dual_model(fusion_mode, **model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # 推論側が dataset に aux を渡すか知れるよう属性として保持。
    model.use_aux_face_shape = use_aux_face_shape
    model.aux_features = list(ckpt.get("aux_features") or [])
    return model


def load_dual_fold_models(
    models_dir: Path, fusion_mode: str,
) -> list[nn.Module]:
    """ディレクトリ内の fold モデルを読み込んでリストで返す。

    アンサンブル合成は呼び出し側で生 t 空間で行う（eyeliner.training.dual_branch.fold_ensemble_raw_t）。
    枝A/B が異なる label_space を持つため、正規化空間での単純平均は不正。
    """
    fold_paths = sorted(models_dir.glob("eyeliner_detection_fold*.pt"))
    if not fold_paths:
        raise FileNotFoundError(
            f"fold モデルが見つかりません: {models_dir}/eyeliner_detection_fold*.pt"
        )
    return [load_dual_model(p) for p in fold_paths]


def load_dual_ensemble(
    models_dir: Path, fusion_mode: str,
) -> DualEnsembleModel:
    """正規化空間で平均する dual アンサンブル。

    枝A(raw_t)/枝B(t_prime) が異なるラベル空間の場合、正規化値の平均は不正。
    生 t 空間での合成には load_dual_fold_models + fold_ensemble_raw_t を使うこと。
    label_space が両枝とも raw_t の構成専用。
    """
    models = load_dual_fold_models(models_dir, fusion_mode)
    return DualEnsembleModel(models, fusion_mode=fusion_mode)

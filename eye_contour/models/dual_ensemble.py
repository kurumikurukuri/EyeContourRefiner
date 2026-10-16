"""Dual モデルの保存・読み込み・アンサンブルユーティリティ。

eye_contour.models.ensemble（単一ブランチ）に対応するデュアルブランチ版。
旧 experiments/EXP020_dual_branch/scr/dual_model_io.py から移植
（EXP020/EXP021 双方・推論スクリプト・WebUI から再利用するため共有化）。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from eye_contour.models.eye_contour_cnn_dual import EyeContourCNNDual
from eye_contour.models.eye_contour_cnn_dual_branch import CrossFuseDualModel
from eye_contour.data.dataset_dual import BranchConfig
from eye_contour.data.patch_utils import IMAGENET_MEAN, IMAGENET_STD


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


class DualRawTOnnxModel(nn.Module):
    """ONNX export用の生t空間dualアンサンブル（本番推論と数値的に等価）。

    `DualEnsembleModel`（正規化空間で平均、[非推奨]）とは異なり、各枝の
    label_space（branch A="raw_t", branch B="t_prime"）に応じた変換を
    グラフ内で行ってから融合する。ロジックは
    `eye_contour.training.dual_branch.fold_ensemble_raw_t` / `branch_pred_to_raw_t`
    と同一（実装を重複させず数値一致を保証するため、この2関数をそのまま呼ぶ）。

    branch Bのt'→生t変換にはpq_dist（元のP-Q距離）とpq_prime_dist（P'-Q'距離、
    eye_scaleから計算）が必要で、これらはパッチ画像だけからは得られない。
    そのため本モデルの入力にはpatch_a/patch_b/is_upperに加え、
    px/py/qx/qy（元のPQ座標）とeye_scaleを含む——推論側はパッチを切り出す時点で
    既にこれらの値を持っているはずなので、追加の準備コストはない想定。

    外部の入力契約は学習データと同じ生[0,1]パッチのまま保つため、
    学習時と同じ ImageNet mean/std 正規化（eye_contour.data.dataset_dual /
    eye_contour.models.dual_predictor と同一）をグラフ内部で適用してから
    各枝のバックボーンに渡す。
    """

    def __init__(
        self,
        models: list[nn.Module],
        fusion_mode: str,
        cfg_a: BranchConfig,
        cfg_b: BranchConfig,
        pq_rescale_factor_b: float,
    ) -> None:
        super().__init__()
        self._models = nn.ModuleList(models)
        self.fusion_mode = fusion_mode
        self.cfg_a = cfg_a
        self.cfg_b = cfg_b
        self.pq_rescale_factor_b = pq_rescale_factor_b
        self.register_buffer("_norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
        self,
        patch_a: torch.Tensor,
        patch_b: torch.Tensor,
        is_upper: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        qx: torch.Tensor,
        qy: torch.Tensor,
        eye_scale: torch.Tensor,
    ) -> torch.Tensor:
        from eye_contour.training.dual_branch import fold_ensemble_raw_t

        patch_a = (patch_a - self._norm_mean) / self._norm_std
        patch_b = (patch_b - self._norm_mean) / self._norm_std

        pq_dist = torch.sqrt((qx - px) ** 2 + (qy - py) ** 2)
        # eye_contour.data.patch_utils.compute_pq_rescale の eye_scale 分岐と同一
        # （xxx = max(eye_scale * factor, 1.0)）。
        pq_prime_dist = torch.clamp(eye_scale * self.pq_rescale_factor_b, min=1.0)
        return fold_ensemble_raw_t(
            list(self._models), patch_a, patch_b, is_upper,
            self.fusion_mode, self.cfg_a, self.cfg_b, pq_dist, pq_prime_dist,
        )


def get_dual_model(fusion_mode: str, **model_kwargs) -> nn.Module:
    """fusion_mode に応じたモデルインスタンスを生成する。"""
    if fusion_mode == "crossfuse":
        return CrossFuseDualModel(**model_kwargs)
    return EyeContourCNNDual(fusion_mode=fusion_mode, **model_kwargs)


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

    アンサンブル合成は呼び出し側で生 t 空間で行う（eye_contour.training.dual_branch.fold_ensemble_raw_t）。
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

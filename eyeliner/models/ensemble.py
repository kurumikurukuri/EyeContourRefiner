"""EnsembleModel — 推論専用アンサンブルモデル（K fold の平均）。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """複数の fold モデルの予測値を平均するアンサンブルラッパー（推論専用）。"""

    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(
        self,
        x: torch.Tensor,
        is_upper: torch.Tensor,
        pq_ratio: torch.Tensor | None = None,
        dist_from_center: torch.Tensor | None = None,
        norm_eye_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = torch.stack([m(x, is_upper, pq_ratio, dist_from_center, norm_eye_scale) for m in self._models])
        return preds.mean(0)


def build_model(
    backbone: str,
    pretrained: bool = True,
    use_pq_ratio: bool = False,
    use_sigmoid: bool = False,
    use_batchnorm: bool = True,
    num_conv_layers: int = 3,
    use_distance_from_center: bool = False,
    use_norm_eye_scale: bool = False,
    use_is_upper: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """バックボーン名からモデルを構築する。"""
    if backbone == "custom":
        from eyeliner.models.eyeliner_cnn import EyelinerCNN
        return EyelinerCNN(use_pq_ratio=use_pq_ratio, use_sigmoid=use_sigmoid, use_batchnorm=use_batchnorm, num_conv_layers=num_conv_layers, use_distance_from_center=use_distance_from_center, use_norm_eye_scale=use_norm_eye_scale, use_is_upper=use_is_upper, dropout_rate=dropout_rate)
    if backbone == "soft_argmax":
        from eyeliner.models.soft_argmax import SoftArgmaxEyelinerCNN
        return SoftArgmaxEyelinerCNN(use_pq_ratio=use_pq_ratio, use_sigmoid=use_sigmoid, use_batchnorm=use_batchnorm, num_conv_layers=num_conv_layers, use_distance_from_center=use_distance_from_center)
    from eyeliner.models.transfer import TransferModel
    return TransferModel(backbone, pretrained=pretrained, use_pq_ratio=use_pq_ratio, use_sigmoid=use_sigmoid, use_distance_from_center=use_distance_from_center, use_norm_eye_scale=use_norm_eye_scale, use_is_upper=use_is_upper)


def load_model(path: Path, _preloaded_ckpt=None) -> nn.Module:
    """チェックポイントからモデルをロードする。新旧両形式に対応。

    新形式: {"backbone": str, "state_dict": dict}
    旧形式: plain state_dict (EyelinerCNN として扱う)

    Args:
        path:            チェックポイントファイルのパス
        _preloaded_ckpt: 既に torch.load 済みの場合はここに渡す（二重読み込み回避）
    """
    from eyeliner.models.eyeliner_cnn import EyelinerCNN

    ckpt = _preloaded_ckpt if _preloaded_ckpt is not None else torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "backbone" in ckpt and ckpt.get("type") != "ensemble":
        use_pq_ratio = ckpt.get("use_pq_ratio", False)
        use_sigmoid = ckpt.get("use_sigmoid", False)
        use_batchnorm = ckpt.get("use_batchnorm", True)
        num_conv_layers = ckpt.get("num_conv_layers", 3)
        use_distance_from_center = ckpt.get("use_distance_from_center", False)
        use_norm_eye_scale = ckpt.get("use_norm_eye_scale", False)
        use_is_upper = ckpt.get("use_is_upper", True)
        dropout_rate = ckpt.get("dropout_rate", 0.3)
        model = build_model(ckpt["backbone"], pretrained=False, use_pq_ratio=use_pq_ratio, use_sigmoid=use_sigmoid, use_batchnorm=use_batchnorm, num_conv_layers=num_conv_layers, use_distance_from_center=use_distance_from_center, use_norm_eye_scale=use_norm_eye_scale, use_is_upper=use_is_upper, dropout_rate=dropout_rate)
        model.load_state_dict(ckpt["state_dict"])
        return model

    if isinstance(ckpt, dict) and ckpt.get("type") == "ensemble":
        backbone = ckpt.get("backbone", "custom")
        n_models = ckpt.get("n_models", 5)
        use_pq_ratio = ckpt.get("use_pq_ratio", False)
        use_sigmoid = ckpt.get("use_sigmoid", False)
        use_batchnorm = ckpt.get("use_batchnorm", True)
        num_conv_layers = ckpt.get("num_conv_layers", 3)
        use_distance_from_center = ckpt.get("use_distance_from_center", False)
        use_norm_eye_scale = ckpt.get("use_norm_eye_scale", False)
        use_is_upper = ckpt.get("use_is_upper", True)
        dropout_rate = ckpt.get("dropout_rate", 0.3)
        models = [build_model(backbone, pretrained=False, use_pq_ratio=use_pq_ratio, use_sigmoid=use_sigmoid, use_batchnorm=use_batchnorm, num_conv_layers=num_conv_layers, use_distance_from_center=use_distance_from_center, use_norm_eye_scale=use_norm_eye_scale, use_is_upper=use_is_upper, dropout_rate=dropout_rate) for _ in range(n_models)]
        ensemble = EnsembleModel(models)
        ensemble.load_state_dict(ckpt["state_dict"])
        return ensemble

    # 後方互換: 旧形式 state_dict → EyelinerCNN
    model = EyelinerCNN()
    model.load_state_dict(ckpt)
    return model


def save_model(model: nn.Module, backbone: str, path: Path) -> None:
    """モデルをバックボーン情報付きで保存する。"""
    use_pq_ratio = getattr(model, "use_pq_ratio", False)
    use_sigmoid = getattr(model, "use_sigmoid", False)
    use_batchnorm = getattr(model, "use_batchnorm", True)
    num_conv_layers = getattr(model, "num_conv_layers", 3)
    use_distance_from_center = getattr(model, "use_distance_from_center", False)
    use_norm_eye_scale = getattr(model, "use_norm_eye_scale", False)
    use_is_upper = getattr(model, "use_is_upper", True)
    dropout_rate = getattr(model, "dropout_rate", 0.3)
    torch.save(
        {"backbone": backbone, "use_pq_ratio": use_pq_ratio, "use_sigmoid": use_sigmoid, "use_batchnorm": use_batchnorm, "num_conv_layers": num_conv_layers, "use_distance_from_center": use_distance_from_center, "use_norm_eye_scale": use_norm_eye_scale, "use_is_upper": use_is_upper, "dropout_rate": dropout_rate, "state_dict": model.state_dict()},
        path,
    )


def save_ensemble(ensemble: EnsembleModel, backbone: str, path: Path) -> None:
    """アンサンブルモデルを保存する。"""
    first_model = ensemble._models[0] if ensemble._models else None
    use_pq_ratio = getattr(first_model, "use_pq_ratio", False)
    use_sigmoid = getattr(first_model, "use_sigmoid", False)
    use_batchnorm = getattr(first_model, "use_batchnorm", True)
    num_conv_layers = getattr(first_model, "num_conv_layers", 3)
    use_distance_from_center = getattr(first_model, "use_distance_from_center", False)
    use_norm_eye_scale = getattr(first_model, "use_norm_eye_scale", False)
    use_is_upper = getattr(first_model, "use_is_upper", True)
    dropout_rate = getattr(first_model, "dropout_rate", 0.3)
    torch.save(
        {
            "type": "ensemble",
            "backbone": backbone,
            "use_pq_ratio": use_pq_ratio,
            "use_sigmoid": use_sigmoid,
            "use_batchnorm": use_batchnorm,
            "num_conv_layers": num_conv_layers,
            "use_distance_from_center": use_distance_from_center,
            "use_norm_eye_scale": use_norm_eye_scale,
            "use_is_upper": use_is_upper,
            "dropout_rate": dropout_rate,
            "n_models": len(ensemble._models),
            "state_dict": ensemble.state_dict(),
        },
        path,
    )


def create_ensemble_from_dir(models_dir: Path) -> tuple[EnsembleModel, str]:
    """ディレクトリ内の fold モデルを読み込んでアンサンブルを作成する。

    Returns:
        (EnsembleModel, backbone_name)
    """
    fold_paths = sorted(models_dir.glob("eyeliner_detection_fold*.pt"))
    if not fold_paths:
        raise FileNotFoundError(
            f"fold モデルが見つかりません: {models_dir}/eyeliner_detection_fold*.pt"
        )

    models = []
    backbone_names: set[str] = set()
    for p in fold_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        bn = ckpt.get("backbone", "custom") if isinstance(ckpt, dict) else "custom"
        backbone_names.add(bn)
        m = load_model(p, _preloaded_ckpt=ckpt)
        m.eval()
        models.append(m)

    if len(backbone_names) > 1:
        import warnings
        warnings.warn(
            f"fold モデルのバックボーンが混在しています: {backbone_names}"
        )

    backbone = next(iter(backbone_names))
    return EnsembleModel(models), backbone

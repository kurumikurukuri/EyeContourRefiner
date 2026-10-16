"""WebUI 向け汎用 dual ブランチ predictor ビルダー。

experiment.json の fusion_mode を見て fold モデルをロードし、
(image_rgb, px, py, qx, qy, is_upper, pq_prime) -> t_norm を返す predictor を構築する。
EXP020/EXP021 以降、dual モデルを出力するどの実験でも共通利用する
（個々の infer_exp*.py に create_predictor を実装し直す必要をなくすため）。
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable

import torch

from eye_contour.data.patch_utils import (
    IMAGENET_MEAN, IMAGENET_STD, T_MIN, T_MAX, extract_patch,
)
from eye_contour.training.dual_branch import build_branch_configs, model_pred_raw_t
from eye_contour.models.dual_ensemble import load_dual_fold_models

logger = logging.getLogger(__name__)


def build_dual_predictor(models_dir: Path) -> Callable:
    """models_dir の experiment.json から dual predictor を構築する。

    Args:
        models_dir: fold モデル (.pt) が格納されたディレクトリ。
                    同ディレクトリの experiment.json から fusion_mode と rescale_b を読む。

    Returns:
        callable: (image_rgb, px, py, qx, qy, is_upper, pq_prime=None) -> t_norm
                  t_norm は WebUI の [T_MIN, T_MAX] 正規化 float。
                  pq_prime は branch B の |P'Q'|（呼び出し側が eye_scale から計算して渡す）。

    Raises:
        FileNotFoundError: experiment.json が無い。
        ValueError: experiment.json に fusion_mode が無い（単一ブランチ実験）。
                    呼び出し側はこれを単一ブランチ読み込みへのフォールバック条件として使う。
    """
    exp_json = models_dir / "experiment.json"
    if not exp_json.exists():
        raise FileNotFoundError(f"experiment.json が見つかりません: {exp_json}")
    with open(exp_json, encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("hyperparameters", {})

    if "fusion_mode" not in hp:
        raise ValueError(
            f"デュアルブランチ実験ではありません (fusion_mode なし): {models_dir}"
        )

    fusion_mode = hp["fusion_mode"]
    rescale_b = hp.get("rescale_b", "eye_scale")

    cfg_a, cfg_b = build_branch_configs(rescale_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device).eval() for m in load_dual_fold_models(models_dir, fusion_mode)]
    logger.info(
        "dual predictor 初期化: fusion_mode=%s rescale_b=%s folds=%d device=%s models_dir=%s",
        fusion_mode, rescale_b, len(models), device, models_dir,
    )

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)

    def _to_tensor(patch):
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        return (x - mean_t) / std_t

    def predictor(image_rgb, px, py, qx, qy, is_upper, pq_prime=None):
        try:
            orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

            # Branch A: raw PQ 座標系
            patch_a = extract_patch(
                image_rgb, px, py, qx, qy,
                patch_size=cfg_a.patch_size,
                t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max,
                perp_scale=cfg_a.patch_perp_scale,
            )
            x_a = _to_tensor(patch_a)

            # Branch B: P'Q' 座標系 (pq_prime = |P'Q'|)
            pq_prime_val = float(pq_prime) if (pq_prime is not None and float(pq_prime) > 1e-6) else orig_pq
            if orig_pq > 1e-6:
                ox = (px + qx) / 2.0
                oy = (py + qy) / 2.0
                ux = (qx - px) / orig_pq
                uy = (qy - py) / orig_pq
                px_b = ox - (pq_prime_val / 2.0) * ux
                py_b = oy - (pq_prime_val / 2.0) * uy
                qx_b = ox + (pq_prime_val / 2.0) * ux
                qy_b = oy + (pq_prime_val / 2.0) * uy
            else:
                px_b, py_b, qx_b, qy_b = px, py, qx, qy

            patch_b = extract_patch(
                image_rgb, px_b, py_b, qx_b, qy_b,
                patch_size=cfg_b.patch_size,
                t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max,
                perp_scale=cfg_b.patch_perp_scale,
            )
            x_b = _to_tensor(patch_b)

            is_upper_t  = torch.tensor([[is_upper]],      dtype=torch.float32, device=device)
            pq_dist_t   = torch.tensor([[orig_pq]],       dtype=torch.float32, device=device)
            pq_prime_t  = torch.tensor([[pq_prime_val]],  dtype=torch.float32, device=device)

            with torch.no_grad():
                raw_preds = [
                    model_pred_raw_t(
                        m, x_a, x_b, is_upper_t,
                        fusion_mode, cfg_a, cfg_b,
                        pq_dist_t, pq_prime_t,
                    )
                    for m in models
                ]
            t_raw = float(torch.stack(raw_preds).mean(0).item())
            return (t_raw - T_MIN) / (T_MAX - T_MIN)
        except Exception as exc:
            logger.warning("dual predictor 推論失敗: %s", exc)
            return None

    return predictor

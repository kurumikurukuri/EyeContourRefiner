"""デュアルブランチ学習・推論: 枝プリセット + t空間変換ユーティリティ。

設計方針（完全忠実版）:
    枝A = EXP019 X0_baseline（生PQ座標系, label_space="raw_t"）
    枝B = EXP019 X1_eye_height または X2_eye_scale（P'Q'座標系, label_space="t_prime"）

各枝は EXP019 の単一ブランチ設定を忠実に再現する。枝Bは t'（P'Q'座標系の t）を
予測するため、アンサンブル前に必ず生 t へ逆変換してから合成する。

旧 experiments/EXP020_dual_branch/scr/dual_branch_utils.py から移植
（EXP020/EXP021 双方から再利用するため eyeliner/training/ へ共有化）。
"""
from __future__ import annotations

import torch

from eyeliner.data.dataset_dual import BranchConfig
from eyeliner.data.patch_utils import invert_pq_rescale_t  # noqa: F401 (re-export 用)


# ---------------------------------------------------------------------------
# 枝プリセット（EXP019 config と一致させること。tests/test_exp020_branch_presets.py で検証）
# ---------------------------------------------------------------------------

def branch_a_x0() -> BranchConfig:
    """EXP019 X0_baseline: 生PQ座標系。"""
    return BranchConfig(
        patch_t_min=-2.0, patch_t_max=3.0,
        label_t_min=-1.0, label_t_max=2.0,
        pq_rescale_mode="none", pq_rescale_factor=1.0,
        patch_perp_scale=2.0, patch_size=(64, 64),
        label_space="raw_t",
    )


def branch_b_eye_scale() -> BranchConfig:
    """EXP019 X2_eye_scale: P'Q'(eye_scale × 0.2) 座標系、t' 予測。"""
    return BranchConfig(
        patch_t_min=-0.5, patch_t_max=1.5,
        label_t_min=-0.25, label_t_max=1.25,
        pq_rescale_mode="eye_scale", pq_rescale_factor=0.2,
        patch_perp_scale=2.0, patch_size=(64, 64),
        label_space="t_prime",
    )


def branch_b_eye_height() -> BranchConfig:
    """EXP019 X1_eye_height: P'Q'(eye_height × 0.5) 座標系、t' 予測。"""
    return BranchConfig(
        patch_t_min=-0.5, patch_t_max=1.5,
        label_t_min=-0.25, label_t_max=1.25,
        pq_rescale_mode="eye_height", pq_rescale_factor=0.5,
        patch_perp_scale=4.0, patch_size=(64, 64),
        label_space="t_prime",
    )


_BRANCH_B_PRESETS = {
    "eye_scale": branch_b_eye_scale,
    "eye_height": branch_b_eye_height,
}


def build_branch_configs(rescale_b: str) -> tuple[BranchConfig, BranchConfig]:
    """rescale_b（"eye_scale" / "eye_height"）から (枝A, 枝B) の BranchConfig を返す。"""
    if rescale_b not in _BRANCH_B_PRESETS:
        raise ValueError(
            f"rescale_b は {list(_BRANCH_B_PRESETS)} のいずれか。受領: {rescale_b!r}"
        )
    return branch_a_x0(), _BRANCH_B_PRESETS[rescale_b]()


# ---------------------------------------------------------------------------
# t空間変換: 正規化予測 → 生 t（raw PQ 座標系）
# ---------------------------------------------------------------------------

def _denorm(pred_norm: torch.Tensor, label_min: float, label_max: float) -> torch.Tensor:
    """正規化予測 [0,1] を label 範囲の t（または t'）に戻す。"""
    return pred_norm * (label_max - label_min) + label_min


def branch_pred_to_raw_t(
    pred_norm: torch.Tensor,
    cfg: BranchConfig,
    pq_dist: torch.Tensor,
    pq_prime_dist: torch.Tensor,
) -> torch.Tensor:
    """枝の正規化予測を生 t（raw PQ 座標系）へ変換する。

    cfg.label_space:
        "raw_t"   → denorm した値がそのまま生 t
        "t_prime" → denorm した値は t' なので invert_pq_rescale_t で生 t に逆変換

    Args:
        pred_norm: (B, 1) 正規化予測 ∈ [0, 1]
        cfg:       その枝の BranchConfig
        pq_dist:       (B, 1) 枝A（生PQ）の |PQ|
        pq_prime_dist: (B, 1) 枝B の |P'Q'|（= eye_scale*factor 等）
    Returns:
        (B, 1) 生 t
    """
    t_val = _denorm(pred_norm, cfg.label_t_min, cfg.label_t_max)
    if cfg.label_space != "t_prime":
        return t_val
    # t' → t: 0.5 + (t' - 0.5) * pq_prime_dist / pq_dist  (pq_dist≈0 は変換不能→t' を維持)
    safe = pq_dist > 1e-6
    raw = 0.5 + (t_val - 0.5) * pq_prime_dist / torch.where(safe, pq_dist, torch.ones_like(pq_dist))
    return torch.where(safe, raw, t_val)


def ensemble_raw_t(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """生 t 空間で2枝を合成する。

    alpha=None       → 単純平均 (t_a + t_b) / 2
    alpha 指定（B,1）→ alpha * t_a + (1 - alpha) * t_b（attention 融合）
    """
    if alpha is None:
        return (t_a + t_b) / 2.0
    return alpha * t_a + (1.0 - alpha) * t_b


def normalize_to_branch_a(t_raw: torch.Tensor, cfg_a: BranchConfig) -> torch.Tensor:
    """生 t を枝A の正規化空間 [0,1] に写す。

    全予測・ラベルをこの空間で評価することで、EXP019/EXP020 X0 の正規化 MAE と
    直接比較できる（生 t MAE は * (label_t_max - label_t_min) で復元）。
    """
    return (t_raw - cfg_a.label_t_min) / (cfg_a.label_t_max - cfg_a.label_t_min)


def model_pred_raw_t(
    model,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    is_upper: torch.Tensor,
    fusion_mode: str,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    pq_dist: torch.Tensor,
    pq_prime_dist: torch.Tensor,
    aux_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """単一モデルの2枝出力を生 t 空間で合成して返す（(B,1) 生 t）。

    attention 融合のみ alpha で重み付け、それ以外は単純平均。
    各枝はそれぞれの label_space に従って生 t に逆変換される。
    aux_features は use_aux_face_shape モデルの 11次元 aux テンソル（None なら従来通り）。
    """
    if fusion_mode == "attention":
        pred_a, pred_b, alpha = model(x_a, x_b, is_upper, aux_features)
    else:
        pred_a, pred_b = model(x_a, x_b, is_upper, aux_features)
        alpha = None
    t_a = branch_pred_to_raw_t(pred_a, cfg_a, pq_dist, pq_prime_dist)
    t_b = branch_pred_to_raw_t(pred_b, cfg_b, pq_dist, pq_prime_dist)
    return ensemble_raw_t(t_a, t_b, alpha)


def fold_ensemble_raw_t(
    models,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    is_upper: torch.Tensor,
    fusion_mode: str,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    pq_dist: torch.Tensor,
    pq_prime_dist: torch.Tensor,
    aux_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """複数 fold モデルの生 t 予測を平均する（(B,1) 生 t）。"""
    preds = [
        model_pred_raw_t(m, x_a, x_b, is_upper, fusion_mode, cfg_a, cfg_b,
                         pq_dist, pq_prime_dist, aux_features)
        for m in models
    ]
    return torch.stack(preds).mean(0)

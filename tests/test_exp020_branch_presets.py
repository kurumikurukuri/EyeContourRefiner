"""EXP020 完全忠実版: 枝プリセットと t空間変換の単体テスト。

検証:
  1. 枝プリセット (X0/X1/X2) が EXP019 の対応 YAML と一致する。
  2. branch_pred_to_raw_t が raw_t / t_prime それぞれで正しく生 t を返す。
  3. dataset → 完璧予測 → branch_pred_to_raw_t の round-trip で生 t を復元する。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

_EXP020_SCR = Path(__file__).parent.parent / "experiments" / "EXP020_dual_branch" / "scr"
if str(_EXP020_SCR) not in sys.path:
    sys.path.insert(0, str(_EXP020_SCR))

_EXP019_CFG = Path(__file__).parent.parent / "experiments" / "EXP019_pq_transform" / "config"

from eyeliner.training.dual_branch import (  # noqa: E402
    branch_a_x0,
    branch_b_eye_scale,
    branch_b_eye_height,
    build_branch_configs,
    branch_pred_to_raw_t,
    ensemble_raw_t,
    normalize_to_branch_a,
    model_pred_raw_t,
    fold_ensemble_raw_t,
)
from eyeliner.data.dataset_dual import BranchConfig, EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.patch_utils import invert_pq_rescale_t  # noqa: E402
import numpy as np  # noqa: E402


# ===========================================================
# 1. プリセットが EXP019 YAML と一致
# ===========================================================

def _load_yaml(name: str) -> dict:
    with open(_EXP019_CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPresetsMatchExp019:

    def test_branch_a_matches_x0_baseline(self):
        cfg = branch_a_x0()
        y = _load_yaml("X0_baseline.yaml")
        assert cfg.patch_t_min == pytest.approx(y["patch_t_min"])
        assert cfg.patch_t_max == pytest.approx(y["patch_t_max"])
        assert cfg.patch_perp_scale == pytest.approx(y["patch_perp_scale"])
        assert cfg.label_t_min == pytest.approx(y["label_t_min"])
        assert cfg.label_t_max == pytest.approx(y["label_t_max"])
        assert (cfg.patch_size[0], cfg.patch_size[1]) == (y["patch_h"], y["patch_w"])
        # X0 は再スケールなし
        assert cfg.pq_rescale_mode == "none"
        assert cfg.label_space == "raw_t"

    def test_branch_b_eye_scale_matches_x2(self):
        cfg = branch_b_eye_scale()
        y = _load_yaml("X2_eye_scale.yaml")
        assert cfg.patch_t_min == pytest.approx(y["patch_t_min"])
        assert cfg.patch_t_max == pytest.approx(y["patch_t_max"])
        assert cfg.patch_perp_scale == pytest.approx(y["patch_perp_scale"])
        assert cfg.label_t_min == pytest.approx(y["label_t_min"])
        assert cfg.label_t_max == pytest.approx(y["label_t_max"])
        assert cfg.pq_rescale_mode == y["pq_rescale_mode"]
        assert cfg.pq_rescale_factor == pytest.approx(y["pq_rescale_factor"])
        assert cfg.label_space == "t_prime"

    def test_branch_b_eye_height_matches_x1(self):
        cfg = branch_b_eye_height()
        y = _load_yaml("X1_eye_height.yaml")
        assert cfg.patch_t_min == pytest.approx(y["patch_t_min"])
        assert cfg.patch_t_max == pytest.approx(y["patch_t_max"])
        assert cfg.patch_perp_scale == pytest.approx(y["patch_perp_scale"])
        assert cfg.label_t_min == pytest.approx(y["label_t_min"])
        assert cfg.label_t_max == pytest.approx(y["label_t_max"])
        assert cfg.pq_rescale_mode == y["pq_rescale_mode"]
        assert cfg.pq_rescale_factor == pytest.approx(y["pq_rescale_factor"])
        assert cfg.label_space == "t_prime"

    def test_build_branch_configs_dispatch(self):
        a_es, b_es = build_branch_configs("eye_scale")
        assert b_es.pq_rescale_mode == "eye_scale"
        a_eh, b_eh = build_branch_configs("eye_height")
        assert b_eh.pq_rescale_mode == "eye_height"
        # 枝A は常に X0
        assert a_es.pq_rescale_mode == "none" and a_eh.pq_rescale_mode == "none"

    def test_build_branch_configs_rejects_unknown(self):
        with pytest.raises(ValueError):
            build_branch_configs("bogus")


# ===========================================================
# 2. branch_pred_to_raw_t の変換
# ===========================================================

class TestBranchPredToRawT:

    def test_raw_t_branch_is_plain_denorm(self):
        """label_space=raw_t は denorm した値がそのまま生 t。"""
        cfg = branch_a_x0()  # label [-1, 2], range 3
        pred = torch.tensor([[0.0], [0.5], [1.0]])
        # pq_dist/pq_prime は raw_t では使われない
        out = branch_pred_to_raw_t(pred, cfg, torch.ones(3, 1), torch.ones(3, 1))
        expected = torch.tensor([[-1.0], [0.5], [2.0]])
        torch.testing.assert_close(out, expected)

    def test_t_prime_branch_inverse_transforms(self):
        """label_space=t_prime は denorm 後に invert_pq_rescale_t を通す。"""
        cfg = branch_b_eye_scale()  # label [-0.25, 1.25], range 1.5
        pred = torch.tensor([[0.5]])  # → t' = 0.5
        pq_dist = torch.tensor([[10.0]])
        pq_prime = torch.tensor([[50.0]])
        out = branch_pred_to_raw_t(pred, cfg, pq_dist, pq_prime)
        expected = invert_pq_rescale_t(0.5, 10.0, 50.0)  # = 0.5 (t'=0.5 は不動点)
        assert out.item() == pytest.approx(expected, abs=1e-6)

    def test_t_prime_extreme_recovers_raw(self):
        """t' が中央から外れた場合、pq_prime/pq_dist 倍に拡大されて生 t に戻る。"""
        cfg = branch_b_eye_scale()
        # t' = 1.25 (label 上端 → pred_norm = 1.0)
        pred = torch.tensor([[1.0]])
        pq_dist = torch.tensor([[10.0]])
        pq_prime = torch.tensor([[50.0]])
        out = branch_pred_to_raw_t(pred, cfg, pq_dist, pq_prime)
        # t = 0.5 + (1.25 - 0.5) * 50/10 = 0.5 + 0.75*5 = 4.25
        assert out.item() == pytest.approx(4.25, abs=1e-5)

    def test_t_prime_zero_pq_dist_falls_back(self):
        """pq_dist≈0 のとき t' をそのまま返す（変換不能）。"""
        cfg = branch_b_eye_scale()
        pred = torch.tensor([[1.0]])  # t' = 1.25
        out = branch_pred_to_raw_t(pred, cfg, torch.zeros(1, 1), torch.tensor([[50.0]]))
        assert out.item() == pytest.approx(1.25, abs=1e-6)


# ===========================================================
# 3. ensemble_raw_t
# ===========================================================

class TestEnsembleRawT:

    def test_mean_when_no_alpha(self):
        t_a = torch.tensor([[0.2], [1.0]])
        t_b = torch.tensor([[0.4], [2.0]])
        out = ensemble_raw_t(t_a, t_b)
        torch.testing.assert_close(out, torch.tensor([[0.3], [1.5]]))

    def test_alpha_weighting(self):
        t_a = torch.tensor([[0.0]])
        t_b = torch.tensor([[1.0]])
        alpha = torch.tensor([[0.25]])
        out = ensemble_raw_t(t_a, t_b, alpha)
        # 0.25*0 + 0.75*1 = 0.75
        assert out.item() == pytest.approx(0.75, abs=1e-6)


# ===========================================================
# 3b. model_pred_raw_t / fold_ensemble_raw_t（スタブモデル）
# ===========================================================

class _StubDual(torch.nn.Module):
    """固定の正規化予測を返すスタブ（fusion_mode 別）。"""
    def __init__(self, pa: float, pb: float, alpha: float | None = None):
        super().__init__()
        self.pa, self.pb, self.alpha = pa, pb, alpha

    def forward(self, x_a, x_b, is_upper, aux_features=None):
        B = x_a.size(0)
        pa = torch.full((B, 1), self.pa)
        pb = torch.full((B, 1), self.pb)
        if self.alpha is not None:
            return pa, pb, torch.full((B, 1), self.alpha)
        return pa, pb


class TestModelEnsembleRawT:

    def test_independent_averages_raw_t(self):
        cfg_a, cfg_b = build_branch_configs("eye_scale")
        # 枝A pred=0.5 → t_a = -1 + 0.5*3 = 0.5
        # 枝B pred=0.5 → t'=0.5 → 不動点 → t_b = 0.5
        model = _StubDual(pa=0.5, pb=0.5)
        x = torch.zeros(2, 3, 64, 64)
        out = model_pred_raw_t(
            model, x, x, torch.zeros(2, 1), "independent",
            cfg_a, cfg_b, torch.full((2, 1), 10.0), torch.full((2, 1), 50.0),
        )
        torch.testing.assert_close(out, torch.full((2, 1), 0.5))

    def test_attention_uses_alpha_in_raw_t(self):
        cfg_a, cfg_b = build_branch_configs("eye_scale")
        # 枝A pred=1.0 → t_a = -1 + 1.0*3 = 2.0
        # 枝B pred=0.5 → t_b = 0.5、alpha=0.25 → 0.25*2.0 + 0.75*0.5 = 0.875
        model = _StubDual(pa=1.0, pb=0.5, alpha=0.25)
        x = torch.zeros(1, 3, 64, 64)
        out = model_pred_raw_t(
            model, x, x, torch.zeros(1, 1), "attention",
            cfg_a, cfg_b, torch.full((1, 1), 10.0), torch.full((1, 1), 50.0),
        )
        assert out.item() == pytest.approx(0.875, abs=1e-5)

    def test_fold_ensemble_averages_models(self):
        cfg_a, cfg_b = build_branch_configs("eye_scale")
        m1 = _StubDual(pa=0.5, pb=0.5)   # → 0.5
        m2 = _StubDual(pa=1.0, pb=0.5)   # t_a=2.0, t_b=0.5 → 1.25
        x = torch.zeros(1, 3, 64, 64)
        out = fold_ensemble_raw_t(
            [m1, m2], x, x, torch.zeros(1, 1), "independent",
            cfg_a, cfg_b, torch.full((1, 1), 10.0), torch.full((1, 1), 50.0),
        )
        # fold 平均: (0.5 + 1.25)/2 = 0.875
        assert out.item() == pytest.approx(0.875, abs=1e-5)

    def test_normalize_to_branch_a(self):
        cfg_a, _ = build_branch_configs("eye_scale")  # [-1,2], range 3
        t = torch.tensor([[-1.0], [0.5], [2.0]])
        out = normalize_to_branch_a(t, cfg_a)
        torch.testing.assert_close(out, torch.tensor([[0.0], [0.5], [1.0]]))


# ===========================================================
# 3c. build_baseline_cmd が pq_rescale_factor を渡す（baseline 経由のバグ回帰）
# ===========================================================

class TestBaselineCmdPassesFactor:

    def _cfg(self, mode: str, factor: float | None) -> dict:
        cfg = {
            "loss_fn": "t", "lr": 1e-3, "weight_decay": 1e-4, "epochs": 200,
            "batch": 32, "k": 3, "patience": 50, "seed": 42,
            "patch_t_min": -0.5, "patch_t_max": 1.5, "patch_perp_scale": 2.0,
            "patch_h": 64, "patch_w": 64, "label_t_min": -0.25, "label_t_max": 1.25,
            "pq_rescale_mode": mode,
        }
        if factor is not None:
            cfg["pq_rescale_factor"] = factor
        return cfg

    def test_eye_scale_passes_factor(self):
        from train_exp020 import build_baseline_cmd
        cmd = build_baseline_cmd("X0_eye_scale_baseline", self._cfg("eye_scale", 0.2))
        assert "--pq_rescale_mode" in cmd
        i = cmd.index("--pq_rescale_factor")
        assert cmd[i + 1] == "0.2", "eye_scale baseline に factor=0.2 が渡らない（旧バグ）"

    def test_none_omits_rescale_flags(self):
        from train_exp020 import build_baseline_cmd
        cmd = build_baseline_cmd("X0_none_baseline", self._cfg("none", None))
        assert "--pq_rescale_mode" not in cmd
        assert "--pq_rescale_factor" not in cmd


# ===========================================================
# 4. round-trip: dataset → 完璧予測 → 生 t 復元
# ===========================================================

@pytest.mark.parametrize("t_raw", [0.5, -0.3, 1.86, -2.4])
def test_roundtrip_recovers_raw_t(t_raw):
    """両枝とも「完璧予測」を入れたとき branch_pred_to_raw_t が生 t を復元する。

    raw t がクランプ範囲内なら、枝A・枝B いずれの経路でも同じ生 t に戻るべき。
    """
    cfg_a, cfg_b = build_branch_configs("eye_scale")
    px, py, qx, qy, eye_scale = 100.0, 200.0, 110.0, 200.0, 150.0
    rec = {
        "image_path": "dummy.jpg",
        "px": px, "py": py, "qx": qx, "qy": qy, "t": t_raw,
        "sub_key": "upper", "point_index": 3,
        "pq_ratio": math.hypot(qx - px, qy - py) / eye_scale,
        "eye_scale": eye_scale, "weight": 1.0, "val_weight": 1.0,
    }
    ds = EyelinerPatchDualDataset([rec], cfg_a, cfg_b, augment_data=False, preload_images=False)
    ds._image_cache["dummy.jpg"] = np.zeros((300, 300, 3), dtype=np.uint8)
    ds._img_diag_map["dummy.jpg"] = math.sqrt(300 * 300)
    out = ds[0]
    y_a = out[4].unsqueeze(0)      # (1,1) 正規化済み 枝A ラベル
    y_b = out[5].unsqueeze(0)
    pq_dist = out[12].unsqueeze(0)
    pq_prime = out[13].unsqueeze(0)

    # 「完璧予測」= ラベルそのもの
    t_a = branch_pred_to_raw_t(y_a, cfg_a, pq_dist, pq_prime)
    t_b = branch_pred_to_raw_t(y_b, cfg_b, pq_dist, pq_prime)

    # 枝A はクランプ [-1, 2]、枝B はクランプ後 t' 経由。範囲内なら一致するはず。
    clamped_a = max(cfg_a.label_t_min, min(cfg_a.label_t_max, t_raw))
    assert t_a.item() == pytest.approx(clamped_a, abs=1e-4)

    # 枝B: t' のクランプ [-0.25,1.25] を逆変換した生 t。範囲内の t_raw は復元される。
    pq_d = pq_dist.item()
    pq_p = pq_prime.item()
    t_prime_unclamped = 0.5 + (t_raw - 0.5) * pq_d / pq_p
    t_prime_clamped = max(cfg_b.label_t_min, min(cfg_b.label_t_max, t_prime_unclamped))
    expected_b = invert_pq_rescale_t(t_prime_clamped, pq_d, pq_p)
    assert t_b.item() == pytest.approx(expected_b, abs=1e-4)

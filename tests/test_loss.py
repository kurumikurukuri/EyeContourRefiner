"""eye_contour/evaluation/loss.py のテスト（EXP021 で追加した損失を含む）。

EXP021 の損失 sweep は「空間 × カーネル」の 2 軸直積で構成される:

    空間＼カーネル | L1            | Huber               | Wing
    --------------|---------------|---------------------|----------
    正規化t       | t_l1          | t (huber_t_loss)    | wing
    目正規化px    | error_ratio   | error_ratio_huber   | wing_px

px 空間は残差を per-sample の scale = (label_t_max - label_t_min) * pq_ratio で
スケールしたもの。scale = 1 のとき px 空間損失は対応する t 空間損失に一致する
（直積の整合性 = これらのテストが保証する不変量）。
"""
from __future__ import annotations

import math

import pytest
import torch

from eye_contour.evaluation.loss import (
    huber_t_loss,
    l1_t_loss,
    wing_t_loss,
    error_ratio_loss,
    error_ratio_huber_loss,
    wing_px_loss,
    dead_zone_huber_loss,
    t_weighted_huber_loss,
    eye_normalized_smooth_l1_loss,
    get_loss_fn,
    LOSS_FN_REGISTRY,
)

WING_OMEGA = 0.1
WING_EPS = 0.01


def _rand_pair(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(n, 1, generator=g)
    y = torch.rand(n, 1, generator=g)
    return pred, y


# ===========================================================
# t_l1 — 正規化t × L1
# ===========================================================

class TestL1TLoss:

    def test_zero_when_perfect(self):
        y = torch.rand(10, 1)
        assert l1_t_loss(y, y).item() == pytest.approx(0.0, abs=1e-12)

    def test_equals_mean_abs_error(self):
        pred, y = _rand_pair()
        expected = (pred - y).abs().mean().item()
        assert l1_t_loss(pred, y).item() == pytest.approx(expected, rel=1e-6)

    def test_non_negative(self):
        pred, y = _rand_pair(seed=3)
        assert l1_t_loss(pred, y).item() >= 0.0

    def test_differentiable(self):
        pred, y = _rand_pair()
        pred.requires_grad_(True)
        l1_t_loss(pred, y).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ===========================================================
# wing — 正規化t × Wing
# ===========================================================

class TestWingLoss:

    def test_zero_when_perfect(self):
        y = torch.rand(10, 1)
        assert wing_t_loss(y, y, omega=WING_OMEGA, epsilon=WING_EPS).item() == pytest.approx(0.0, abs=1e-9)

    def test_continuous_at_omega(self):
        """d=omega で非線形部と線形部が連続。"""
        d = WING_OMEGA
        nonlinear = WING_OMEGA * math.log(1.0 + d / WING_EPS)
        c = WING_OMEGA - WING_OMEGA * math.log(1.0 + WING_OMEGA / WING_EPS)
        linear = d - c
        assert nonlinear == pytest.approx(linear, rel=1e-9)

    def test_amplifies_small_errors_vs_l1(self):
        """小誤差では Wing > L1（小〜中誤差を強調する設計）。"""
        pred = torch.full((1, 1), 0.01)
        y = torch.zeros(1, 1)
        wing = wing_t_loss(pred, y, omega=WING_OMEGA, epsilon=WING_EPS).item()
        l1 = l1_t_loss(pred, y).item()
        assert wing > l1

    def test_linear_tail_for_large_errors(self):
        """大誤差では L1 と一定差 C の線形（Wing - L1 ≈ -C 一定）。"""
        c = WING_OMEGA - WING_OMEGA * math.log(1.0 + WING_OMEGA / WING_EPS)
        for big in (0.5, 0.9):
            pred = torch.full((1, 1), big)
            y = torch.zeros(1, 1)
            wing = wing_t_loss(pred, y, omega=WING_OMEGA, epsilon=WING_EPS).item()
            assert wing == pytest.approx(big - c, rel=1e-6)

    def test_non_negative_and_finite(self):
        pred, y = _rand_pair(seed=5)
        loss = wing_t_loss(pred, y, omega=WING_OMEGA, epsilon=WING_EPS)
        assert loss.item() >= 0.0 and torch.isfinite(loss)

    def test_differentiable(self):
        pred, y = _rand_pair()
        pred.requires_grad_(True)
        wing_t_loss(pred, y, omega=WING_OMEGA, epsilon=WING_EPS).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ===========================================================
# error_ratio — 目正規化px × L1（ユーザ提案: |dt_raw| * pq_dist/eye_scale）
# ===========================================================

class TestErrorRatioLoss:

    def test_equals_weighted_abs(self):
        pred, y = _rand_pair()
        scale = torch.rand(64, 1) + 0.1
        expected = ((pred - y).abs() * scale).mean().item()
        assert error_ratio_loss(pred, y, scale).item() == pytest.approx(expected, rel=1e-6)

    def test_scales_linearly(self):
        pred, y = _rand_pair(seed=1)
        scale = torch.rand(64, 1) + 0.1
        base = error_ratio_loss(pred, y, scale).item()
        doubled = error_ratio_loss(pred, y, 2.0 * scale).item()
        assert doubled == pytest.approx(2.0 * base, rel=1e-6)

    def test_reduces_to_t_l1_when_scale_one(self):
        """scale=1 で error_ratio == t_l1（直積の整合性）。"""
        pred, y = _rand_pair(seed=2)
        ones = torch.ones(64, 1)
        assert error_ratio_loss(pred, y, ones).item() == pytest.approx(
            l1_t_loss(pred, y).item(), rel=1e-6
        )

    def test_differentiable(self):
        pred, y = _rand_pair()
        pred.requires_grad_(True)
        scale = torch.rand(64, 1) + 0.1
        error_ratio_loss(pred, y, scale).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ===========================================================
# error_ratio_huber — 目正規化px × Huber
# ===========================================================

class TestErrorRatioHuberLoss:

    def test_reduces_to_huber_t_when_scale_one(self):
        """scale=1 で error_ratio_huber == t(huber_t_loss)（直積の整合性）。"""
        pred, y = _rand_pair(seed=4)
        ones = torch.ones(64, 1)
        assert error_ratio_huber_loss(pred, y, ones, delta=0.05).item() == pytest.approx(
            huber_t_loss(pred, y, delta=0.05).item(), rel=1e-6
        )

    def test_quadratic_below_delta(self):
        """スケール後残差 < delta で 0.5*r^2/delta。"""
        delta = 0.05
        pred = torch.full((1, 1), 0.02)
        y = torch.zeros(1, 1)
        scale = torch.ones(1, 1)  # r = 0.02 < delta
        expected = 0.5 * 0.02**2 / delta
        assert error_ratio_huber_loss(pred, y, scale, delta=delta).item() == pytest.approx(expected, rel=1e-6)

    def test_linear_above_delta(self):
        """スケール後残差 >= delta で r - 0.5*delta。"""
        delta = 0.05
        pred = torch.full((1, 1), 0.5)
        y = torch.zeros(1, 1)
        scale = torch.ones(1, 1)  # r = 0.5 >= delta
        expected = 0.5 - 0.5 * delta
        assert error_ratio_huber_loss(pred, y, scale, delta=delta).item() == pytest.approx(expected, rel=1e-6)

    def test_scale_shifts_quadratic_linear_boundary(self):
        """scale により遷移点が px 単位で動く（r = dt*scale で評価される）。"""
        delta = 0.05
        pred = torch.full((1, 1), 0.2)
        y = torch.zeros(1, 1)
        # dt=0.2, scale=0.1 → r=0.02 < delta → quadratic
        small = error_ratio_huber_loss(pred, y, torch.full((1, 1), 0.1), delta=delta).item()
        assert small == pytest.approx(0.5 * 0.02**2 / delta, rel=1e-6)

    def test_non_negative_finite_differentiable(self):
        pred, y = _rand_pair(seed=6)
        pred.requires_grad_(True)
        scale = torch.rand(64, 1) + 0.1
        loss = error_ratio_huber_loss(pred, y, scale, delta=0.05)
        loss.backward()
        assert loss.item() >= 0.0 and torch.isfinite(loss)
        assert torch.isfinite(pred.grad).all()


# ===========================================================
# wing_px — 目正規化px × Wing
# ===========================================================

class TestWingPxLoss:

    def test_reduces_to_wing_when_scale_one(self):
        pred, y = _rand_pair(seed=7)
        ones = torch.ones(64, 1)
        assert wing_px_loss(pred, y, ones, omega=WING_OMEGA, epsilon=WING_EPS).item() == pytest.approx(
            wing_t_loss(pred, y, omega=WING_OMEGA, epsilon=WING_EPS).item(), rel=1e-6
        )

    def test_non_negative_finite_differentiable(self):
        pred, y = _rand_pair(seed=8)
        pred.requires_grad_(True)
        scale = torch.rand(64, 1) + 0.1
        loss = wing_px_loss(pred, y, scale, omega=WING_OMEGA, epsilon=WING_EPS)
        loss.backward()
        assert loss.item() >= 0.0 and torch.isfinite(loss)
        assert torch.isfinite(pred.grad).all()


# ===========================================================
# t_weighted — 正規化t × 重み付き Huber
# ===========================================================

class TestTWeightedHuberLoss:

    def test_zero_when_perfect(self):
        y = torch.rand(10, 1)
        w = torch.rand(10, 1)
        assert t_weighted_huber_loss(y, y, w).item() == pytest.approx(0.0, abs=1e-12)

    def test_weights_multiply_per_sample(self):
        """loss = mean(huber(dt_i) * w_i)（重みは正規化しない）。"""
        pred = torch.tensor([[0.02], [0.06]])
        y = torch.zeros(2, 1)
        w = torch.tensor([[2.0], [4.0]])
        delta = 0.05
        # dt=0.02 < delta -> huber = 0.5*0.02^2/delta = 0.004
        # dt=0.06 >= delta -> huber = 0.06 - 0.025 = 0.035
        expected = (0.004 * 2.0 + 0.035 * 4.0) / 2.0
        assert t_weighted_huber_loss(pred, y, w, delta=delta).item() == pytest.approx(
            expected, rel=1e-6
        )

    def test_equal_weights_reduce_to_huber_t(self):
        """重みが全て 1 のとき huber_t_loss に一致する。"""
        pred, y = _rand_pair(seed=11)
        ones = torch.ones(64, 1)
        assert t_weighted_huber_loss(pred, y, ones).item() == pytest.approx(
            huber_t_loss(pred, y).item(), rel=1e-6
        )


# ===========================================================
# dead_zone — epsilon-insensitive Huber
# ===========================================================

class TestDeadZoneHuberLoss:

    def test_zero_in_dead_zone(self):
        """|dt| < epsilon は損失 0。"""
        pred = torch.tensor([[0.005]])
        y = torch.zeros(1, 1)
        assert dead_zone_huber_loss(pred, y, epsilon=0.01, delta=0.05).item() == pytest.approx(
            0.0, abs=1e-12
        )

    def test_zero_at_dead_zone_boundary(self):
        """|dt| == epsilon でちょうど 0。"""
        pred = torch.tensor([[0.01]])
        y = torch.zeros(1, 1)
        assert dead_zone_huber_loss(pred, y, epsilon=0.01, delta=0.05).item() == pytest.approx(
            0.0, abs=1e-12
        )

    def test_quadratic_above_dead_zone(self):
        """epsilon <= |dt| < epsilon+delta は 0.5*(|dt|-eps)^2/delta。"""
        pred = torch.tensor([[0.03]])  # |dt|=0.03, shifted=0.02 < delta=0.05
        y = torch.zeros(1, 1)
        expected = 0.5 * 0.02**2 / 0.05
        assert dead_zone_huber_loss(pred, y, epsilon=0.01, delta=0.05).item() == pytest.approx(
            expected, rel=1e-6
        )

    def test_linear_above_dead_zone_plus_delta(self):
        """|dt| >= epsilon+delta は (|dt|-eps) - 0.5*delta。"""
        pred = torch.tensor([[0.10]])  # |dt|=0.10, shifted=0.09 >= 0.05
        y = torch.zeros(1, 1)
        expected = 0.09 - 0.5 * 0.05
        assert dead_zone_huber_loss(pred, y, epsilon=0.01, delta=0.05).item() == pytest.approx(
            expected, rel=1e-6
        )

    def test_negative_error_is_symmetric(self):
        """残差の符号に依らず |dt| で評価される。"""
        plus = dead_zone_huber_loss(torch.tensor([[0.10]]), torch.zeros(1, 1), epsilon=0.01, delta=0.05)
        minus = dead_zone_huber_loss(torch.tensor([[-0.10]]), torch.zeros(1, 1), epsilon=0.01, delta=0.05)
        assert plus.item() == pytest.approx(minus.item(), rel=1e-6)


# ===========================================================
# normalized_pixel — 目正規化 Smooth L1
# ===========================================================

class TestEyeNormalizedSmoothL1Loss:

    def test_zero_when_perfect(self):
        y = torch.rand(10, 1)
        w = torch.rand(10, 1)
        assert eye_normalized_smooth_l1_loss(y, y, w).item() == pytest.approx(0.0, abs=1e-12)

    def test_weight_multiplies_per_sample(self):
        """loss = mean(huber(dt_i) * weight_i)。"""
        pred = torch.tensor([[0.02], [0.06]])
        y = torch.zeros(2, 1)
        weight = torch.tensor([[3.0], [5.0]])
        delta = 0.05
        expected = (0.004 * 3.0 + 0.035 * 5.0) / 2.0
        assert eye_normalized_smooth_l1_loss(pred, y, weight, delta=delta).item() == pytest.approx(
            expected, rel=1e-6
        )


# ===========================================================
# registry
# ===========================================================

class TestRegistry:

    @pytest.mark.parametrize("name", [
        "t", "t_l1", "wing", "error_ratio", "error_ratio_huber", "wing_px",
    ])
    def test_name_registered(self, name):
        assert name in LOSS_FN_REGISTRY
        assert callable(get_loss_fn(name))

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_loss_fn("does_not_exist")

    def test_get_loss_fn_partial_kwargs(self):
        """kwargs を渡すと固定済み関数を返す。"""
        fn = get_loss_fn("wing", omega=0.2, epsilon=0.02)
        pred, y = _rand_pair(seed=9)
        # omega/epsilon 固定で呼べる（追加引数不要）
        assert torch.isfinite(fn(pred, y))


# ===========================================================
# trainer dispatch（新損失の配線。px 空間は scale=pq_ratio*label_range を使う）
# ===========================================================

class _StubModel(torch.nn.Module):
    """train_one_epoch の forward シグネチャを満たす最小モデル。"""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * 8 * 8, 1)

    def forward(self, x, is_upper, pq_ratio=None, dfc=None, nes=None, aux=None):
        return torch.sigmoid(self.lin(x.flatten(1)))


def _fake_batches(n_batches=2, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.rand(b, 3, 8, 8, generator=g)
        is_upper = (torch.rand(b, 1, generator=g) > 0.5).float()
        pq_ratio = torch.rand(b, 1, generator=g) * 0.3 + 0.05  # 0.05〜0.35
        y = torch.rand(b, 1, generator=g)
        w_train = torch.rand(b, 1, generator=g) + 0.1
        val_weight = torch.rand(b, 1, generator=g) + 0.1
        dfc = torch.rand(b, 1, generator=g)
        nes = torch.rand(b, 1, generator=g)
        batches.append((x, is_upper, pq_ratio, y, w_train, val_weight, dfc, nes))
    return batches


class TestTrainerDispatch:

    @pytest.mark.parametrize("loss_name", [
        "t", "t_l1", "wing", "error_ratio", "error_ratio_huber", "wing_px",
    ])
    def test_train_one_epoch_runs(self, loss_name):
        """各損失で 1 エポックが有限の loss/mae を返し、パラメータが更新される。"""
        from eye_contour.training.trainer import train_one_epoch

        torch.manual_seed(0)
        model = _StubModel()
        before = model.lin.weight.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        loader = _fake_batches()

        loss, mae = train_one_epoch(
            model, loader, opt, sched, loss_name, torch.device("cpu"),
            label_range=3.0,
        )
        assert math.isfinite(loss) and math.isfinite(mae)
        assert loss >= 0.0
        # 学習が進みパラメータが変化している
        assert not torch.allclose(before, model.lin.weight)

    def test_error_ratio_uses_pq_ratio_scale_not_w_train(self):
        """error_ratio は w_train(batch[4]) ではなく pq_ratio*label_range で重み付けする。

        単一バッチで train の平均 loss が手計算の error_ratio に一致することを確認する
        （オプティマイザ更新前の forward に対する値で検証するため lr=0 で固定）。
        """
        from eye_contour.training.trainer import train_one_epoch

        torch.manual_seed(0)
        model = _StubModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # 更新なし
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        batch = _fake_batches(n_batches=1, b=4, seed=3)[0]
        x, is_upper, pq_ratio, y = batch[0], batch[1], batch[2], batch[3]

        model.eval()
        with torch.no_grad():
            pred = model(x, is_upper)
        label_range = 3.0
        expected = ((pred - y).abs() * (pq_ratio * label_range)).mean().item()

        loss, _ = train_one_epoch(
            model, [batch], opt, sched, "error_ratio", torch.device("cpu"),
            label_range=label_range,
        )
        assert loss == pytest.approx(expected, rel=1e-5)

"""PQ 回転摂動 augmentation（EXP035）のテスト。

augment_pq_jitter: R を中心に P, Q を回転する。R・t・|RP|・|RQ| は不変。
"""
from __future__ import annotations

import math
import random

import pytest

from eyeliner.data.augmentation import (
    AUG_ALL,
    AUG_ENDPOINT_BLEND,
    AUG_NONE,
    AUG_PQ_JITTER,
    augment_pq_jitter,
    check_aug_flags,
)
from eyeliner.data.patch_utils import compute_t_from_point, reconstruct_point

P = (120.0, 80.0)
Q = (100.0, 90.0)
FLAGS_ON = 63 | AUG_PQ_JITTER


def _angle(px, py, qx, qy) -> float:
    return math.atan2(py - qy, px - qx)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class TestFlag:
    def test_bit_value(self):
        assert AUG_PQ_JITTER == 0x200

    def test_included_in_aug_all(self):
        assert AUG_ALL & AUG_PQ_JITTER == AUG_PQ_JITTER

    def test_no_unknown_warning_for_pq_jitter(self, caplog):
        with caplog.at_level("WARNING"):
            check_aug_flags(FLAGS_ON)
        assert "未定義ビット" not in caplog.text

    def test_not_in_default_dual_flags(self):
        """既存実験の既定 aug_flags(127) には含まれない。"""
        assert 127 & AUG_PQ_JITTER == 0


class TestIdentity:
    @pytest.mark.parametrize("flags", [AUG_NONE, 63, AUG_ENDPOINT_BLEND])
    def test_flag_off_is_identity(self, flags):
        out = augment_pq_jitter(*P, *Q, 0.4, aug_flags=flags, rot_deg=10.0, prob=1.0)
        assert out == (*P, *Q, 0.4)

    def test_zero_rot_is_identity(self):
        out = augment_pq_jitter(*P, *Q, 0.4, aug_flags=FLAGS_ON, rot_deg=0.0, prob=1.0)
        assert out == (*P, *Q, 0.4)

    def test_zero_prob_is_identity(self):
        out = augment_pq_jitter(*P, *Q, 0.4, aug_flags=FLAGS_ON, rot_deg=10.0, prob=0.0)
        assert out == (*P, *Q, 0.4)

    def test_disabled_does_not_consume_rng(self):
        rng = random.Random(0)
        state = rng.getstate()
        augment_pq_jitter(*P, *Q, 0.4, aug_flags=AUG_NONE, rot_deg=10.0, prob=1.0, rng=rng)
        augment_pq_jitter(*P, *Q, 0.4, aug_flags=FLAGS_ON, rot_deg=0.0, prob=1.0, rng=rng)
        assert rng.getstate() == state

    def test_disabled_does_not_touch_global_random(self):
        random.seed(123)
        expected = random.random()
        random.seed(123)
        augment_pq_jitter(*P, *Q, 0.4, aug_flags=FLAGS_ON, rot_deg=0.0, prob=1.0)
        assert random.random() == expected


class TestRotation:
    @pytest.mark.parametrize("t", [-0.5, 0.0, 0.3, 1.0, 1.2, 1.8])
    def test_t_and_r_invariant(self, t):
        rng = random.Random(1)
        r0 = reconstruct_point(*P, *Q, t)
        for _ in range(50):
            px, py, qx, qy, t2 = augment_pq_jitter(
                *P, *Q, t, aug_flags=FLAGS_ON, rot_deg=10.0, prob=1.0, rng=rng
            )
            assert t2 == t
            rx, ry = reconstruct_point(px, py, qx, qy, t2)
            assert (rx, ry) == pytest.approx(r0, abs=1e-9)
            # 回転後の P'Q' に R を射影し直しても同じ t
            assert compute_t_from_point(px, py, qx, qy, *r0) == pytest.approx(t, abs=1e-9)

    def test_distances_from_r_preserved(self):
        rng = random.Random(2)
        t = 0.3
        rx, ry = reconstruct_point(*P, *Q, t)
        for _ in range(20):
            px, py, qx, qy, _ = augment_pq_jitter(
                *P, *Q, t, aug_flags=FLAGS_ON, rot_deg=10.0, prob=1.0, rng=rng
            )
            assert math.hypot(px - rx, py - ry) == pytest.approx(math.hypot(P[0] - rx, P[1] - ry))
            assert math.hypot(qx - rx, qy - ry) == pytest.approx(math.hypot(Q[0] - rx, Q[1] - ry))
            assert math.hypot(px - qx, py - qy) == pytest.approx(math.hypot(P[0] - Q[0], P[1] - Q[1]))

    def test_angle_within_range_and_perturbs(self):
        rng = random.Random(3)
        base = _angle(*P, *Q)
        deltas = []
        for _ in range(200):
            px, py, qx, qy, _ = augment_pq_jitter(
                *P, *Q, 0.5, aug_flags=FLAGS_ON, rot_deg=5.0, prob=1.0, rng=rng
            )
            deltas.append(math.degrees(_wrap(_angle(px, py, qx, qy) - base)))
        assert max(abs(d) for d in deltas) <= 5.0 + 1e-9
        assert max(deltas) > 3.0 and min(deltas) < -3.0  # 両方向に十分振れる

    def test_prob_controls_firing_rate(self):
        rng = random.Random(4)
        fired = 0
        n = 2000
        for _ in range(n):
            out = augment_pq_jitter(
                *P, *Q, 0.5, aug_flags=FLAGS_ON, rot_deg=10.0, prob=0.3, rng=rng
            )
            fired += out != (*P, *Q, 0.5)
        assert 0.25 < fired / n < 0.35

    def test_same_seed_reproducible(self):
        a = augment_pq_jitter(*P, *Q, 0.5, aug_flags=FLAGS_ON, rot_deg=10.0, prob=1.0, rng=random.Random(7))
        b = augment_pq_jitter(*P, *Q, 0.5, aug_flags=FLAGS_ON, rot_deg=10.0, prob=1.0, rng=random.Random(7))
        assert a == b

    def test_negative_rot_deg_rejected(self):
        with pytest.raises(ValueError):
            augment_pq_jitter(*P, *Q, 0.5, aug_flags=FLAGS_ON, rot_deg=-1.0, prob=1.0)

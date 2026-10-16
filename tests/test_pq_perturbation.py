"""テストレコードの PQ 回転摂動（EXP035 頑健性評価）のテスト。"""
from __future__ import annotations

import math

import pytest

from eye_contour.data.patch_utils import reconstruct_point
from eye_contour.data.pq_perturbation import perturb_records_pq_rotation


def _records() -> list[dict]:
    return [
        {"idx": i, "px": 120.0 + i, "py": 80.0, "qx": 100.0, "qy": 90.0 - i, "t": 0.1 * i, "extra": "x"}
        for i in range(10)
    ]


def test_inputs_not_mutated():
    recs = _records()
    snapshot = [dict(r) for r in recs]
    perturb_records_pq_rotation(recs, rot_deg=3.0, seed=0)
    assert recs == snapshot


def test_t_and_r_preserved_and_other_fields_kept():
    recs = _records()
    out = perturb_records_pq_rotation(recs, rot_deg=3.0, seed=0)
    assert len(out) == len(recs)
    for a, b in zip(recs, out):
        assert b["t"] == a["t"] and b["idx"] == a["idx"] and b["extra"] == "x"
        ra = reconstruct_point(a["px"], a["py"], a["qx"], a["qy"], a["t"])
        rb = reconstruct_point(b["px"], b["py"], b["qx"], b["qy"], b["t"])
        assert rb == pytest.approx(ra, abs=1e-9)


def test_actually_rotated_within_range():
    recs = _records()
    out = perturb_records_pq_rotation(recs, rot_deg=3.0, seed=0)
    angles = []
    for a, b in zip(recs, out):
        ang_a = math.atan2(a["py"] - a["qy"], a["px"] - a["qx"])
        ang_b = math.atan2(b["py"] - b["qy"], b["px"] - b["qx"])
        angles.append(math.degrees(ang_b - ang_a))
    assert all(abs(x) <= 3.0 + 1e-9 for x in angles)
    assert any(abs(x) > 0.1 for x in angles)


def test_fixed_seed_is_reproducible_and_seed_matters():
    recs = _records()
    a = perturb_records_pq_rotation(recs, rot_deg=3.0, seed=0)
    b = perturb_records_pq_rotation(recs, rot_deg=3.0, seed=0)
    c = perturb_records_pq_rotation(recs, rot_deg=3.0, seed=1)
    assert a == b
    assert a != c


def test_does_not_touch_global_random():
    import random
    random.seed(5)
    expected = random.random()
    random.seed(5)
    perturb_records_pq_rotation(_records(), rot_deg=3.0, seed=0)
    assert random.random() == expected


def test_zero_rot_is_identity():
    recs = _records()
    assert perturb_records_pq_rotation(recs, rot_deg=0.0, seed=0) == recs

"""EXP023 intersect_candidates.py の単体テスト。

実験スクリプトはパッケージ外の standalone module なので importlib でファイルパスから読み込み、
pure function（join_residuals / rank_intersection）を検証する。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "EXP023_oof_mislabel" / "scr" / "intersect_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("intersect_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(idx: str, t_true: float, t_pred: float, **extra) -> dict:
    """OOF CSV の1行を模倣する dict を作る。"""
    base = {
        "idx": idx,
        "image_path": f"data/images/{idx}.jpg",
        "part_name": "eyeliner_left",
        "sub_key": "upper",
        "point_index": "0",
        "t_true": str(t_true),
        "t_pred": str(t_pred),
        "fold_idx": "1",
    }
    base.update(extra)
    return base


class TestJoinResiduals:
    def test_joins_on_common_idx_and_computes_residuals(self):
        m = _load_module()
        rows16 = {"1": _row("1", 0.8, 0.3), "2": _row("2", 0.7, 0.7)}
        rows32 = {"1": _row("1", 0.8, 0.2), "3": _row("3", 0.5, 0.5)}
        joined = m.join_residuals(rows16, rows32)

        # 共通 idx=1 のみ残る
        assert [j["idx"] for j in joined] == ["1"]
        j = joined[0]
        assert j["t_true"] == 0.8
        assert j["t_pred_16"] == 0.3
        assert j["t_pred_32"] == 0.2
        # 残差は絶対値（減算を伴うため浮動小数点誤差を許容）
        assert j["resid_16"] == pytest.approx(0.5)
        assert j["resid_32"] == pytest.approx(0.6)

    def test_t_true_taken_from_rows16(self):
        m = _load_module()
        # 仮に rows32 側の t_true が異なっていても rows16 側を採用する（本来は同一）
        rows16 = {"1": _row("1", 0.9, 0.4)}
        rows32 = {"1": _row("1", 0.0, 0.4)}
        joined = m.join_residuals(rows16, rows32)
        assert joined[0]["t_true"] == 0.9


class TestRankIntersection:
    def test_filters_by_majority_range(self):
        m = _load_module()
        joined = [
            # 多数派範囲 [0.5, 1.0) 内
            {"idx": "a", "t_true": 0.7, "t_pred_16": 0.7, "t_pred_32": 0.7, "resid_16": 0.0, "resid_32": 0.0},
            # 範囲外（< 0.5）
            {"idx": "b", "t_true": 0.2, "t_pred_16": 0.0, "t_pred_32": 0.0, "resid_16": 0.2, "resid_32": 0.2},
            # 範囲外（>= 1.0）
            {"idx": "c", "t_true": 1.1, "t_pred_16": 0.0, "t_pred_32": 0.0, "resid_16": 1.1, "resid_32": 1.1},
        ]
        ranked = m.rank_intersection(joined, 0.5, 1.0)
        assert [r["idx"] for r in ranked] == ["a"]

    def test_sorts_by_min_resid_descending(self):
        m = _load_module()
        joined = [
            {"idx": "lo", "t_true": 0.7, "t_pred_16": 0.6, "t_pred_32": 0.6, "resid_16": 0.1, "resid_32": 0.1},
            {"idx": "hi", "t_true": 0.7, "t_pred_16": 0.1, "t_pred_32": 0.2, "resid_16": 0.6, "resid_32": 0.5},
            {"idx": "mid", "t_true": 0.7, "t_pred_16": 0.2, "t_pred_32": 0.5, "resid_16": 0.5, "resid_32": 0.2},
        ]
        ranked = m.rank_intersection(joined, 0.5, 1.0)
        # min_resid: hi=0.5, mid=0.2, lo=0.1
        assert [r["idx"] for r in ranked] == ["hi", "mid", "lo"]
        assert ranked[0]["min_resid"] == 0.5

    def test_pred_diff_is_absolute_model_prediction_gap(self):
        m = _load_module()
        joined = [
            {"idx": "a", "t_true": 0.7, "t_pred_16": 0.3, "t_pred_32": 0.9, "resid_16": 0.4, "resid_32": 0.2},
        ]
        ranked = m.rank_intersection(joined, 0.5, 1.0)
        assert ranked[0]["pred_diff"] == pytest.approx(0.6)

    def test_sign_agree_one_when_both_models_err_same_direction(self):
        """誤ラベルの典型: 両モデルとも t_true より低く（同じ方向に）予測する。"""
        m = _load_module()
        joined = [
            {"idx": "a", "t_true": 0.8, "t_pred_16": 0.3, "t_pred_32": 0.2, "resid_16": 0.5, "resid_32": 0.6},
        ]
        ranked = m.rank_intersection(joined, 0.5, 1.0)
        assert ranked[0]["sign_agree"] == 1

    def test_sign_agree_zero_when_models_err_opposite_directions(self):
        m = _load_module()
        joined = [
            {"idx": "a", "t_true": 0.5, "t_pred_16": 0.3, "t_pred_32": 0.7, "resid_16": 0.2, "resid_32": 0.2},
        ]
        ranked = m.rank_intersection(joined, 0.5, 1.0)
        assert ranked[0]["sign_agree"] == 0

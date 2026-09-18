"""実験 outputs のディレクトリ構成に対する評価スクリプトの探索関数テスト。

現在は全 EXP とも outputs/{config}/ に平置きで成果物（predictions CSV / experiment.json）
を出力する。旧学習ドライバは outputs/{config}/{config}/ に出力しており、
当時の未整理の成果物が残っている場合に備えたフォールバックが _common.py に残っている。
compute_passrate / paired_ttest / generate_summary が両方の構成を扱えることを保証する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _common import find_predictions_csv  # noqa: E402
from compute_passrate import collect_prediction_files  # noqa: E402
from generate_summary import load_experiment_json  # noqa: E402
from paired_ttest import list_configs  # noqa: E402


def _write(path: Path, content: str = "idx,t_true,t_pred\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    return tmp_path / "EXP999_layout"


class TestFindPredictionsCsv:
    def test_flat_layout(self, exp_dir):
        csv_path = _write(exp_dir / "outputs" / "cfg" / "test_predictions.csv")
        assert find_predictions_csv(exp_dir, "cfg") == csv_path

    def test_nested_layout(self, exp_dir):
        csv_path = _write(exp_dir / "outputs" / "cfg" / "cfg" / "test_predictions.csv")
        assert find_predictions_csv(exp_dir, "cfg") == csv_path

    def test_flat_directory_is_searched_before_nested(self, exp_dir):
        flat = _write(exp_dir / "outputs" / "cfg" / "test_predictions.csv")
        _write(exp_dir / "outputs" / "cfg" / "cfg" / "predictions.csv")
        assert find_predictions_csv(exp_dir, "cfg") == flat

    def test_predictions_csv_preferred_within_same_directory(self, exp_dir):
        preferred = _write(exp_dir / "outputs" / "cfg" / "cfg" / "predictions.csv")
        _write(exp_dir / "outputs" / "cfg" / "cfg" / "test_predictions.csv")
        assert find_predictions_csv(exp_dir, "cfg") == preferred

    def test_train_predictions_is_not_used(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg" / "cfg" / "train_predictions.csv")
        with pytest.raises(FileNotFoundError):
            find_predictions_csv(exp_dir, "cfg")

    def test_differently_named_subdirectory_is_not_searched(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg" / "other" / "test_predictions.csv")
        with pytest.raises(FileNotFoundError):
            find_predictions_csv(exp_dir, "cfg")

    def test_error_message_lists_both_directories(self, exp_dir):
        (exp_dir / "outputs" / "cfg").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match=r"cfg[\\/]+cfg"):
            find_predictions_csv(exp_dir, "cfg")


class TestCollectPredictionFiles:
    def test_mixed_layouts_are_collected(self, exp_dir):
        flat = _write(exp_dir / "outputs" / "cfg_a" / "test_predictions.csv")
        nested = _write(exp_dir / "outputs" / "cfg_b" / "cfg_b" / "test_predictions.csv")
        assert collect_prediction_files(exp_dir) == [("cfg_a", flat), ("cfg_b", nested)]

    def test_config_filter_applies_to_nested_layout(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg_a" / "cfg_a" / "test_predictions.csv")
        nested_b = _write(exp_dir / "outputs" / "cfg_b" / "cfg_b" / "test_predictions.csv")
        assert collect_prediction_files(exp_dir, ["cfg_b"]) == [("cfg_b", nested_b)]

    def test_directory_without_predictions_is_skipped(self, exp_dir):
        (exp_dir / "outputs" / "empty_cfg").mkdir(parents=True)
        assert collect_prediction_files(exp_dir) == []


class TestListConfigs:
    def test_nested_and_flat_configs_are_listed(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg_a" / "test_predictions.csv")
        _write(exp_dir / "outputs" / "cfg_b" / "cfg_b" / "test_predictions.csv")
        (exp_dir / "outputs" / "no_predictions").mkdir(parents=True)
        assert list_configs(exp_dir) == ["cfg_a", "cfg_b"]


class TestLoadExperimentJson:
    def test_nested_layout(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg" / "cfg" / "experiment.json", json.dumps({"layout": "nested"}))
        assert load_experiment_json(exp_dir, "cfg") == {"layout": "nested"}

    def test_flat_directory_is_searched_before_nested(self, exp_dir):
        _write(exp_dir / "outputs" / "cfg" / "experiment.json", json.dumps({"layout": "flat"}))
        _write(exp_dir / "outputs" / "cfg" / "cfg" / "experiment.json", json.dumps({"layout": "nested"}))
        assert load_experiment_json(exp_dir, "cfg") == {"layout": "flat"}

    def test_missing_returns_none(self, exp_dir):
        (exp_dir / "outputs" / "cfg").mkdir(parents=True)
        assert load_experiment_json(exp_dir, "cfg") is None

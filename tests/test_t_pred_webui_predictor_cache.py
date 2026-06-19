"""tools/annotation_support/t_pred_webui/predictor_cache.py の get_predictor テスト。

eye_crop_renderer.make_predictor は重いモデルロードを伴うためモック化する。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.annotation_support.t_pred_webui import predictor_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """各テスト前後でキャッシュ状態をリセットし、テスト間の汚染を防ぐ。"""
    predictor_cache._reset_for_test()
    yield
    predictor_cache._reset_for_test()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def fake_model_dir(tmp_path) -> Path:
    """プロジェクトルート配下に experiment.json を持つ仮モデルディレクトリを作る。"""
    model_dir = _project_root() / "_tmp_test_model_dir_predictor_cache"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "experiment.json").write_text("{}", encoding="utf-8")
    yield model_dir
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)


class TestGetPredictor:
    def test_loads_and_returns_predictor(self, fake_model_dir):
        sentinel = object()
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", return_value=sentinel) as mock_make:
            result = predictor_cache.get_predictor(str(fake_model_dir))
        assert result is sentinel
        mock_make.assert_called_once()

    def test_caches_result_no_duplicate_load(self, fake_model_dir):
        sentinel = object()
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", return_value=sentinel) as mock_make:
            predictor_cache.get_predictor(str(fake_model_dir))
            predictor_cache.get_predictor(str(fake_model_dir))
        assert mock_make.call_count == 1

    def test_missing_experiment_json_returns_none(self, tmp_path):
        model_dir = _project_root() / "_tmp_test_model_dir_no_exp_json"
        model_dir.mkdir(exist_ok=True)
        try:
            with patch.object(predictor_cache.eye_crop_renderer, "make_predictor") as mock_make:
                result = predictor_cache.get_predictor(str(model_dir))
            assert result is None
            mock_make.assert_not_called()
        finally:
            model_dir.rmdir()

    def test_nonexistent_dir_returns_none(self, tmp_path):
        nonexistent = _project_root() / "_tmp_test_model_dir_does_not_exist"
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor") as mock_make:
            result = predictor_cache.get_predictor(str(nonexistent))
        assert result is None
        mock_make.assert_not_called()

    def test_path_traversal_outside_project_root_returns_none(self, tmp_path):
        """プロジェクトルート外のパスは None を返す（パストラバーサル対策）。"""
        outside_dir = tmp_path / "outside_model"
        outside_dir.mkdir()
        (outside_dir / "experiment.json").write_text("{}", encoding="utf-8")
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor") as mock_make:
            result = predictor_cache.get_predictor(str(outside_dir))
        assert result is None
        mock_make.assert_not_called()

    def test_make_predictor_exception_returns_none(self, fake_model_dir):
        """make_predictor が例外を出した場合（壊れた experiment.json 等）は None を返す。"""
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", side_effect=RuntimeError("boom")):
            result = predictor_cache.get_predictor(str(fake_model_dir))
        assert result is None

    def test_lru_eviction_when_exceeding_max_size(self, tmp_path):
        """上限件数(3)を超えると最古のエントリが破棄され、再アクセス時に再ロードされる。"""
        dirs = []
        for i in range(4):
            d = _project_root() / f"_tmp_test_model_dir_lru_{i}"
            d.mkdir(exist_ok=True)
            (d / "experiment.json").write_text("{}", encoding="utf-8")
            dirs.append(d)
        try:
            with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", side_effect=lambda d, device="cpu": object()) as mock_make:
                for d in dirs:
                    predictor_cache.get_predictor(str(d))
                assert mock_make.call_count == 4
                # 最初の dirs[0] は破棄されているはずなので再アクセスで再ロードされる
                predictor_cache.get_predictor(str(dirs[0]))
                assert mock_make.call_count == 5
        finally:
            import shutil
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_concurrent_access_same_key_loads_once(self, fake_model_dir):
        """同一キーへの同時アクセスはロックで直列化され、ロードは1回のみ。"""
        call_count = {"n": 0}
        lock_inside = threading.Lock()

        def slow_make_predictor(model_dir, device="cpu"):
            with lock_inside:
                call_count["n"] += 1
            time.sleep(0.05)
            return object()

        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", side_effect=slow_make_predictor):
            threads = [
                threading.Thread(target=predictor_cache.get_predictor, args=(str(fake_model_dir),))
                for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert call_count["n"] == 1

    def test_different_device_same_dir_treated_as_distinct_or_same(self, fake_model_dir):
        """device パラメータを渡しても例外なく動作する。"""
        with patch.object(predictor_cache.eye_crop_renderer, "make_predictor", return_value=object()):
            result = predictor_cache.get_predictor(str(fake_model_dir), device="cpu")
        assert result is not None

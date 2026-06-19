"""デュアルブランチモデル読み込みバグの TDD テスト。

バグの詳細:
  select_models で exp_native_predictor が取得できているにもかかわらず、
  _reload_models に dual ckpt の paths を渡してしまい、EyelinerCNN として
  読み込もうとして全て失敗 → `if not loaded:` ブロックで
  _active_predictor = None にリセットされる。

修正後に保証すること:
  1. _reload_models([], predictor=mock, infer_mode="exp_native") 後に
     _active_predictor が mock であること（空パスは既存動作で OK）
  2. _reload_models(fake_dual_paths, predictor=mock, infer_mode="exp_native") でも
     _active_predictor が mock であること（現状 NG → 修正後 OK）
  3. _reload_models([], predictor=mock, infer_mode="exp_native",
     predictor_dir=Path("some/dir")) 後に _active_predictor_dir がそのパスであること
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

import annotator.server as server


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _reset_globals() -> None:
    """テスト後にグローバル状態をクリアする。"""
    server._reload_models([])


def _make_mock_predictor():
    """ダミー predictor（callable）。"""
    mock = MagicMock(name="dual_predictor")
    return mock


# ---------------------------------------------------------------------------
# テスト 1: 空パスで predictor が保持されること（既存動作の回帰テスト）
# ---------------------------------------------------------------------------

class TestReloadModelsEmptyPaths:
    def teardown_method(self):
        _reset_globals()

    def test_empty_paths_preserves_predictor(self):
        """_reload_models([], predictor=mock) 後に _active_predictor == mock。"""
        mock = _make_mock_predictor()

        server._reload_models([], predictor=mock, infer_mode="exp_native")

        assert server._active_predictor is mock, (
            f"_active_predictor should be mock, got {server._active_predictor}"
        )

    def test_empty_paths_sets_infer_mode(self):
        """_reload_models([], infer_mode='exp_native') 後に _active_infer_mode == 'exp_native'。"""
        mock = _make_mock_predictor()

        server._reload_models([], predictor=mock, infer_mode="exp_native")

        assert server._active_infer_mode == "exp_native"


# ---------------------------------------------------------------------------
# テスト 2: dual ckpt パスで predictor が破棄されないこと（修正対象バグ）
# ---------------------------------------------------------------------------

class TestReloadModelsWithFakeDualPaths:
    """KEY TEST: dual ckpt を paths に渡しても predictor が保持されるか。

    現状: EyelinerCNN として読み込もうとして全部失敗 →
          `if not loaded:` ブロックで _active_predictor = None にリセット。
    修正後: infer_mode == "exp_native" のとき paths をスキップし predictor を保持。
    """

    def teardown_method(self):
        _reset_globals()

    def test_fake_dual_paths_preserves_predictor(self, tmp_path):
        """fake な .pt ファイルを渡しても predictor が保持される。"""
        # 実在しないが pathlib.Path として存在するダミーファイルを作成
        fake_pt = tmp_path / "eyeliner_detection_fold0.pt"
        fake_pt.write_bytes(b"not a real model")  # 読み込もうとすれば必ず失敗する内容

        mock = _make_mock_predictor()
        paths: List[Path] = [fake_pt]

        server._reload_models(paths, predictor=mock, infer_mode="exp_native")

        assert server._active_predictor is mock, (
            "_active_predictor が None にリセットされた。"
            "infer_mode='exp_native' のとき paths を無視して predictor を保持すべき。"
        )

    def test_fake_dual_paths_sets_infer_mode(self, tmp_path):
        """fake な .pt ファイルを渡しても infer_mode が exp_native に保たれる。"""
        fake_pt = tmp_path / "eyeliner_detection_fold1.pt"
        fake_pt.write_bytes(b"not a real model")

        mock = _make_mock_predictor()

        server._reload_models([fake_pt], predictor=mock, infer_mode="exp_native")

        assert server._active_infer_mode == "exp_native", (
            f"_active_infer_mode should be 'exp_native', got '{server._active_infer_mode}'"
        )

    def test_default_mode_still_resets_on_load_failure(self, tmp_path):
        """infer_mode='default' では従来通り失敗時に None にリセットされること（回帰防止）。"""
        fake_pt = tmp_path / "eyeliner_detection_fold0.pt"
        fake_pt.write_bytes(b"not a real model")

        server._reload_models([fake_pt], predictor=None, infer_mode="default")

        # default モードでは失敗時に None になるはず（従来動作）
        assert server._active_predictor is None


# ---------------------------------------------------------------------------
# テスト 3: predictor_dir が _active_predictor_dir に反映されること
# ---------------------------------------------------------------------------

class TestReloadModelsPreictorDir:
    def teardown_method(self):
        _reset_globals()

    def test_predictor_dir_stored_when_exp_native(self):
        """predictor_dir 引数が _active_predictor_dir に設定される。"""
        mock = _make_mock_predictor()
        expected_dir = Path("some/dir")

        server._reload_models(
            [],
            predictor=mock,
            infer_mode="exp_native",
            predictor_dir=expected_dir,
        )

        assert server._active_predictor_dir == expected_dir, (
            f"_active_predictor_dir should be {expected_dir}, "
            f"got {server._active_predictor_dir}"
        )

    def test_predictor_dir_cleared_when_reset(self):
        """_reload_models([]) でリセットしたとき _active_predictor_dir が None になる。"""
        mock = _make_mock_predictor()
        server._reload_models(
            [],
            predictor=mock,
            infer_mode="exp_native",
            predictor_dir=Path("some/dir"),
        )

        # リセット
        server._reload_models([])

        assert server._active_predictor_dir is None, (
            f"_active_predictor_dir should be None after reset, "
            f"got {server._active_predictor_dir}"
        )


# ---------------------------------------------------------------------------
# テスト 4: get_model_folders で _active_predictor_dir が active_dir として返る
# ---------------------------------------------------------------------------

class TestGetModelFoldersActiveDir:
    def teardown_method(self):
        _reset_globals()

    def test_active_dir_from_predictor_dir_when_no_model_paths(self, monkeypatch):
        """_active_model_paths が空でも _active_predictor_dir があれば active_dir に反映される。"""
        mock = _make_mock_predictor()
        # experiments/ 配下のディレクトリを設定する
        target_dir = server._PROJECT_ROOT / "experiments" / "EXP020_dual_branch" / "outputs" / "X0_baseline"

        server._reload_models(
            [],
            predictor=mock,
            infer_mode="exp_native",
            predictor_dir=target_dir,
        )

        result = server.get_model_folders()

        expected_active = str(target_dir.relative_to(server._PROJECT_ROOT))
        assert result["active_dir"] == expected_active, (
            f"active_dir should be '{expected_active}', got '{result['active_dir']}'"
        )

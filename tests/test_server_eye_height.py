"""eye_height モード修正の TDD テスト (RED → GREEN)。

修正対象:
  1. get_eye_height_for_eye: DB から eye_height を算出する関数
  2. _get_pq_prime "eye_height": DB クエリで正しい pq_prime を返す
  3. _predict_t 標準パス: pq_prime != orig_pq のとき P'Q' 座標でパッチ抽出する
  4. _predict_t: _infer_patch_perp_scale が extract_patch の perp_scale に渡される
  5. _load_exp_params: patch_perp_scale を experiment.json から読む
  6. _reload_models: patch_perp_scale が _infer_patch_perp_scale に反映される
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import annotator.db as db_module
import annotator.server as server


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _reset_server_globals() -> None:
    server._reload_models([])


def _make_minimal_db(tmp_path: Path, rows: list[dict]) -> Path:
    """最小限の samples テーブルを持つ SQLite DB を作成して返す。"""
    db_path = tmp_path / "samples.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """CREATE TABLE samples (
                idx INTEGER PRIMARY KEY,
                image_path TEXT NOT NULL,
                face_index INTEGER NOT NULL DEFAULT 0,
                part_name TEXT NOT NULL,
                sub_key TEXT NOT NULL,
                point_index INTEGER NOT NULL,
                px REAL, py REAL, qx REAL, qy REAL,
                t REAL, skip_flag INTEGER DEFAULT 0
            )"""
        )
        conn.executemany(
            "INSERT INTO samples (image_path, face_index, part_name, sub_key, point_index, px, py, qx, qy)"
            " VALUES (:image_path, :face_index, :part_name, :sub_key, :point_index, :px, :py, :qx, :qy)",
            rows,
        )
        conn.commit()
    return db_path


# ---------------------------------------------------------------------------
# テスト 1: get_eye_height_for_eye
# ---------------------------------------------------------------------------

class TestGetEyeHeightForEye:
    def test_basic_eye_height(self, tmp_path):
        """3 点の P/Q y 座標から eye_height を正しく算出する。"""
        rows = [
            {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 0, "px": 10, "py": 100, "qx": 20, "qy": 80},
            {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 1, "px": 30, "py": 90, "qx": 40, "qy": 70},
            {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 2, "px": 50, "py": 85, "qx": 60, "qy": 110},
        ]
        db_path = _make_minimal_db(tmp_path, rows)
        # ys = [100, 80, 90, 70, 85, 110] → max=110, min=70 → height=40
        result = db_module.get_eye_height_for_eye("img.jpg", 0, "eyeliner_right", db_path)
        assert result == pytest.approx(40.0)

    def test_returns_none_when_no_rows(self, tmp_path):
        """対応する行がない場合は None を返す。"""
        db_path = _make_minimal_db(tmp_path, [])
        result = db_module.get_eye_height_for_eye("img.jpg", 0, "eyeliner_right", db_path)
        assert result is None

    def test_minimum_value_is_1(self, tmp_path):
        """全点が同一 y のとき min は 1.0（compute_eye_meta と同様）。"""
        rows = [
            {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 0, "px": 10, "py": 50, "qx": 20, "qy": 50},
        ]
        db_path = _make_minimal_db(tmp_path, rows)
        result = db_module.get_eye_height_for_eye("img.jpg", 0, "eyeliner_right", db_path)
        assert result == pytest.approx(1.0)

    def test_face_index_filtering(self, tmp_path):
        """face_index が異なるグループは別扱い。"""
        rows = [
            {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 0, "px": 10, "py": 100, "qx": 20, "qy": 200},
            {"image_path": "img.jpg", "face_index": 1, "part_name": "eyeliner_right",
             "sub_key": "upper", "point_index": 0, "px": 10, "py": 300, "qx": 20, "qy": 600},
        ]
        db_path = _make_minimal_db(tmp_path, rows)
        # face_index=0: ys=[100,200] → height=100
        result = db_module.get_eye_height_for_eye("img.jpg", 0, "eyeliner_right", db_path)
        assert result == pytest.approx(100.0)
        # face_index=1: ys=[300,600] → height=300
        result2 = db_module.get_eye_height_for_eye("img.jpg", 1, "eyeliner_right", db_path)
        assert result2 == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# テスト 2: _get_pq_prime "eye_height" モード
# ---------------------------------------------------------------------------

class TestGetPqPrimeEyeHeight:
    def test_eye_height_uses_db_query(self, monkeypatch):
        """"eye_height" モードで _db.get_eye_height_for_eye が呼ばれ pq_prime を返す。"""
        monkeypatch.setattr(server._db, "get_eye_height_for_eye", lambda *a, **kw: 80.0)

        sample = {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right"}
        orig_pq = 50.0
        # pq_prime = max(80.0 * 0.5, 1.0) = 40.0
        result = server._get_pq_prime(sample, orig_pq, "eye_height", 0.5)

        assert result == pytest.approx(40.0)

    def test_eye_height_fallback_when_db_returns_none(self, monkeypatch):
        """DB が None を返したとき orig_pq_dist を返す（フォールバック）。"""
        monkeypatch.setattr(server._db, "get_eye_height_for_eye", lambda *a, **kw: None)

        sample = {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right"}
        orig_pq = 50.0
        result = server._get_pq_prime(sample, orig_pq, "eye_height", 0.5)

        assert result == pytest.approx(orig_pq)

    def test_eye_height_minimum_pq_prime(self, monkeypatch):
        """eye_height * factor が 1.0 未満のとき 1.0 に切り上げる。"""
        monkeypatch.setattr(server._db, "get_eye_height_for_eye", lambda *a, **kw: 0.5)

        sample = {"image_path": "img.jpg", "face_index": 0, "part_name": "eyeliner_right"}
        result = server._get_pq_prime(sample, 50.0, "eye_height", 0.5)

        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# テスト 3: _load_exp_params で patch_perp_scale を読む
# ---------------------------------------------------------------------------

class TestLoadExpParamsPatchPerpScale:
    def test_reads_patch_perp_scale(self, tmp_path):
        """experiment.json に patch_perp_scale があれば dict に含まれる。"""
        exp_json = tmp_path / "experiment.json"
        exp_json.write_text(json.dumps({
            "hyperparameters": {
                "patch_h": 64, "patch_w": 64,
                "patch_t_min": -0.5, "patch_t_max": 1.5,
                "label_t_min": -0.25, "label_t_max": 1.25,
                "pq_rescale_mode": "eye_height",
                "pq_rescale_factor": 0.5,
                "patch_perp_scale": 4.0,
            }
        }), encoding="utf-8")

        params = server._load_exp_params(tmp_path)
        assert params.get("patch_perp_scale") == pytest.approx(4.0)

    def test_default_patch_perp_scale_when_absent(self, tmp_path):
        """experiment.json に patch_perp_scale がなければ 1.0 になる。"""
        exp_json = tmp_path / "experiment.json"
        exp_json.write_text(json.dumps({
            "hyperparameters": {
                "patch_h": 64, "patch_w": 64,
            }
        }), encoding="utf-8")

        params = server._load_exp_params(tmp_path)
        assert params.get("patch_perp_scale", 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# テスト 4: _reload_models が _infer_patch_perp_scale を更新する
# ---------------------------------------------------------------------------

class TestReloadModelsPatchPerpScale:
    def teardown_method(self):
        _reset_server_globals()

    def test_sets_infer_patch_perp_scale_from_params(self):
        """patch_params に patch_perp_scale がある場合 _infer_patch_perp_scale が更新される。"""
        server._reload_models(
            [],
            patch_params={"patch_perp_scale": 4.0},
        )
        assert server._infer_patch_perp_scale == pytest.approx(4.0)

    def test_default_perp_scale_when_absent(self):
        """patch_params に patch_perp_scale がない場合 _infer_patch_perp_scale は 1.0 になる。"""
        server._reload_models(
            [],
            patch_params={"patch_t_min": -0.5, "patch_t_max": 1.5,
                          "label_t_min": -0.25, "label_t_max": 1.25},
        )
        assert server._infer_patch_perp_scale == pytest.approx(1.0)

    def test_perp_scale_reset_to_default_on_clear(self):
        """モデルクリア（_reload_models([])）後に 1.0 に戻る。"""
        # まず 4.0 に設定
        server._reload_models([], patch_params={"patch_perp_scale": 4.0})
        # クリア
        server._reload_models([])
        assert server._infer_patch_perp_scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# テスト 5: _predict_t が P'Q' 座標でパッチ抽出する
# ---------------------------------------------------------------------------

class TestPredictTUsesRescaledPatch:
    """pq_prime != orig_pq のとき extract_patch が P'Q' 座標で呼ばれることを確認。"""

    def teardown_method(self):
        _reset_server_globals()

    def test_patch_extracted_with_pq_prime_coords(self):
        """pq_prime が orig_pq と異なる場合、extract_patch は P'Q' 座標で呼ばれる。"""
        import numpy as np

        captured = {}

        def mock_extract_patch(image, px, py, qx, qy, patch_size=None, t_min=0.0, t_max=1.0, perp_scale=1.0):
            captured["px"] = px
            captured["py"] = py
            captured["qx"] = qx
            captured["qy"] = qy
            captured["perp_scale"] = perp_scale
            # ダミーパッチ (64x64x3)
            return np.zeros((64, 64, 3), dtype=np.float32)

        # ダミーモデル
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.item.return_value = 0.5

        # サーバーグローバルを直接設定
        import torch
        device = torch.device("cpu")
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with server._models_lock:
            server._infer_models = [mock_model]
            server._infer_device = device
            server._imagenet_mean_t = mean_t
            server._imagenet_std_t  = std_t
            server._active_predictor = None
            server._infer_patch_size = (64, 64)
            server._infer_patch_t_min = -0.5
            server._infer_patch_t_max = 1.5
            server._infer_label_t_min = -0.25
            server._infer_label_t_max = 1.25
            server._infer_patch_perp_scale = 4.0

        image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        # P=(10,50), Q=(30,50) → orig_pq=20, pq_prime=40
        px, py, qx, qy = 10.0, 50.0, 30.0, 50.0
        orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)  # = 20.0
        pq_prime = 40.0  # different from orig_pq

        with patch("annotator.server.extract_patch", side_effect=mock_extract_patch):
            server._predict_t(image_rgb, px, py, qx, qy, 1.0, pq_prime=pq_prime)

        # 中点 O = (20, 50), 単位ベクトル (1, 0)
        # P' = O - (40/2)*ux = 20 - 20 = 0
        # Q' = O + (40/2)*ux = 20 + 20 = 40
        assert captured.get("px") == pytest.approx(0.0)
        assert captured.get("py") == pytest.approx(50.0)
        assert captured.get("qx") == pytest.approx(40.0)
        assert captured.get("qy") == pytest.approx(50.0)

    def test_perp_scale_passed_to_extract_patch(self):
        """_infer_patch_perp_scale が extract_patch の perp_scale 引数に渡される。"""
        import numpy as np

        captured = {}

        def mock_extract_patch(image, px, py, qx, qy, patch_size=None, t_min=0.0, t_max=1.0, perp_scale=1.0):
            captured["perp_scale"] = perp_scale
            return np.zeros((64, 64, 3), dtype=np.float32)

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.item.return_value = 0.5

        import torch
        device = torch.device("cpu")
        with server._models_lock:
            server._infer_models = [mock_model]
            server._infer_device = device
            server._imagenet_mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            server._imagenet_std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            server._active_predictor = None
            server._infer_patch_size = (64, 64)
            server._infer_patch_t_min = -0.5
            server._infer_patch_t_max = 1.5
            server._infer_label_t_min = -0.25
            server._infer_label_t_max = 1.25
            server._infer_patch_perp_scale = 4.0

        image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("annotator.server.extract_patch", side_effect=mock_extract_patch):
            server._predict_t(image_rgb, 10.0, 50.0, 30.0, 50.0, 1.0)

        assert captured.get("perp_scale") == pytest.approx(4.0)

    def test_raw_coords_used_when_pq_prime_equals_orig(self):
        """pq_prime == orig_pq のときは生 PQ 座標でパッチ抽出する。"""
        import numpy as np

        captured = {}

        def mock_extract_patch(image, px, py, qx, qy, patch_size=None, t_min=0.0, t_max=1.0, perp_scale=1.0):
            captured["px"] = px
            captured["py"] = py
            captured["qx"] = qx
            captured["qy"] = qy
            return np.zeros((64, 64, 3), dtype=np.float32)

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.item.return_value = 0.5

        import torch
        device = torch.device("cpu")
        with server._models_lock:
            server._infer_models = [mock_model]
            server._infer_device = device
            server._imagenet_mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            server._imagenet_std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            server._active_predictor = None
            server._infer_patch_size = (64, 64)
            server._infer_patch_t_min = -0.5
            server._infer_patch_t_max = 1.5
            server._infer_label_t_min = -0.25
            server._infer_label_t_max = 1.25
            server._infer_patch_perp_scale = 1.0

        image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        px, py, qx, qy = 10.0, 50.0, 30.0, 50.0
        orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

        with patch("annotator.server.extract_patch", side_effect=mock_extract_patch):
            server._predict_t(image_rgb, px, py, qx, qy, 1.0, pq_prime=orig_pq)

        # pq_prime == orig_pq → 生 PQ 座標
        assert captured.get("px") == pytest.approx(px)
        assert captured.get("qx") == pytest.approx(qx)

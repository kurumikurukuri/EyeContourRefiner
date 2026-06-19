"""tools/annotation_support/t_pred_webui/server.py の FastAPI エンドポイントテスト。

実モデルロード・実画像処理は行わず、predictor_cache / query / eye_crop_renderer を
モック化して検証する。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tools.annotation_support.t_pred_webui import server
from tools.annotation_support.t_pred_webui.t_pred_db import init_db, upsert


@pytest.fixture()
def hints_db(tmp_path) -> Path:
    path = tmp_path / "t_pred_hints.db"
    init_db(path)
    return path


@pytest.fixture()
def client(tmp_path, hints_db, monkeypatch):
    """HINTS_DB / SAMPLES_DB をテスト用パスに差し替えたクライアント。"""
    samples_db = tmp_path / "samples.db"
    monkeypatch.setattr(server, "HINTS_DB", hints_db)
    monkeypatch.setattr(server, "SAMPLES_DB", samples_db)
    return TestClient(server.app)


def _upsert_row(
    db_path: Path,
    image_path: str,
    part_name: str = "eyeliner_left",
    *,
    repr_idx: int = 0,
    min_t_pred: float = 0.3,
    max_t_pred: float = 0.8,
    n_extreme: int = 0,
    n_points: int = 7,
    scan_model: str = "model_a",
) -> None:
    upsert(
        image_path=image_path,
        part_name=part_name,
        repr_idx=repr_idx,
        min_t_pred=min_t_pred,
        max_t_pred=max_t_pred,
        n_extreme=n_extreme,
        n_points=n_points,
        scan_model=scan_model,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndex:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_db(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "extreme": 0}

    def test_with_data(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg", n_extreme=0)
        _upsert_row(hints_db, "b.jpg", n_extreme=2)
        resp = client.get("/api/stats")
        body = resp.json()
        assert body["total"] == 2
        assert body["extreme"] == 1

    def test_db_not_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "HINTS_DB", tmp_path / "missing.db")
        monkeypatch.setattr(server, "SAMPLES_DB", tmp_path / "samples.db")
        c = TestClient(server.app)
        resp = c.get("/api/stats")
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "extreme": 0}


# ---------------------------------------------------------------------------
# GET /api/hints
# ---------------------------------------------------------------------------

class TestHints:
    def test_empty_returns_empty_data(self, client):
        resp = client.get("/api/hints")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"] == []
        assert body["meta"]["total"] == 0

    def test_basic_listing_with_annotator_url(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg", repr_idx=42)
        resp = client.get("/api/hints")
        body = resp.json()
        assert len(body["data"]) == 1
        row = body["data"][0]
        assert row["image_path"] == "a.jpg"
        assert row["annotator_url"] == "http://127.0.0.1:8000/?idx=42"

    def test_meta_pagination(self, client, hints_db):
        for i in range(5):
            _upsert_row(hints_db, f"img{i}.jpg")
        resp = client.get("/api/hints", params={"limit": 2, "offset": 1})
        body = resp.json()
        assert len(body["data"]) == 2
        assert body["meta"]["total"] == 5
        assert body["meta"]["limit"] == 2
        assert body["meta"]["offset"] == 1

    def test_filter_min_t_max(self, client, hints_db):
        _upsert_row(hints_db, "low.jpg", min_t_pred=-0.5)
        _upsert_row(hints_db, "high.jpg", min_t_pred=0.9)
        resp = client.get("/api/hints", params={"min_t_max": 0.0})
        body = resp.json()
        assert [r["image_path"] for r in body["data"]] == ["low.jpg"]
        assert body["meta"]["total"] == 1

    def test_filter_max_t_min(self, client, hints_db):
        _upsert_row(hints_db, "low.jpg", max_t_pred=0.5)
        _upsert_row(hints_db, "high.jpg", max_t_pred=1.5)
        resp = client.get("/api/hints", params={"max_t_min": 1.2})
        body = resp.json()
        assert [r["image_path"] for r in body["data"]] == ["high.jpg"]

    def test_filter_min_n_extreme(self, client, hints_db):
        _upsert_row(hints_db, "none.jpg", n_extreme=0)
        _upsert_row(hints_db, "some.jpg", n_extreme=3)
        resp = client.get("/api/hints", params={"min_n_extreme": 1})
        body = resp.json()
        assert [r["image_path"] for r in body["data"]] == ["some.jpg"]

    def test_sort_by_and_order_asc(self, client, hints_db):
        _upsert_row(hints_db, "low.jpg", n_extreme=1)
        _upsert_row(hints_db, "high.jpg", n_extreme=5)
        resp = client.get("/api/hints", params={"sort_by": "n_extreme", "order": "asc"})
        body = resp.json()
        assert [r["image_path"] for r in body["data"]] == ["low.jpg", "high.jpg"]

    def test_sort_by_and_order_desc_default(self, client, hints_db):
        _upsert_row(hints_db, "low.jpg", n_extreme=1)
        _upsert_row(hints_db, "high.jpg", n_extreme=5)
        resp = client.get("/api/hints", params={"sort_by": "n_extreme"})
        body = resp.json()
        assert [r["image_path"] for r in body["data"]] == ["high.jpg", "low.jpg"]

    def test_invalid_sort_by_returns_400(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg")
        resp = client.get("/api/hints", params={"sort_by": "image_path"})
        assert resp.status_code == 400

    def test_invalid_order_returns_400(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg")
        resp = client.get("/api/hints", params={"order": "sideways"})
        assert resp.status_code == 400

    def test_default_limit_is_100(self, client, hints_db):
        for i in range(3):
            _upsert_row(hints_db, f"img{i}.jpg")
        resp = client.get("/api/hints")
        assert resp.json()["meta"]["limit"] == 100


# ---------------------------------------------------------------------------
# GET /api/hints/crop
# ---------------------------------------------------------------------------

class TestHintsCrop:
    def test_hint_not_found_returns_404(self, client, hints_db):
        resp = client.get("/api/hints/crop", params={"image_path": "nope.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 404

    def test_predictor_none_returns_404(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg", scan_model="bad_model_dir")
        with patch.object(server.predictor_cache, "get_predictor", return_value=None) as mock_pred:
            resp = client.get("/api/hints/crop", params={"image_path": "a.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 404
        mock_pred.assert_called_once()

    def test_no_eye_samples_returns_404(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg")
        with patch.object(server.predictor_cache, "get_predictor", return_value=object()):
            with patch.object(server.query, "fetch_eye_samples", return_value=[]) as mock_fetch:
                resp = client.get("/api/hints/crop", params={"image_path": "a.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 404
        mock_fetch.assert_called_once()

    def test_render_none_returns_404(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg")
        fake_samples = [{"sub_key": "upper", "point_index": 0, "px": 1.0, "py": 1.0, "qx": 2.0, "qy": 2.0, "eye_scale": 60.0}]
        with patch.object(server.predictor_cache, "get_predictor", return_value=object()):
            with patch.object(server.query, "fetch_eye_samples", return_value=fake_samples):
                with patch.object(server.eye_crop_renderer, "load_eye_region_rgb", return_value=("fake_img", 0.0, 0.0)):
                    with patch.object(server.eye_crop_renderer, "render_eye_crop_b64", return_value=None) as mock_render:
                        resp = client.get("/api/hints/crop", params={"image_path": "a.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 404
        mock_render.assert_called_once()

    def test_missing_image_file_returns_404(self, client, hints_db):
        """stale な t_pred_hints.db 行が画像削除後の image_path を指すケース。"""
        _upsert_row(hints_db, "a.jpg")
        fake_samples = [{"sub_key": "upper", "point_index": 0, "px": 1.0, "py": 1.0, "qx": 2.0, "qy": 2.0, "eye_scale": 60.0}]
        with patch.object(server.predictor_cache, "get_predictor", return_value=object()):
            with patch.object(server.query, "fetch_eye_samples", return_value=fake_samples):
                with patch.object(server.eye_crop_renderer, "load_eye_region_rgb", side_effect=FileNotFoundError):
                    resp = client.get("/api/hints/crop", params={"image_path": "a.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 404

    def test_success_returns_base64_text(self, client, hints_db):
        _upsert_row(hints_db, "a.jpg")
        fake_samples = [{"sub_key": "upper", "point_index": 0, "px": 1.0, "py": 1.0, "qx": 2.0, "qy": 2.0, "eye_scale": 60.0}]
        with patch.object(server.predictor_cache, "get_predictor", return_value=object()):
            with patch.object(server.query, "fetch_eye_samples", return_value=fake_samples):
                with patch.object(server.eye_crop_renderer, "load_eye_region_rgb", return_value=("fake_img", 0.0, 0.0)):
                    with patch.object(server.eye_crop_renderer, "render_eye_crop_b64", return_value="ZmFrZWJhc2U2NA=="):
                        resp = client.get("/api/hints/crop", params={"image_path": "a.jpg", "part_name": "eyeliner_left"})
        assert resp.status_code == 200
        assert resp.text == "ZmFrZWJhc2U2NA=="
        assert "text/plain" in resp.headers["content-type"]

    def test_missing_query_params_returns_422(self, client, hints_db):
        resp = client.get("/api/hints/crop")
        assert resp.status_code == 422

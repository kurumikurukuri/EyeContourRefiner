"""apply_pred_eye が、手動保存（/api/annotate）と同じ t_norm [0, 1] の範囲外を DB に書かないことのテスト。"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import annotator.server as server

T_SPAN = server.T_MAX - server.T_MIN


def _run_apply_pred(monkeypatch, t_norms: list[float | None]):
    """各サンプルに t_norms の予測を割り当てて apply_pred_eye を実行し、(レスポンス, 書込内容) を返す。"""
    targets = [
        {"idx": i, "image_path": "a.jpg", "face_index": 0, "part_name": "eyeliner_right",
         "sub_key": "upper", "point_index": i, "px": 0.0, "py": 0.0, "qx": float(i + 1), "qy": 0.0}
        for i in range(len(t_norms))
    ]
    pred_by_qx = {float(i + 1): t for i, t in enumerate(t_norms)}
    written: list[tuple] = []

    monkeypatch.setattr(server, "_infer_models", [object()])
    monkeypatch.setattr(server._db, "get_sample_by_idx", lambda idx: targets[0])
    monkeypatch.setattr(server._db, "get_unannotated_samples_for_eye", lambda *a: targets)
    monkeypatch.setattr(
        server._db, "fill_unannotated_t",
        lambda updates: written.extend(updates) or len(updates),
    )
    monkeypatch.setattr(server, "_load_image_rgb", lambda path: None)
    monkeypatch.setattr(server, "_get_pq_prime", lambda *a, **k: None)
    monkeypatch.setattr(
        server, "_predict_t",
        lambda image, px, py, qx, qy, is_upper, pq_prime=None: pred_by_qx[qx],
    )
    monkeypatch.setattr(server, "_build_image_index", lambda: None)

    return server.apply_pred_eye(0), written


def test_in_range_predictions_are_written_as_raw_t(monkeypatch):
    response, written = _run_apply_pred(monkeypatch, [0.0, 0.5, 1.0])

    assert response == {"ok": True, "updated": 3}
    assert [idx for idx, _ in written] == [0, 1, 2]
    assert [t for _, t in written] == pytest.approx(
        [server.T_MIN, 0.5 * T_SPAN + server.T_MIN, server.T_MAX]
    )


def test_out_of_range_predictions_are_not_written(monkeypatch):
    response, written = _run_apply_pred(monkeypatch, [1.2, 0.5, -0.1])

    assert response == {"ok": True, "updated": 1}
    assert [idx for idx, _ in written] == [1]


def test_all_out_of_range_predictions_raise(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _run_apply_pred(monkeypatch, [1.2, -0.1])

    assert exc_info.value.status_code == 500

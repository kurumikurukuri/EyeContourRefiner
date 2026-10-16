"""eye_contour.training.dual_inference.infer_dual の単体テスト。

旧 experiments/EXP020_dual_branch/scr/infer_exp020.py の infer_dual() を eye_contour/ に移したもの。
EXP020/EXP021/EXP022/EXP024 が共有する推論実装として、manifest の各レコードに対し
生 t 空間の予測を CSV に書き出すことを検証する。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_contour.models.dual_ensemble import get_dual_model, save_dual_model
from eye_contour.training.dual_inference import infer_dual

FIELDNAMES = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
MANIFEST_T = [0.2, 0.5, 0.9]
MODEL_CONFIG = {
    "num_conv_layers": 4,
    "use_sigmoid": True,
    "use_batchnorm": True,
    "dropout_rate": 0.3,
    "base_channel": 32,
    "pool_width": 32,
    "use_aux_face_shape": False,
    "use_is_upper": True,
}


def _write_manifest(directory: Path, t_values: list[float] = MANIFEST_T) -> Path:
    rng = np.random.default_rng(0)
    image_dir = directory / "images"
    image_dir.mkdir()
    rows = []
    for i, t in enumerate(t_values):
        img_path = image_dir / f"img_{i}.jpg"
        cv2.imwrite(str(img_path), rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
        rows.append({
            "idx": 100 + i,
            "image_path": str(img_path),
            "part_name": "eyeliner_right",
            "sub_key": "upper",
            "point_index": i,
            "px": 60.0, "py": 100.0, "qx": 140.0, "qy": 100.0,
            "t": t,
        })
    manifest = directory / "manifest_test.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _write_fold_models(model_dir: Path, fusion_mode: str, n_folds: int = 2) -> None:
    model_dir.mkdir()
    for fold in range(1, n_folds + 1):
        model = get_dual_model(fusion_mode, **MODEL_CONFIG)
        save_dual_model(model, fusion_mode, MODEL_CONFIG, model_dir / f"eyeliner_detection_fold{fold}.pt")


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


@pytest.fixture
def inference_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = _write_manifest(tmp_path)
    model_dir = tmp_path / "models"
    _write_fold_models(model_dir, "independent")
    return manifest, model_dir, tmp_path / "out" / "test_predictions.csv"


def test_writes_one_row_per_manifest_record_in_order(inference_inputs):
    manifest, model_dir, output = inference_inputs
    infer_dual(model_dir, manifest, output, fusion_mode="independent")

    fieldnames, rows = _read_csv(output)
    assert fieldnames == FIELDNAMES
    assert [int(r["idx"]) for r in rows] == [100, 101, 102]
    assert [int(r["point_index"]) for r in rows] == [0, 1, 2]


def test_t_true_is_restored_to_raw_t(inference_inputs):
    manifest, model_dir, output = inference_inputs
    infer_dual(model_dir, manifest, output, fusion_mode="independent")

    _, rows = _read_csv(output)
    assert [float(r["t_true"]) for r in rows] == pytest.approx(MANIFEST_T, abs=1e-5)


def test_t_true_is_not_clamped_to_branch_a_label_range(tmp_path):
    """枝A の label 範囲 [-1, 2] の外にある t も、t_true には生の値がそのまま書かれる（頭打ちされない）。"""
    out_of_range_t = [-1.935, 0.5, 2.53]
    manifest = _write_manifest(tmp_path, out_of_range_t)
    model_dir = tmp_path / "models"
    _write_fold_models(model_dir, "independent")
    output = tmp_path / "out" / "test_predictions.csv"
    infer_dual(model_dir, manifest, output, fusion_mode="independent")

    _, rows = _read_csv(output)
    assert [float(r["t_true"]) for r in rows] == pytest.approx(out_of_range_t, abs=1e-5)


def test_t_pred_is_finite_and_independent_of_batch_size(inference_inputs, tmp_path):
    manifest, model_dir, output = inference_inputs
    second_output = tmp_path / "out" / "batch2.csv"
    infer_dual(model_dir, manifest, output, fusion_mode="independent")
    infer_dual(model_dir, manifest, second_output, fusion_mode="independent", batch_size=2)

    _, first_rows = _read_csv(output)
    _, second_rows = _read_csv(second_output)
    first_preds = [float(r["t_pred"]) for r in first_rows]
    assert np.all(np.isfinite(first_preds))
    assert first_preds == pytest.approx([float(r["t_pred"]) for r in second_rows], abs=1e-5)


def test_reports_progress_through_logging(inference_inputs, caplog):
    manifest, model_dir, output = inference_inputs
    with caplog.at_level(logging.INFO, logger="eye_contour.training.dual_inference"):
        infer_dual(model_dir, manifest, output, fusion_mode="independent")

    assert "推論完了" in caplog.text
    assert str(output) in caplog.text


def test_missing_fold_models_raise(tmp_path):
    manifest = _write_manifest(tmp_path)
    empty_model_dir = tmp_path / "no_models"
    empty_model_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        infer_dual(empty_model_dir, manifest, tmp_path / "out.csv", fusion_mode="independent")

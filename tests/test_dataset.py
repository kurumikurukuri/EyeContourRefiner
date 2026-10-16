"""eye_contour/data/dataset.py のテスト。"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from eye_contour.data.dataset import (
    load_manifest,
    load_manifest_df,
    get_manifest_version,
    EyeContourPatchDataset,
)
from eye_contour.data.weights import compute_eye_meta


def _make_manifest_csv(path: Path, rows: list[dict]) -> None:
    """テスト用 manifest.csv を生成する。"""
    fieldnames = [
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy", "t",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(
    n: int = 5,
    image_path: str = "dummy.jpg",
) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "idx": i,
            "image_path": image_path,
            "part_name": "eyeliner_left",
            "sub_key": "upper",
            "point_index": i,
            "px": 100.0 + i * 10,
            "py": 100.0,
            "qx": 200.0 + i * 10,
            "qy": 100.0,
            "t": f"{0.5 + i * 0.05:.3f}",
        })
    return rows


# ===========================================================
# load_manifest
# ===========================================================

class TestLoadManifest:

    def test_basic_load(self, tmp_path):
        """有効な t 値を持つ全行を読み込む。"""
        csv_path = tmp_path / "manifest.csv"
        _make_manifest_csv(csv_path, _sample_rows(5))
        records = load_manifest(csv_path)
        assert len(records) == 5

    def test_invalid_t_skipped(self, tmp_path):
        """t が数値でない行は除外される。"""
        rows = _sample_rows(3)
        rows[1]["t"] = "invalid"
        csv_path = tmp_path / "manifest.csv"
        _make_manifest_csv(csv_path, rows)
        records = load_manifest(csv_path)
        assert len(records) == 2

    def test_not_found_raises(self, tmp_path):
        """存在しないファイルは FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv")


# ===========================================================
# get_manifest_version
# ===========================================================

class TestGetManifestVersion:

    def test_returns_8char_hex(self, tmp_path):
        """MD5 ハッシュの先頭 8 文字（16進数）を返す。"""
        csv_path = tmp_path / "manifest.csv"
        _make_manifest_csv(csv_path, _sample_rows(3))
        version = get_manifest_version(csv_path)
        assert len(version) == 8
        assert all(c in "0123456789abcdef" for c in version)

    def test_changes_when_file_changes(self, tmp_path):
        """ファイルの内容が変わると version も変わる。"""
        csv_path = tmp_path / "manifest.csv"
        _make_manifest_csv(csv_path, _sample_rows(3))
        v1 = get_manifest_version(csv_path)
        _make_manifest_csv(csv_path, _sample_rows(4))
        v2 = get_manifest_version(csv_path)
        assert v1 != v2


# ===========================================================
# EyeContourPatchDataset
# ===========================================================

class TestEyeContourPatchDataset:

    def _make_dataset_records(self, n=3):
        """テスト用レコードを生成する（weight/val_weight 付き）。"""
        rows = _sample_rows(n)
        compute_eye_meta(rows)
        return rows

    def test_len(self):
        """__len__ がレコード数を返す。"""
        records = self._make_dataset_records(5)
        ds = EyeContourPatchDataset(records, preload_images=False)
        assert len(ds) == 5

    def test_getitem_shapes(self):
        """__getitem__ の出力 shape が正しい。"""
        records = self._make_dataset_records(3)
        ds = EyeContourPatchDataset(records, preload_images=False)
        x, is_upper, pq_ratio, y, weight, val_weight, dist_from_center, norm_eye_scale = ds[0]
        assert x.shape == (3, 64, 128)       # CHW
        assert is_upper.shape == (1,)
        assert pq_ratio.shape == (1,)
        assert y.shape == (1,)
        assert weight.shape == (1,)
        assert val_weight.shape == (1,)
        assert dist_from_center.shape == (1,)
        assert norm_eye_scale.shape == (1,)

    def test_getitem_dtypes(self):
        """出力がすべて float32 テンソル。"""
        records = self._make_dataset_records(2)
        ds = EyeContourPatchDataset(records, preload_images=False)
        x, is_upper, pq_ratio, y, weight, val_weight, dist_from_center, norm_eye_scale = ds[0]
        assert x.dtype == torch.float32
        assert is_upper.dtype == torch.float32
        assert pq_ratio.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_is_upper_value(self):
        """sub_key='upper' のとき is_upper=1.0、それ以外は 0.0。"""
        rows = _sample_rows(2)
        rows[0]["sub_key"] = "upper"
        rows[1]["sub_key"] = "lower"
        compute_eye_meta(rows)
        ds = EyeContourPatchDataset(rows, preload_images=False)
        _, is_upper_0, *_ = ds[0]
        _, is_upper_1, *_ = ds[1]
        assert is_upper_0.item() == 1.0
        assert is_upper_1.item() == 0.0

    def test_t_quant_step(self):
        """t_quant_step=0.1 のとき t が指定桁数に量子化される。"""
        rows = [_sample_rows(1)[0]]
        rows[0]["t"] = "0.567"
        compute_eye_meta(rows)
        ds = EyeContourPatchDataset(rows, preload_images=False, t_quant_step=0.1, use_t_norm=False)
        _, _, _, y, *_ = ds[0]
        assert abs(y.item() - 0.6) < 1e-5

    def test_missing_image_returns_zero_patch(self):
        """画像が存在しない場合はゼロパッチを返す（例外を送出しない）。"""
        rows = _sample_rows(1)
        rows[0]["image_path"] = "/nonexistent/image.jpg"
        compute_eye_meta(rows)
        ds = EyeContourPatchDataset(rows, preload_images=False)
        x, *_ = ds[0]
        assert x.shape == (3, 64, 128)

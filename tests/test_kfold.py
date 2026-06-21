"""eyeliner.training.kfold.run_kfold() の回帰テスト。

_train_single_fold() への抽出リファクタ後も既存の入出力・挙動が不変であることを確認する。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from eyeliner.training.kfold import FoldResult, run_kfold
from eyeliner.training.trainer import TrainConfig

MANIFEST_FIELDNAMES = [
    "image_path", "part_name", "sub_key", "point_index", "px", "py", "qx", "qy", "t",
]


def _write_image(path: Path, size: int = 200) -> None:
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_manifest(tmp_path: Path, n_images: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n_images):
        img_path = image_dir / f"img_{i}.jpg"
        _write_image(img_path)
        rows.append({
            "image_path": str(img_path),
            "part_name": "eyeliner_right",
            "sub_key": "upper",
            "point_index": 3,
            "px": 60.0, "py": 100.0, "qx": 140.0, "qy": 100.0,
            "t": 0.5,
        })
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def _fast_config(**overrides) -> TrainConfig:
    base = dict(
        backbone="custom",
        loss_fn="t",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=2,
        patience=1,
        seed=42,
        num_conv_layers=2,
        use_sigmoid=True,
        use_batchnorm=False,
        dropout_rate=0.0,
        aug_flags=0,
        num_workers=0,
    )
    base.update(overrides)
    return TrainConfig(**base)


class TestRunKfold:

    def test_returns_one_fold_result_per_k(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        result = run_kfold(manifest, tmp_path / "outputs", _fast_config(), k_folds=2)
        assert all(isinstance(r, FoldResult) for r in result)
        assert len(result) == 2

    def test_saves_fold_model_files(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        exp_dir = tmp_path / "outputs"
        run_kfold(manifest, exp_dir, _fast_config(), k_folds=2)
        saved = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
        assert len(saved) == 2

    def test_k_equals_one_uses_group_shuffle_split(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, n_images=8)
        result = run_kfold(manifest, tmp_path / "outputs", _fast_config(), k_folds=1)
        assert len(result) == 1

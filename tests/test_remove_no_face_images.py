"""tools/data_management/remove_no_face_images.py の attribution CSV 書き戻しテスト。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import remove_no_face_images as rnf  # noqa: E402


def test_attribution_csv_keeps_fields_with_commas_and_quotes(tmp_path, monkeypatch):
    """author / title にカンマ・引用符があっても列ずれせずに書き戻す。"""
    csv_path = tmp_path / "open_images_attribution.csv"
    fieldnames = ["image_id", "author", "title"]
    rows = [
        {"image_id": "keep", "author": "Doe, John", "title": 'Say "hi", world'},
        {"image_id": "drop", "author": "x", "title": "y"},
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(rnf, "ATTRIBUTION_CSV", csv_path)

    removed = rnf._remove_from_attribution_csv({"drop"}, dry_run=False)

    with csv_path.open(newline="", encoding="utf-8") as f:
        result = list(csv.DictReader(f))
    assert removed == 1
    assert result == [rows[0]]


def test_attribution_csv_preserves_legacy_malformed_rows(tmp_path, monkeypatch):
    """旧 writer が残した列ずれ行（引用なしのカンマ入り）があってもクラッシュせず、そのまま残す。"""
    csv_path = tmp_path / "open_images_attribution.csv"
    header = "image_id,author,title"
    malformed = "keep,Doe, John,Hot, Tamales"
    csv_path.write_bytes(f"{header}\r\n{malformed}\r\ndrop,x,y\r\n".encode("utf-8"))
    monkeypatch.setattr(rnf, "ATTRIBUTION_CSV", csv_path)

    removed = rnf._remove_from_attribution_csv({"drop"}, dry_run=False)

    assert removed == 1
    assert csv_path.read_bytes().decode("utf-8").splitlines() == [header, malformed]

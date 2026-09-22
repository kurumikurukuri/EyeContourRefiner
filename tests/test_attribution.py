"""tools/data_management/attribution.py のテスト（追記・重複防止・排他制御）。"""
from __future__ import annotations

import csv
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import attribution as attr  # noqa: E402
from tools.data_management.oi_metadata import ImageMeta  # noqa: E402

CC_BY = "https://creativecommons.org/licenses/by/2.0/"


def _meta(image_id: str, **kw) -> ImageMeta:
    return ImageMeta(
        image_id=image_id,
        license_url=CC_BY,
        original_url=kw.get("original_url", f"https://flickr.example/{image_id}.jpg"),
        author=kw.get("author", "Alice"),
        title=kw.get("title", "Untitled"),
    )


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestAppend:

    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "data" / "attr.csv"

        added = attr.append_attribution(path, [_meta("a1"), _meta("b2")])

        assert added == 2
        assert path.read_text(encoding="utf-8").splitlines()[0] == (
            "image_id,license_url,original_url,author,title"
        )
        assert [r["image_id"] for r in _rows(path)] == ["a1", "b2"]

    def test_round_trips_special_characters(self, tmp_path):
        path = tmp_path / "attr.csv"
        meta = _meta("a1", author="Zoë, \"Z\"", title="日本語\nタイトル")

        attr.append_attribution(path, [meta])

        (row,) = _rows(path)
        assert row["author"] == 'Zoë, "Z"'
        assert row["title"] == "日本語\nタイトル"

    def test_skips_ids_already_in_file(self, tmp_path):
        path = tmp_path / "attr.csv"
        attr.append_attribution(path, [_meta("a1", author="First")])

        added = attr.append_attribution(path, [_meta("a1", author="Second"), _meta("b2")])

        assert added == 1
        rows = _rows(path)
        assert [r["image_id"] for r in rows] == ["a1", "b2"]
        assert rows[0]["author"] == "First"

    def test_skips_duplicates_within_one_call(self, tmp_path):
        path = tmp_path / "attr.csv"

        added = attr.append_attribution(path, [_meta("a1"), _meta("a1")])

        assert added == 1
        assert len(_rows(path)) == 1

    def test_empty_input_does_not_create_file(self, tmp_path):
        path = tmp_path / "attr.csv"

        assert attr.append_attribution(path, []) == 0
        assert not path.exists()

    def test_repairs_missing_trailing_newline(self, tmp_path):
        path = tmp_path / "attr.csv"
        path.write_bytes(
            b"image_id,license_url,original_url,author,title\n"
            b"a1,https://l,https://o,au,ti"
        )

        attr.append_attribution(path, [_meta("b2")])

        assert [r["image_id"] for r in _rows(path)] == ["a1", "b2"]

    def test_preserves_existing_crlf_line_endings(self, tmp_path):
        path = tmp_path / "attr.csv"
        path.write_bytes(
            b"image_id,license_url,original_url,author,title\r\n"
            b"a1,https://l,https://o,au,ti\r\n"
        )

        attr.append_attribution(path, [_meta("b2")])

        raw = path.read_bytes()
        assert raw.count(b"\r\n") == 3
        assert raw.replace(b"\r\n", b"").count(b"\n") == 0

    def test_rejects_file_without_image_id_column(self, tmp_path):
        path = tmp_path / "attr.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")

        with pytest.raises(attr.AttributionError, match="image_id"):
            attr.append_attribution(path, [_meta("a1")])

        assert path.read_text(encoding="utf-8") == "foo,bar\n1,2\n"


class TestRead:

    def test_missing_file_is_empty(self, tmp_path):
        assert attr.read_attribution(tmp_path / "none.csv") == {}
        assert attr.known_image_ids(tmp_path / "none.csv") == set()

    def test_reads_back_image_meta(self, tmp_path):
        path = tmp_path / "attr.csv"
        attr.append_attribution(path, [_meta("a1"), _meta("b2")])

        metas = attr.read_attribution(path)

        assert metas["a1"] == _meta("a1")
        assert attr.known_image_ids(path) == {"a1", "b2"}

    def test_bundled_attribution_csv_is_readable(self):
        ids = attr.known_image_ids(ROOT / "data" / "open_images_attribution.csv")

        assert len(ids) > 1000
        assert "0021520c13029a24" in ids


class TestConcurrency:

    def test_threads_never_duplicate_ids(self, tmp_path):
        path = tmp_path / "attr.csv"
        shared = [_meta(f"id{i:03d}") for i in range(40)]

        def worker(offset: int) -> None:
            for i in range(0, 40, 4):
                attr.append_attribution(path, shared[(i + offset) % 40:(i + offset) % 40 + 4])

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids = [r["image_id"] for r in _rows(path)]
        assert len(ids) == len(set(ids))
        assert set(ids) == {m.image_id for m in shared}

    def test_processes_never_duplicate_ids(self, tmp_path):
        path = tmp_path / "attr.csv"
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, {root!r})
            from pathlib import Path
            from tools.data_management import attribution as attr
            from tools.data_management.oi_metadata import ImageMeta
            metas = [ImageMeta(f"id{{i:03d}}", "l", "o", "a", "t") for i in range(60)]
            for start in range(0, 60, 5):
                attr.append_attribution(Path({path!r}), metas[start:start + 5 + int(sys.argv[1])])
            """
        ).format(root=str(ROOT), path=str(path))

        procs = [
            subprocess.Popen([sys.executable, "-c", script, str(k)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for k in range(4)
        ]
        outputs = [p.communicate(timeout=120) for p in procs]

        for proc, (_, err) in zip(procs, outputs, strict=True):
            assert proc.returncode == 0, err.decode("utf-8", errors="replace")
        ids = [r["image_id"] for r in _rows(path)]
        assert len(ids) == len(set(ids)) == 60

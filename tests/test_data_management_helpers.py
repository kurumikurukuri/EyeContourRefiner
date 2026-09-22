"""tools/data_management/_fileio.py・_console.py の共通ヘルパーのテスト。"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import _console, _fileio  # noqa: E402


class TestPartPath:

    def test_is_a_sibling_of_the_destination_and_unique_per_process(self, tmp_path):
        dest = tmp_path / "a1.jpg"

        part = _fileio.part_path(dest)

        assert part.parent == dest.parent
        assert part.name == f"a1.jpg.{os.getpid()}.part"

    def test_differs_from_the_legacy_fixed_name(self, tmp_path):
        assert _fileio.part_path(tmp_path / "x.csv").name != "x.csv.part"


class TestAtomicReplace:

    def test_replaces_on_success(self, tmp_path):
        dest = tmp_path / "out" / "f.txt"

        with _fileio.atomic_replace(dest) as part:
            part.write_text("new", encoding="utf-8")
            assert not dest.exists()  # 書き込み中は最終ファイルに現れない

        assert dest.read_text(encoding="utf-8") == "new"
        assert list(dest.parent.glob("*.part")) == []

    def test_keeps_the_previous_file_and_leaves_no_part_on_failure(self, tmp_path):
        dest = tmp_path / "f.txt"
        dest.write_text("old", encoding="utf-8")

        with pytest.raises(RuntimeError):
            with _fileio.atomic_replace(dest) as part:
                part.write_text("half-written", encoding="utf-8")
                raise RuntimeError("boom")

        assert dest.read_text(encoding="utf-8") == "old"
        assert list(tmp_path.glob("*.part")) == []

    def test_an_unwritten_part_is_an_error_not_a_silent_no_op(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            with _fileio.atomic_replace(tmp_path / "f.txt"):
                pass


class TestConsoleSafety:

    @staticmethod
    def _cp932_stream() -> io.TextIOWrapper:
        return io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")

    def test_precondition_the_cp932_console_cannot_encode_a_check_mark(self):
        stream = self._cp932_stream()

        with pytest.raises(UnicodeEncodeError):
            stream.write("ok ✓")

    def test_does_not_crash_on_characters_the_console_cannot_encode(self, monkeypatch):
        out, err = self._cp932_stream(), self._cp932_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        _console.make_console_safe()
        print("画像 data/images/😀.jpg は不正です")  # 外部由来の文字（絵文字入りのパス等）
        print("エラー ✓", file=sys.stderr)

        out.flush()
        assert out.buffer.getvalue().decode("cp932").startswith("画像 data/images/")

    def test_streams_without_reconfigure_are_left_alone(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        _console.make_console_safe()  # 例外にならない

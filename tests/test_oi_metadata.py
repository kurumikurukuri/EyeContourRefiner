"""tools/data_management/oi_metadata.py のテスト。

Open Images のメタデータ・注釈 CSV の取得とキャッシュ（ネットワークは差し替え）。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import oi_metadata as oim  # noqa: E402

CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
CC_BY_CA = "https://creativecommons.org/licenses/by/2.0/ca/"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/2.0/"
CC_BY_NC = "https://creativecommons.org/licenses/by-nc/2.0/"


def _meta_row(image_id, license_url, **extra) -> dict:
    return {
        "ImageID": image_id,
        "License": license_url,
        "OriginalURL": extra.get("original_url", f"https://flickr.example/{image_id}.jpg"),
        "Author": extra.get("author", "Alice"),
        "Title": extra.get("title", "Untitled"),
    }


@pytest.fixture
def stream(monkeypatch):
    """_stream_csv_rows を差し替え、呼び出し URL を記録する。"""
    state = {"rows": [], "calls": []}

    def fake_stream(url):
        state["calls"].append(url)
        yield from state["rows"]

    monkeypatch.setattr(oim, "_stream_csv_rows", fake_stream)
    return state


class TestLicense:

    @pytest.mark.parametrize("url", [CC0, CC_BY, CC0.rstrip("/"), f"  {CC_BY}  "])
    def test_permissive_licenses_are_accepted(self, url):
        assert oim.is_permissive_license(url)

    @pytest.mark.parametrize("url", [CC_BY_CA, CC_BY_SA, CC_BY_NC, "", "https://example.com/"])
    def test_other_licenses_and_jurisdiction_ports_are_rejected(self, url):
        assert not oim.is_permissive_license(url)


class TestFetchPermissiveMetadata:

    def test_keeps_only_permissive_rows(self, tmp_path, stream):
        stream["rows"] = [
            _meta_row("a1", CC0),
            _meta_row("b2", CC_BY),
            _meta_row("c3", CC_BY_SA),
            _meta_row("d4", CC_BY_CA),
        ]

        metas = oim.fetch_permissive_metadata("validation", tmp_path)

        assert set(metas) == {"a1", "b2"}
        assert metas["a1"].license_url == CC0
        assert metas["a1"].author == "Alice"
        assert stream["calls"] == [oim.METADATA_URLS["validation"]]

    def test_rows_with_unsafe_image_ids_are_dropped_at_ingest(self, tmp_path, stream):
        """リモート由来の ID は取り込み時に検証し、キャッシュにも残さない（後段で実行を止めない）。"""
        stream["rows"] = [
            _meta_row("../evil", CC0), _meta_row("a/b", CC0), _meta_row("a b", CC0), _meta_row("ok", CC0)]

        metas = oim.fetch_permissive_metadata("validation", tmp_path)

        assert set(metas) == {"ok"}
        assert "evil" not in (tmp_path / "validation_permissive_meta.csv").read_text(encoding="utf-8")

    def test_skips_rows_without_image_id(self, tmp_path, stream):
        stream["rows"] = [_meta_row("", CC0), {"License": CC0}, _meta_row(" ok ", CC0)]

        metas = oim.fetch_permissive_metadata("test", tmp_path)

        assert set(metas) == {"ok"}

    def test_second_call_uses_cache_without_network(self, tmp_path, stream):
        stream["rows"] = [_meta_row("a1", CC0, title='He said "hi", ok', author="日本語")]
        first = oim.fetch_permissive_metadata("train", tmp_path)
        stream["rows"] = []

        second = oim.fetch_permissive_metadata("train", tmp_path)

        assert len(stream["calls"]) == 1
        assert second == first
        assert second["a1"].title == 'He said "hi", ok'
        assert second["a1"].author == "日本語"
        assert (tmp_path / "train_permissive_meta.csv").is_file()

    def test_refresh_ignores_cache(self, tmp_path, stream):
        stream["rows"] = [_meta_row("old", CC0)]
        oim.fetch_permissive_metadata("train", tmp_path)
        stream["rows"] = [_meta_row("new", CC0)]

        metas = oim.fetch_permissive_metadata("train", tmp_path, refresh=True)

        assert set(metas) == {"new"}
        assert len(stream["calls"]) == 2

    def test_unknown_split_is_rejected(self, tmp_path, stream):
        with pytest.raises(ValueError, match="split"):
            oim.fetch_permissive_metadata("dev", tmp_path)

    def test_failed_fetch_leaves_no_cache_behind(self, tmp_path, monkeypatch):
        def broken(url):
            yield _meta_row("a1", CC0)
            raise ConnectionError("boom")

        monkeypatch.setattr(oim, "_stream_csv_rows", broken)

        with pytest.raises(ConnectionError):
            oim.fetch_permissive_metadata("train", tmp_path)

        assert not (tmp_path / "train_permissive_meta.csv").exists()

    def test_image_meta_is_immutable(self):
        meta = oim.ImageMeta("a", CC0, "u", "au", "t")
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.author = "x"  # type: ignore[misc]

    def test_url_tables_are_read_only(self):
        with pytest.raises(TypeError):
            oim.METADATA_URLS["dev"] = "http://evil"  # type: ignore[index]
        with pytest.raises(TypeError):
            oim.ANNOTATION_URLS["dev"] = "http://evil"  # type: ignore[index]


def _fake_download(content: bytes = b"downloaded"):
    calls = []

    def download(url: str, dest: Path) -> None:
        calls.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    download.calls = calls  # type: ignore[attr-defined]
    return download


def _make_fiftyone_cache(root: Path, split: str) -> tuple[Path, Path]:
    classes = root / split / "metadata" / "classes.csv"
    detections = root / split / "labels" / "detections.csv"
    classes.parent.mkdir(parents=True)
    detections.parent.mkdir(parents=True)
    classes.write_text("/m/0dzct,Human face\n", encoding="utf-8")
    detections.write_text("ImageID,LabelName\n", encoding="utf-8")
    return classes, detections


class TestResolveCsv:

    def test_reuses_fiftyone_cache_read_only(self, tmp_path):
        fo_root = tmp_path / "fiftyone"
        classes, detections = _make_fiftyone_cache(fo_root, "validation")
        download = _fake_download()

        got_classes = oim.resolve_classes_csv(
            "validation", tmp_path / "cache", fiftyone_root=fo_root, download=download)
        got_ann = oim.resolve_annotations_csv(
            "validation", tmp_path / "cache", fiftyone_root=fo_root, download=download)

        assert got_classes == classes
        assert got_ann == detections
        assert download.calls == []

    def test_downloads_into_cache_dir_when_no_fiftyone_cache(self, tmp_path):
        download = _fake_download(b"x")
        cache = tmp_path / "cache"

        ann = oim.resolve_annotations_csv(
            "test", cache, fiftyone_root=tmp_path / "none", download=download)
        classes = oim.resolve_classes_csv(
            "test", cache, fiftyone_root=tmp_path / "none", download=download)

        assert ann == cache / "test_annotations_bbox.csv"
        assert classes == cache / "classes.csv"
        assert [c[0] for c in download.calls] == [oim.ANNOTATION_URLS["test"], oim.CLASSES_URL]

    def test_own_cache_is_reused_on_second_call(self, tmp_path):
        download = _fake_download()
        oim.resolve_annotations_csv("test", tmp_path, fiftyone_root=tmp_path / "n", download=download)
        oim.resolve_annotations_csv("test", tmp_path, fiftyone_root=tmp_path / "n", download=download)

        assert len(download.calls) == 1

    def test_refresh_downloads_again_and_never_touches_fiftyone_cache(self, tmp_path):
        fo_root = tmp_path / "fiftyone"
        _, detections = _make_fiftyone_cache(fo_root, "validation")
        before = detections.read_bytes()
        download = _fake_download(b"fresh")

        got = oim.resolve_annotations_csv(
            "validation", tmp_path / "cache", refresh=True,
            fiftyone_root=fo_root, download=download)

        assert got == tmp_path / "cache" / "validation_annotations_bbox.csv"
        assert got.read_bytes() == b"fresh"
        assert detections.read_bytes() == before


class TestResolveClassesCsvValidation:

    def test_unknown_split_is_rejected_like_the_annotation_csv(self, tmp_path):
        with pytest.raises(ValueError, match="split"):
            oim.resolve_classes_csv("dev", tmp_path, fiftyone_root=tmp_path / "n", download=_fake_download())


class TestDownloadFile:

    class _Response:
        def __init__(self, chunks, fail_after=None, status=200, content_length=None, headers=None):
            self._chunks = chunks
            self._fail_after = fail_after
            self.status_code = status
            self.headers = dict(headers or {})
            if content_length is not None:
                self.headers["Content-Length"] = str(content_length)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def iter_content(self, chunk_size):
            for i, chunk in enumerate(self._chunks):
                if self._fail_after is not None and i >= self._fail_after:
                    raise ConnectionError("dropped")
                yield chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_writes_via_part_file_and_renames(self, tmp_path):
        dest = tmp_path / "sub" / "f.csv"

        oim.download_file("http://x", dest, get=lambda url, **kw: self._Response([b"ab", b"cd"]))

        assert dest.read_bytes() == b"abcd"
        assert list(dest.parent.glob("*.part")) == []

    def test_interrupted_download_leaves_no_final_file(self, tmp_path):
        dest = tmp_path / "f.csv"

        with pytest.raises(ConnectionError):
            oim.download_file(
                "http://x", dest,
                get=lambda url, **kw: self._Response([b"ab", b"cd"], fail_after=1))

        assert not dest.exists()
        assert list(tmp_path.glob("*.part")) == []

    def test_truncated_download_is_rejected_by_content_length(self, tmp_path):
        """urllib3 1.x は切れた本体を例外にしない。長さ不一致のままキャッシュにしない。"""
        dest = tmp_path / "f.csv"

        with pytest.raises(OSError, match="サイズ"):
            oim.download_file(
                "http://x", dest,
                get=lambda url, **kw: self._Response([b"ab", b"cd"], content_length=1000))

        assert not dest.exists()
        assert list(tmp_path.glob("*.part")) == []

    def test_matching_content_length_is_accepted(self, tmp_path):
        dest = tmp_path / "f.csv"

        oim.download_file(
            "http://x", dest, get=lambda url, **kw: self._Response([b"ab", b"cd"], content_length=4))

        assert dest.read_bytes() == b"abcd"

    def test_content_length_is_ignored_for_encoded_responses(self, tmp_path):
        dest = tmp_path / "f.csv"

        oim.download_file(
            "http://x", dest,
            get=lambda url, **kw: self._Response(
                [b"ab"], content_length=999, headers={"Content-Encoding": "gzip"}))

        assert dest.read_bytes() == b"ab"

    def test_http_error_is_raised(self, tmp_path):
        with pytest.raises(RuntimeError, match="404"):
            oim.download_file(
                "http://x", tmp_path / "f.csv",
                get=lambda url, **kw: self._Response([], status=404))


class TestLoadLabelMaps:

    def test_returns_both_directions(self, tmp_path):
        path = tmp_path / "classes.csv"
        path.write_text('/m/0dzct,Human face\n/m/01g317,Person\n"/m/x","Teddy bear"\n\nbad\n', encoding="utf-8")

        name_to_id, id_to_name = oim.load_label_maps(path)

        assert name_to_id["Human face"] == "/m/0dzct"
        assert id_to_name["/m/01g317"] == "Person"
        assert name_to_id["Teddy bear"] == "/m/x"
        assert len(name_to_id) == 3

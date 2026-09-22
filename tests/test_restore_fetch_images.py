"""tools/data_management/s3_download.py の S3 直接取得のテスト。

ネットワークは差し替え（フェイクの session）。`.part` 経由の保存・Content-Length・サイズ上限・
JPEG マジックバイトの検証・リトライ方針・リダイレクト拒否・split 探索・再開性を確認する。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import pytest
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import s3_download as fi  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60 + b"\xff\xd9"
BASE = "https://open-images-dataset.s3.amazonaws.com"


class FakeResponse:
    def __init__(self, status=200, body=JPEG, headers=None, fail_mid_body=False, content_length="auto"):
        self.status_code = status
        self._body = body
        self._fail_mid_body = fail_mid_body
        self.headers = dict(headers or {})
        if content_length == "auto":
            self.headers.setdefault("Content-Length", str(len(body)))
        elif content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.closed = False

    def iter_content(self, chunk_size=1):
        half = max(len(self._body) // 2, 1)
        yield self._body[:half]
        if self._fail_mid_body:
            raise requests.exceptions.ChunkedEncodingError("connection broken")
        yield self._body[half:]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeSession:
    """URL ごとに用意した応答（または例外）を順に返す。尽きたら最後の応答を返し続ける。"""

    def __init__(self, routes: dict[str, list]):
        self._routes = {url: list(seq) for url, seq in routes.items()}
        self.calls: list[str] = []
        self.request_options: list[tuple] = []
        self._lock = threading.Lock()

    def get(self, url, stream=True, timeout=None, headers=None, allow_redirects=True):
        with self._lock:
            self.calls.append(url)
            self.request_options.append((dict(headers or {}), allow_redirects))
            seq = self._routes.get(url)
            if not seq:
                return FakeResponse(status=404, body=b"")
            item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, BaseException):
            raise item
        return item


def _url(split: str, image_id: str) -> str:
    return f"{BASE}/{split}/{image_id}.jpg"


def _download(tmp_path, target, session, *, retries=3, sleeps: Optional[list] = None):
    sleeps = sleeps if sleeps is not None else []
    return fi.download_one(
        target, tmp_path, session=session, retries=retries, backoff=1.0,
        sleep=sleeps.append,
    )


class TestImageUrl:

    def test_builds_public_s3_url(self):
        assert fi.image_url("train", "abc123") == _url("train", "abc123")

    @pytest.mark.parametrize("split, image_id", [
        ("dev", "abc"), ("train", "../abc"), ("train", "a/b"), ("train", ""), ("train", "a.b"),
    ])
    def test_rejects_unsafe_input(self, split, image_id):
        with pytest.raises(ValueError):
            fi.image_url(split, image_id)


class TestDownloadOne:

    def test_downloads_via_part_file_and_renames(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"
        assert result.split == "train"
        assert (tmp_path / "a1.jpg").read_bytes() == JPEG
        assert list(tmp_path.glob("*.part")) == []
        assert session.calls == [_url("train", "a1")]

    def test_creates_missing_output_dir(self, tmp_path):
        session = FakeSession({_url("test", "a1"): [FakeResponse()]})

        fi.download_one(fi.FetchTarget("a1", "test"), tmp_path / "deep" / "dir",
                        session=session, sleep=lambda s: None)

        assert (tmp_path / "deep" / "dir" / "a1.jpg").is_file()

    def test_existing_file_is_skipped_without_request(self, tmp_path):
        (tmp_path / "a1.jpg").write_bytes(b"already")
        session = FakeSession({})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "skipped"
        assert session.calls == []
        assert (tmp_path / "a1.jpg").read_bytes() == b"already"

    def test_part_files_of_other_processes_are_never_touched(self, tmp_path):
        """一時ファイル名は PID 入り。別プロセス（や旧方式）の `.part` を上書き・削除しない。"""
        other = tmp_path / f"a1.jpg.{os.getpid() + 1}.part"
        legacy = tmp_path / "a1.jpg.part"
        other.write_bytes(b"other-process-in-progress")
        legacy.write_bytes(b"legacy")
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"
        assert (tmp_path / "a1.jpg").read_bytes() == JPEG
        assert other.read_bytes() == b"other-process-in-progress"
        assert legacy.read_bytes() == b"legacy"
        assert sorted(p.name for p in tmp_path.glob("*.part")) == sorted([other.name, legacy.name])

    def test_own_stale_part_file_is_overwritten(self, tmp_path):
        (tmp_path / f"a1.jpg.{os.getpid()}.part").write_bytes(b"partial-garbage" * 100)
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"
        assert (tmp_path / "a1.jpg").read_bytes() == JPEG
        assert list(tmp_path.glob("*.part")) == []

    def test_truncated_body_is_detected_by_content_length_and_retried(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [
            FakeResponse(content_length=len(JPEG) + 500),
            FakeResponse(),
        ]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, sleeps=sleeps)

        assert result.status == "downloaded"
        assert len(session.calls) == 2
        assert sleeps == [1.0]
        assert (tmp_path / "a1.jpg").read_bytes() == JPEG

    def test_persistently_truncated_body_fails_and_leaves_nothing(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse(content_length=len(JPEG) + 1)]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, retries=2)

        assert result.status == "failed"
        assert len(session.calls) == 3
        assert list(tmp_path.iterdir()) == []

    def test_non_jpeg_body_fails_without_retry(self, tmp_path):
        html = b"<html>error page</html>"
        session = FakeSession({_url("train", "a1"): [FakeResponse(body=html)]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "failed"
        assert "JPEG" in result.detail
        assert len(session.calls) == 1
        assert list(tmp_path.iterdir()) == []

    def test_empty_body_is_not_a_jpeg(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse(body=b"")]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "failed"
        assert list(tmp_path.iterdir()) == []

    def test_content_length_is_not_checked_for_encoded_responses(self, tmp_path):
        response = FakeResponse(headers={"Content-Encoding": "gzip"}, content_length=9999)
        session = FakeSession({_url("train", "a1"): [response]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"

    def test_missing_content_length_is_tolerated(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse(content_length=None)]})

        assert _download(tmp_path, fi.FetchTarget("a1", "train"), session).status == "downloaded"

    def test_5xx_is_retried_with_exponential_backoff(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [
            FakeResponse(status=503, body=b""),
            FakeResponse(status=500, body=b""),
            FakeResponse(),
        ]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, sleeps=sleeps)

        assert result.status == "downloaded"
        assert sleeps == [1.0, 2.0]

    def test_5xx_exhausting_retries_fails(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse(status=502, body=b"")]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, retries=3, sleeps=sleeps)

        assert result.status == "failed"
        assert "502" in result.detail
        assert len(session.calls) == 4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_connection_errors_are_retried(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [
            requests.exceptions.ConnectionError("reset"),
            requests.exceptions.Timeout("slow"),
            FakeResponse(),
        ]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"
        assert len(session.calls) == 3

    def test_connection_drop_mid_body_is_retried_and_part_removed(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [
            FakeResponse(fail_mid_body=True),
            FakeResponse(),
        ]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "downloaded"
        assert list(tmp_path.glob("*.part")) == []

    def test_persistent_connection_error_fails(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [requests.exceptions.ConnectionError("down")]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, retries=2)

        assert result.status == "failed"
        assert len(session.calls) == 3

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_client_auth_errors_fail_immediately(self, tmp_path, status):
        session = FakeSession({_url("train", "a1"): [FakeResponse(status=status, body=b"")]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, sleeps=sleeps)

        assert result.status == "failed"
        assert str(status) in result.detail
        assert len(session.calls) == 1
        assert sleeps == []

    def test_404_is_reported_as_not_found_without_retry(self, tmp_path):
        session = FakeSession({})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "not_found"
        assert len(session.calls) == 1

    def test_responses_are_closed(self, tmp_path):
        response = FakeResponse()
        session = FakeSession({_url("train", "a1"): [response]})

        _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert response.closed


class TestRequestPolicy:

    def test_requests_are_sent_without_following_redirects_or_compression(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        ((headers, allow_redirects),) = session.request_options
        assert allow_redirects is False
        assert headers == {"Accept-Encoding": "identity"}

    @pytest.mark.parametrize("status", [301, 302, 307])
    def test_redirects_are_failures_not_followed_and_not_retried(self, tmp_path, status):
        session = FakeSession({_url("train", "a1"): [FakeResponse(status=status, body=b"")]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, sleeps=sleeps)

        assert result.status == "failed" and str(status) in result.detail
        assert len(session.calls) == 1 and sleeps == []

    def test_request_headers_table_is_read_only(self):
        with pytest.raises(TypeError):
            fi.REQUEST_HEADERS["Accept-Encoding"] = "gzip"  # type: ignore[index]


class TestSizeCap:

    def test_declared_size_over_the_cap_is_rejected_before_writing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fi, "MAX_IMAGE_BYTES", 10)
        session = FakeSession({_url("train", "a1"): [FakeResponse(content_length=11)]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "failed" and "上限" in result.detail
        assert len(session.calls) == 1
        assert list(tmp_path.iterdir()) == []

    def test_body_growing_past_the_cap_without_a_length_header_is_aborted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fi, "MAX_IMAGE_BYTES", 30)
        session = FakeSession({_url("train", "a1"): [FakeResponse(content_length=None)]})  # 本体は 66 bytes

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "failed" and "上限" in result.detail
        assert len(session.calls) == 1
        assert list(tmp_path.iterdir()) == []

    def test_a_compressed_response_cannot_bypass_the_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fi, "MAX_IMAGE_BYTES", 30)
        response = FakeResponse(headers={"Content-Encoding": "gzip"}, content_length=None)
        session = FakeSession({_url("train", "a1"): [response]})

        assert _download(tmp_path, fi.FetchTarget("a1", "train"), session).status == "failed"

    def test_body_at_exactly_the_cap_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fi, "MAX_IMAGE_BYTES", len(JPEG))
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        assert _download(tmp_path, fi.FetchTarget("a1", "train"), session).status == "downloaded"


class TestFileSystemResilience:
    """Windows ではウイルス対策ソフト等が書き込み直後のファイルを一時的に掴み、rename が失敗する。"""

    @staticmethod
    def _flaky_replace(monkeypatch, failures: int):
        error = PermissionError("in use")
        original = Path.replace
        state = {"calls": 0}

        def replace(self, target):
            state["calls"] += 1
            if state["calls"] <= failures:
                raise error
            return original(self, target)

        monkeypatch.setattr(Path, "replace", replace)
        return state

    def test_a_transient_permission_error_on_rename_is_retried(self, tmp_path, monkeypatch):
        state = self._flaky_replace(monkeypatch, failures=2)
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})
        sleeps: list[float] = []

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session, sleeps=sleeps)

        assert result.status == "downloaded" and state["calls"] == 3
        assert sleeps == [0.2, 0.4]
        assert (tmp_path / "a1.jpg").read_bytes() == JPEG

    def test_a_persistent_error_fails_only_that_image_and_leaves_no_part(self, tmp_path, monkeypatch):
        self._flaky_replace(monkeypatch, failures=10_000)
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", "train"), session)

        assert result.status == "failed" and "PermissionError" in result.detail
        assert list(tmp_path.iterdir()) == []

    def test_an_os_error_on_one_image_does_not_abort_the_rest(self, tmp_path, monkeypatch):
        original_open = open

        def failing_open(path, *args, **kwargs):
            if "bad" in str(path):
                raise OSError("No space left on device")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", failing_open)
        ids = ["good1", "bad", "good2"]
        session = FakeSession({_url("train", i): [FakeResponse()] for i in ids})

        results = fi.fetch_many([fi.FetchTarget(i, "train") for i in ids], tmp_path,
                                session=session, workers=1, sleep=lambda s: None)

        assert [r.status for r in results] == ["downloaded", "failed", "downloaded"]
        assert "No space left" in results[1].detail


class TestSplitProbing:

    def test_unknown_split_tries_train_validation_test_in_order(self, tmp_path):
        session = FakeSession({_url("validation", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", None), session)

        assert result.status == "downloaded"
        assert result.split == "validation"
        assert session.calls == [_url("train", "a1"), _url("validation", "a1")]

    def test_stops_at_first_hit(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse()]})

        result = _download(tmp_path, fi.FetchTarget("a1", None), session)

        assert result.split == "train"
        assert session.calls == [_url("train", "a1")]

    def test_not_found_in_every_split(self, tmp_path):
        session = FakeSession({})

        result = _download(tmp_path, fi.FetchTarget("a1", None), session)

        assert result.status == "not_found"
        assert session.calls == [_url(s, "a1") for s in ("train", "validation", "test")]

    def test_fatal_status_stops_probing(self, tmp_path):
        session = FakeSession({_url("train", "a1"): [FakeResponse(status=403, body=b"")]})

        result = _download(tmp_path, fi.FetchTarget("a1", None), session)

        assert result.status == "failed"
        assert session.calls == [_url("train", "a1")]

    def test_retry_applies_within_a_probed_split(self, tmp_path):
        session = FakeSession({
            _url("train", "a1"): [FakeResponse(status=404, body=b"")],
            _url("validation", "a1"): [FakeResponse(status=503, body=b""), FakeResponse()],
        })

        result = _download(tmp_path, fi.FetchTarget("a1", None), session)

        assert result.status == "downloaded" and result.split == "validation"


class TestFetchTargetValidation:

    @pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "a\\b", "", "a.jpg", "x y"])
    def test_unsafe_image_id_is_rejected_before_any_request(self, tmp_path, bad):
        session = FakeSession({})

        with pytest.raises(ValueError):
            _download(tmp_path, fi.FetchTarget(bad, "train"), session)

        assert session.calls == []
        assert list(tmp_path.iterdir()) == []

    def test_unknown_split_value_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _download(tmp_path, fi.FetchTarget("a1", "dev"), FakeSession({}))


class TestFetchMany:

    def test_downloads_all_and_reports_each_result(self, tmp_path):
        ids = [f"id{i:02d}" for i in range(12)]
        session = FakeSession({_url("train", i): [FakeResponse()] for i in ids})
        seen: list[str] = []

        results = fi.fetch_many(
            [fi.FetchTarget(i, "train") for i in ids], tmp_path,
            session=session, workers=4, sleep=lambda s: None,
            on_result=lambda r: seen.append(r.image_id),
        )

        assert [r.image_id for r in results] == ids
        assert all(r.status == "downloaded" for r in results)
        assert sorted(seen) == ids
        assert sorted(p.name for p in tmp_path.glob("*.jpg")) == [f"{i}.jpg" for i in ids]

    def test_one_failure_does_not_stop_the_others(self, tmp_path):
        session = FakeSession({
            _url("train", "good1"): [FakeResponse()],
            _url("train", "bad"): [FakeResponse(status=403, body=b"")],
            _url("train", "good2"): [FakeResponse()],
        })

        results = fi.fetch_many(
            [fi.FetchTarget(i, "train") for i in ("good1", "bad", "good2")], tmp_path,
            session=session, workers=2, sleep=lambda s: None)

        assert [r.status for r in results] == ["downloaded", "failed", "downloaded"]

    def test_rerun_resumes_by_skipping_finished_images(self, tmp_path):
        targets = [fi.FetchTarget(i, "train") for i in ("a1", "b2")]
        first = FakeSession({_url("train", "a1"): [FakeResponse()],
                             _url("train", "b2"): [FakeResponse(status=403, body=b"")]})
        fi.fetch_many(targets, tmp_path, session=first, workers=1, sleep=lambda s: None)
        second = FakeSession({_url("train", "b2"): [FakeResponse()]})

        results = fi.fetch_many(targets, tmp_path, session=second, workers=1, sleep=lambda s: None)

        assert [r.status for r in results] == ["skipped", "downloaded"]
        assert second.calls == [_url("train", "b2")]

    def test_summary_counts_by_status(self):
        results = [
            fi.FetchResult("a", "downloaded", "train"),
            fi.FetchResult("b", "skipped", None),
            fi.FetchResult("c", "not_found", None),
            fi.FetchResult("d", "failed", "train", "HTTP 403"),
            fi.FetchResult("e", "downloaded", "test"),
        ]

        summary = fi.summarize(results)

        assert (summary.downloaded, summary.skipped, summary.not_found, summary.failed) == (2, 1, 1, 1)
        assert summary.ok is False
        assert fi.summarize(results[:2]).ok is True

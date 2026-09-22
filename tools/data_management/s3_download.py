"""Open Images の画像を公開 S3 から 1 枚ずつ取得し、並列に実行する。

    https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg

fiftyone を介さず直接取得する（fiftyone 経由の画像とバイト単位で一致することを SHA-256 で確認済み）。

取得の堅牢性:
    - 既存ファイルはスキップ。`<id>.jpg.<pid>.part` に保存し、Content-Length・サイズ上限・
      JPEG マジックバイトを検証してから rename する（別プロセスの一時ファイルと衝突しない）
    - 5xx・429・408・通信断は指数バックオフでリトライ、400/401/403 は即失敗、404 は次の split を探索
    - リダイレクトは追従しない（別ホスト・非 HTTPS への誘導を避ける）。圧縮転送は要求しない
    - 1 枚が `MAX_IMAGE_BYTES` を超えたら打ち切る（ディスク枯渇の防止）
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import requests

from tools.data_management import _fileio
from tools.data_management.manifest_source import validate_image_id
from tools.data_management.oi_metadata import SPLITS

S3_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
SPLIT_SEARCH_ORDER: tuple[str, ...] = ("train", "validation", "test")

FATAL_STATUSES = frozenset({400, 401, 403})
RETRYABLE_STATUSES = frozenset({408, 429})
JPEG_MAGIC = b"\xff\xd8\xff"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
REQUEST_HEADERS: Mapping[str, str] = MappingProxyType({"Accept-Encoding": "identity"})

DEFAULT_RETRIES = 4
DEFAULT_BACKOFF_SEC = 1.0
DEFAULT_TIMEOUT: Tuple[float, float] = (10, 60)  # (connect, read) 秒
DEFAULT_WORKERS = 8
_CHUNK_SIZE = 1 << 16
_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SEC = 0.2

Sleep = Callable[[float], None]
Timeout = Union[float, Tuple[float, float]]


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...
    def __enter__(self) -> "HttpResponse": ...
    def __exit__(self, *exc: object) -> Optional[bool]: ...


class HttpSession(Protocol):
    def get(self, url: str, *, stream: bool, timeout: Timeout,
            headers: Mapping[str, str], allow_redirects: bool) -> HttpResponse: ...


class FetchStatus(StrEnum):
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    NOT_FOUND = "not_found"
    FAILED = "failed"


class _Kind(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    FATAL = "fatal"
    RETRY = "retry"


@dataclass(frozen=True)
class FetchTarget:
    """取得対象 1 枚。split が None なら S3 を train → validation → test の順に探索する。"""

    image_id: str
    split: Optional[str] = None


@dataclass(frozen=True)
class FetchResult:
    image_id: str
    status: FetchStatus
    split: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class FetchSummary:
    downloaded: int = 0
    skipped: int = 0
    not_found: int = 0
    failed: int = 0

    @property
    def ok(self) -> bool:
        return self.not_found == 0 and self.failed == 0


@dataclass(frozen=True)
class _Attempt:
    kind: _Kind
    detail: str = ""


def image_url(split: str, image_id: str) -> str:
    """公開 S3 の画像 URL。split・Image ID を検証してから組み立てる。"""
    if split not in SPLITS:
        raise ValueError(f"不正な split: {split!r}。有効な値: {list(SPLITS)}")
    return S3_URL_TEMPLATE.format(split=split, image_id=validate_image_id(image_id))


def _save_body(response: HttpResponse, part: Path) -> _Attempt:
    expected = _fileio.expected_length(response.headers)
    if expected is not None and expected > MAX_IMAGE_BYTES:
        return _Attempt(_Kind.FATAL, f"Content-Length {expected} が上限 {MAX_IMAGE_BYTES} bytes を超えています")
    written = 0
    head = b""
    try:
        with open(part, "wb") as f:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_IMAGE_BYTES:
                    return _Attempt(_Kind.FATAL, f"受信サイズが上限 {MAX_IMAGE_BYTES} bytes を超えました")
                if len(head) < len(JPEG_MAGIC):
                    head = (head + chunk)[: len(JPEG_MAGIC)]
                f.write(chunk)
    except requests.RequestException as exc:
        return _Attempt(_Kind.RETRY, f"通信が途中で切れました: {type(exc).__name__}")
    if expected is not None and written != expected:
        return _Attempt(_Kind.RETRY, f"サイズ不一致: {written} / {expected} bytes")
    if head != JPEG_MAGIC:
        return _Attempt(_Kind.FATAL, "JPEG ではありません（マジックバイト不一致）")
    return _Attempt(_Kind.OK)


def _attempt(session: HttpSession, url: str, part: Path, timeout: Timeout) -> _Attempt:
    try:
        response = session.get(url, stream=True, timeout=timeout,
                               headers=REQUEST_HEADERS, allow_redirects=False)
    except requests.RequestException as exc:
        return _Attempt(_Kind.RETRY, f"{type(exc).__name__}: {exc}")
    with response:
        status = response.status_code
        if status == 200:
            return _save_body(response, part)
        if status == 404:
            return _Attempt(_Kind.NOT_FOUND, "HTTP 404")
        if status >= 500 or status in RETRYABLE_STATUSES:
            return _Attempt(_Kind.RETRY, f"HTTP {status}")
        return _Attempt(_Kind.FATAL, f"HTTP {status}")  # 400/401/403・リダイレクト(3xx)・その他 4xx


def _replace_with_retry(part: Path, dest: Path, sleep: Sleep) -> None:
    """一時ファイルを本来の名前へ rename する。

    Windows ではウイルス対策ソフト等が書き込み直後のファイルを一時的に掴み、`PermissionError` に
    なることがあるため、短い待機を挟んで数回再試行する。
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            part.replace(dest)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            sleep(_REPLACE_WAIT_SEC * (attempt + 1))


def _download_from_split(
    image_id: str,
    split: str,
    dest: Path,
    *,
    session: HttpSession,
    retries: int,
    backoff: float,
    timeout: Timeout,
    sleep: Sleep,
) -> FetchResult:
    url = image_url(split, image_id)
    part = _fileio.part_path(dest)
    detail = ""
    try:
        for attempt in range(retries + 1):
            outcome = _attempt(session, url, part, timeout)
            if outcome.kind is _Kind.OK:
                _replace_with_retry(part, dest, sleep)
                return FetchResult(image_id, FetchStatus.DOWNLOADED, split)
            if outcome.kind is _Kind.NOT_FOUND:
                return FetchResult(image_id, FetchStatus.NOT_FOUND, split, outcome.detail)
            detail = outcome.detail
            part.unlink(missing_ok=True)
            if outcome.kind is _Kind.FATAL:
                break
            if attempt < retries:
                sleep(backoff * (2 ** attempt))
        return FetchResult(image_id, FetchStatus.FAILED, split, f"{url}: {detail}")
    finally:
        part.unlink(missing_ok=True)


def download_one(
    target: FetchTarget,
    output_dir: Path,
    *,
    session: HttpSession,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF_SEC,
    timeout: Timeout = DEFAULT_TIMEOUT,
    sleep: Optional[Sleep] = None,
) -> FetchResult:
    """1 枚を取得する。既存ファイルはスキップ。split 不明なら 404 の間 train→validation→test を探索。

    Raises:
        ValueError: Image ID・split が不正（リクエストは一切送らない）。
    """
    image_id = validate_image_id(target.image_id)
    if target.split is not None and target.split not in SPLITS:
        raise ValueError(f"不正な split: {target.split!r}")
    dest = output_dir / f"{image_id}.jpg"
    if dest.exists():
        return FetchResult(image_id, FetchStatus.SKIPPED, target.split)

    sleep = sleep or time.sleep
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in (target.split,) if target.split else SPLIT_SEARCH_ORDER:
            result = _download_from_split(
                image_id, split, dest, session=session, retries=retries,
                backoff=backoff, timeout=timeout, sleep=sleep,
            )
            if result.status is not FetchStatus.NOT_FOUND:
                return result
    except OSError as exc:  # 1 枚のファイル操作エラーで、残りの取得を止めない
        return FetchResult(image_id, FetchStatus.FAILED, target.split, f"{type(exc).__name__}: {exc}")
    return FetchResult(image_id, FetchStatus.NOT_FOUND, target.split, "S3 のどの split にも存在しません")


def fetch_many(
    targets: Sequence[FetchTarget],
    output_dir: Path,
    *,
    session: HttpSession,
    workers: int = DEFAULT_WORKERS,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF_SEC,
    timeout: Timeout = DEFAULT_TIMEOUT,
    sleep: Optional[Sleep] = None,
    on_result: Optional[Callable[[FetchResult], None]] = None,
) -> List[FetchResult]:
    """複数枚を並列取得し、targets と同じ順序で結果を返す。

    1 枚の失敗で他は止めない。中断（Ctrl+C 等）時は未着手を取り消し、完了済みの結果を
    `on_result` に渡してから例外を再送出する（attribution 記録の取りこぼしを防ぐ）。
    """
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures: Dict[Future, int] = {
        pool.submit(
            download_one, target, output_dir, session=session, retries=retries,
            backoff=backoff, timeout=timeout, sleep=sleep,
        ): index
        for index, target in enumerate(targets)
    }
    results: Dict[int, FetchResult] = {}

    def collect(future: Future) -> None:
        result = future.result()
        results[futures[future]] = result
        if on_result is not None:
            on_result(result)

    try:
        for future in as_completed(futures):
            collect(future)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        for future in futures:
            if (futures[future] not in results and future.done()
                    and not future.cancelled() and future.exception() is None):
                collect(future)
        raise
    pool.shutdown(wait=True)
    return [results[i] for i in range(len(targets))]


def summarize(results: Iterable[FetchResult]) -> FetchSummary:
    counts = {status: 0 for status in FetchStatus}
    for result in results:
        counts[FetchStatus(result.status)] += 1
    return FetchSummary(
        downloaded=counts[FetchStatus.DOWNLOADED], skipped=counts[FetchStatus.SKIPPED],
        not_found=counts[FetchStatus.NOT_FOUND], failed=counts[FetchStatus.FAILED],
    )

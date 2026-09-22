"""Open Images の顔画像を公開 S3 から直接取得する（取得計画・実行・CLI）。

取得元（--source）:
    manifest  最新実験（--exp で指定可）の manifest に載っている画像を取得する（既定）。
              データセット復元（restore_dataset.py）用。split が不明なので S3 を train → validation → test の順に探索する。
    all       manifest の画像に加えて、Open Images 全体から oi_filter の条件に合う画像を検索して追加取得する（split は既知）。
              manifest の画像は常に取得する（アノテーションの付け先なので、--max-images の枚数には数えない）。

除外の規則: 手元（--output）にある画像は再取得しない。手元に無い画像は、attribution に登録済みでも取得する
（リポジトリ同梱の attribution は作者の取得履歴で、手元にある画像の記録ではないため）。

画像の取得そのもの（S3 直接取得・検証・リトライ・並列化）は s3_download.py。fiftyone は使わない。

シャード: 対象 ID を先に決めてソートし、シャード分割 → 取得済み除外 → --max-images の順に適用する。
取得済みの有無でシャード境界が動かないので、複数プロセスで `--shard k --shard-count N` を安全に並列実行できる。

attribution: `all` で取得した Open Images 側の画像は著者・出典・ライセンスを attribution CSV に追記する。出力先に画像があるのに
attribution へ未登録の画像（強制終了で記録が漏れた場合など）は、再取得せず attribution だけ補完する。

使い方:
    python tools/data_management/fetch_images.py --dry-run
    python tools/data_management/fetch_images.py --source all --split validation --max-images 500
    python tools/data_management/fetch_images.py --source all --split train --shard 0 --shard-count 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, Union

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.data_management import attribution as attribution_mod  # noqa: E402
from tools.data_management import manifest_source, oi_filter, oi_metadata  # noqa: E402
from tools.data_management._console import make_console_safe  # noqa: E402
from tools.data_management.manifest_source import is_valid_image_id, validate_image_id  # noqa: E402
from tools.data_management.oi_metadata import ImageMeta  # noqa: E402
from tools.data_management.s3_download import (  # noqa: E402
    DEFAULT_RETRIES,
    DEFAULT_WORKERS,
    FetchResult,
    FetchStatus,
    FetchSummary,
    FetchTarget,
    HttpSession,
    Sleep,
    fetch_many,
    summarize,
)

logger = logging.getLogger(__name__)

ESTIMATED_BYTES_PER_IMAGE = 320 * 1024  # --dry-run の容量見積り（1 枚あたり約 320KB）
# --source all で --max-images 省略時の追加取得の上限（誤って全件取得しない）。manifest の画像と手元にある画像は数えない
DEFAULT_ALL_MAX_IMAGES = 500
_ATTRIBUTION_FLUSH_EVERY = 50
_PROGRESS_EVERY = 100

T = TypeVar("T")


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FetchOptions:
    source: str = "manifest"
    # manifest 側の ID を呼び出し側が渡す（manifest を読まない）。空タプルなら manifest 側は空
    image_ids: Optional[Tuple[str, ...]] = None
    exp: Union[int, str, None] = None
    experiments_dir: Path = Path("experiments")
    output_dir: Path = Path("data/images")
    splits: Tuple[str, ...] = ("validation",)
    cache_dir: Path = oi_metadata.DEFAULT_CACHE_DIR
    attribution_csv: Path = attribution_mod.DEFAULT_ATTRIBUTION_CSV
    filter_config: oi_filter.FilterConfig = field(default_factory=oi_filter.FilterConfig)
    refresh_cache: bool = False
    shard: Optional[int] = None
    shard_count: Optional[int] = None
    max_images: Optional[int] = None
    dry_run: bool = False
    workers: int = DEFAULT_WORKERS
    retries: int = DEFAULT_RETRIES


class SourceTargets(NamedTuple):
    """取得元が決めた候補と、計画に効かせる 2 つの ID 集合。"""

    targets: List[FetchTarget]
    metas: Dict[str, ImageMeta]
    skip_ids: Set[str]      # 計画から外す ID（何もすることが無い画像）
    uncapped_ids: Set[str]  # 必ず残し、--max-images の枚数に数えない ID


@dataclass(frozen=True)
class FetchPlan:
    targets: Tuple[FetchTarget, ...]
    metas: Dict[str, ImageMeta]
    n_candidates: int  # シャード適用後・除外前
    n_existing: int    # 出力先に既に画像がある候補数


@dataclass(frozen=True)
class FetchReport:
    plan: FetchPlan
    results: Tuple[FetchResult, ...]
    summary: FetchSummary
    dry_run: bool
    estimated_bytes: int


# ---------------------------------------------------------------------------
# 取得計画（対象 ID の決定 → シャード → 取得済み除外 → 件数制限）
# ---------------------------------------------------------------------------

def shard_slice(items: Sequence[T], shard: Optional[int], shard_count: Optional[int]) -> List[T]:
    """ソート済み `items` を shard_count 個の連続チャンクに分け、shard 番目を返す。"""
    if (shard is None) != (shard_count is None):
        raise ValueError("--shard と --shard-count は両方指定してください")
    if shard is None or shard_count is None:
        return list(items)
    if shard_count < 1:
        raise ValueError(f"--shard-count は 1 以上が必要です: {shard_count}")
    if not 0 <= shard < shard_count:
        raise ValueError(f"--shard は 0 以上 {shard_count - 1} 以下が必要です: {shard}")
    chunk = (len(items) + shard_count - 1) // shard_count
    return list(items[shard * chunk:(shard + 1) * chunk])


def plan_targets(
    targets: Sequence[FetchTarget],
    *,
    shard: Optional[int],
    shard_count: Optional[int],
    skip_ids: Set[str],
    max_images: Optional[int],
    uncapped_ids: AbstractSet[str] = frozenset(),
) -> List[FetchTarget]:
    """重複除去（同じ ID は後のものを採用）・ID ソート → シャード分割 → `skip_ids` 除外 → `max_images` 適用。

    `uncapped_ids` の ID は `max_images` の枚数に数えず、必ず残す（シャードと `skip_ids` は効く）。
    """
    unique = {t.image_id: t for t in targets}
    ordered = [unique[image_id] for image_id in sorted(unique)]
    remaining = [t for t in shard_slice(ordered, shard, shard_count) if t.image_id not in skip_ids]
    if max_images is None:
        return remaining
    planned: List[FetchTarget] = []
    counted = 0
    for target in remaining:
        if target.image_id in uncapped_ids:
            planned.append(target)
        elif counted < max_images:
            planned.append(target)
            counted += 1
    return planned


def _existing_ids(output_dir: Path) -> Set[str]:
    if not output_dir.is_dir():
        return set()
    return {p.stem for p in output_dir.glob("*.jpg")}


def _manifest_ids(options: FetchOptions) -> List[str]:
    """manifest 側の Image ID。`image_ids` があれば検証して使い、無ければ最新（`--exp` 指定ならその）実験の manifest を読む。"""
    if options.image_ids is not None:
        return [validate_image_id(image_id) for image_id in options.image_ids]
    source = manifest_source.find_latest_manifest(options.experiments_dir, options.exp)
    ids = manifest_source.image_ids(manifest_source.load_manifest(source))
    logger.info("manifest: EXP%03d (%d 枚)", source.exp_id, len(ids))
    return ids


def _manifest_source(options: FetchOptions) -> SourceTargets:
    ids = _manifest_ids(options)
    # 除外するのは手元にあるファイルだけ。attribution 登録済みでも、手元に無ければ復元のために取得する。
    return SourceTargets([FetchTarget(i, None) for i in ids], {}, _existing_ids(options.output_dir), set())


def _filtered_targets(options: FetchOptions, split: str) -> Tuple[List[FetchTarget], Dict[str, ImageMeta]]:
    """1 つの split で、ライセンス・注釈条件に合う取得対象とそのメタデータを返す。"""
    config = options.filter_config
    allowed = {oi_metadata.normalize_license(u) for u in config.allowed_licenses}
    split_metas = oi_metadata.fetch_permissive_metadata(
        split, options.cache_dir, refresh=options.refresh_cache)
    candidates = {
        image_id for image_id, meta in split_metas.items()
        if is_valid_image_id(image_id) and oi_metadata.normalize_license(meta.license_url) in allowed
    }

    def compute() -> Set[str]:
        return oi_filter.filter_image_ids(
            candidates,
            oi_metadata.resolve_annotations_csv(split, options.cache_dir, options.refresh_cache),
            oi_metadata.resolve_classes_csv(split, options.cache_dir, options.refresh_cache),
            config,
        )

    ids = {
        image_id for image_id in oi_filter.load_or_compute_filtered_ids(
            split, options.cache_dir, config, compute=compute, refresh=options.refresh_cache)
        if is_valid_image_id(image_id)  # キャッシュ経由でも、不正な ID は取得計画に載せない
    }
    return ([FetchTarget(i, split) for i in ids],
            {i: split_metas[i] for i in ids if i in split_metas})


def _all_source(options: FetchOptions) -> SourceTargets:
    """manifest の画像 + Open Images 全体から条件に合う画像。同じ ID は split が分かる Open Images 側を採用する。"""
    for split in options.splits:
        if split not in oi_metadata.SPLITS:
            raise ValueError(f"不正な split: {split!r}。有効な値: {list(oi_metadata.SPLITS)}")
    manifest_ids = _manifest_ids(options)
    targets: List[FetchTarget] = [FetchTarget(i, None) for i in manifest_ids]
    metas: Dict[str, ImageMeta] = {}
    for split in options.splits:
        split_targets, split_metas = _filtered_targets(options, split)
        targets.extend(split_targets)
        metas.update(split_metas)
    # 手元にあって記録すべき帰属情報が無い画像（attribution 登録済み、または Open Images の候補ではない）は何もしない。
    # 手元にあるが未登録の画像は対象に残し、再取得せず attribution だけ補完する（強制終了などで記録が漏れた画像を
    # 恒久的に取りこぼさないため）。手元に無い画像は、attribution 登録済みでも取得する。
    # 注意: 「候補」は今のフィルタ・split を通る画像だけ。manifest の画像は同梱の attribution に載っている前提で、
    # 載っていない画像が候補から外れていると、帰属情報は補完されない。
    existing = _existing_ids(options.output_dir)
    registered = attribution_mod.known_image_ids(options.attribution_csv)
    skip = {i for i in existing if i in registered or i not in metas}
    return SourceTargets(targets, metas, skip, set(manifest_ids) | existing)


def build_plan(options: FetchOptions) -> FetchPlan:
    """取得元から対象を決め、シャード・除外・件数制限を適用した取得計画を返す（副作用なし）。"""
    if options.source == "manifest":
        source = _manifest_source(options)
        max_images = options.max_images
    elif options.source == "all":
        source = _all_source(options)
        max_images = DEFAULT_ALL_MAX_IMAGES if options.max_images is None else options.max_images
    else:
        raise ValueError(f"不正な source: {options.source!r}（manifest / all）")

    sharded = shard_slice(sorted({t.image_id for t in source.targets}), options.shard, options.shard_count)
    planned = plan_targets(
        source.targets, shard=options.shard, shard_count=options.shard_count,
        skip_ids=source.skip_ids, max_images=max_images, uncapped_ids=source.uncapped_ids,
    )
    existing = _existing_ids(options.output_dir)
    return FetchPlan(
        targets=tuple(planned),
        metas={t.image_id: source.metas[t.image_id] for t in planned if t.image_id in source.metas},
        n_candidates=len(sharded),
        n_existing=sum(1 for image_id in sharded if image_id in existing),
    )


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def _new_session(workers: int) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    return session


class _AttributionRecorder:
    """取得結果のうち attribution 未登録の画像を、まとめて CSV に追記する。"""

    def __init__(self, path: Path, metas: Dict[str, ImageMeta], total: int) -> None:
        self._path = path
        self._metas = metas
        self._total = total
        self._registered = attribution_mod.known_image_ids(path) if metas else set()
        self._pending: List[ImageMeta] = []
        self._done = 0

    def record(self, result: FetchResult) -> None:
        self._done += 1
        if result.status in (FetchStatus.DOWNLOADED, FetchStatus.SKIPPED):
            meta = self._metas.get(result.image_id)
            if meta is not None and result.image_id not in self._registered:
                self._pending.append(meta)
                if len(self._pending) >= _ATTRIBUTION_FLUSH_EVERY:
                    self.flush()
        elif result.status in (FetchStatus.NOT_FOUND, FetchStatus.FAILED):
            logger.warning("取得失敗 %s: %s %s", result.image_id, result.status, result.detail)
        if self._done % _PROGRESS_EVERY == 0:
            logger.info("  %d / %d", self._done, self._total)

    def flush(self) -> None:
        if self._pending:
            attribution_mod.append_attribution(self._path, list(self._pending))
            self._registered.update(m.image_id for m in self._pending)
            self._pending.clear()


def run_fetch(
    options: FetchOptions,
    *,
    session: Optional[HttpSession] = None,
    sleep: Optional[Sleep] = None,
) -> FetchReport:
    """取得計画を立て、dry-run でなければ取得して attribution を追記する。"""
    plan = build_plan(options)
    if options.dry_run:
        missing = [t for t in plan.targets if not (options.output_dir / f"{t.image_id}.jpg").exists()]
        return FetchReport(plan, (), FetchSummary(), True, len(missing) * ESTIMATED_BYTES_PER_IMAGE)

    recorder = _AttributionRecorder(options.attribution_csv, plan.metas, len(plan.targets))
    session = session if session is not None else _new_session(options.workers)
    try:
        results = fetch_many(
            list(plan.targets), options.output_dir, session=session, workers=options.workers,
            retries=options.retries, sleep=sleep, on_result=recorder.record,
        )
    finally:
        recorder.flush()  # 中断・例外でも、取得済み画像の attribution は記録する
    return FetchReport(plan, tuple(results), summarize(results), False, 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open Images の顔画像を公開 S3 から直接取得する。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", choices=("manifest", "all"), default="manifest",
                        help="manifest: 最新実験の manifest の画像（復元用） / "
                             "all: manifest に加えて Open Images 全体から条件に合う画像を追加 (default: manifest)")
    parser.add_argument("--exp", default=None, help="manifest を使う実験番号（例: 32）。省略で最新の実験")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, default=Path("data/images"), help="保存先 (default: data/images)")
    parser.add_argument("--split", nargs="+", choices=oi_metadata.SPLITS, default=["validation"],
                        help="all 用: 追加で検索する split (default: validation)")
    parser.add_argument("--max-images", type=int, default=None, metavar="N",
                        help=f"最大取得枚数（manifest は無制限、all は省略時 {DEFAULT_ALL_MAX_IMAGES}。"
                             "all の上限には manifest の画像と手元にある画像を数えない）")
    parser.add_argument("--shard", type=int, default=None, help="シャード番号（0 始まり、--shard-count と併用）")
    parser.add_argument("--shard-count", type=int, default=None, help="総シャード数（--shard と併用）")
    parser.add_argument("--refresh-cache", action="store_true", help="メタデータ・フィルタ結果のキャッシュを再作成")
    parser.add_argument("--cache-dir", type=Path, default=oi_metadata.DEFAULT_CACHE_DIR)
    parser.add_argument("--attribution-csv", type=Path, default=attribution_mod.DEFAULT_ATTRIBUTION_CSV)
    parser.add_argument("--dry-run", action="store_true", help="取得せず対象数と容量の目安だけ表示")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="並列取得数")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="一時エラー時のリトライ回数")
    filters = parser.add_argument_group("all 用フィルタ条件（oi_filter.FilterConfig）")
    defaults = oi_filter.FilterConfig()
    filters.add_argument("--min-face-area", type=float, default=defaults.min_face_area)
    filters.add_argument("--min-face-aspect", type=float, default=defaults.min_face_aspect)
    filters.add_argument("--max-face-aspect", type=float, default=defaults.max_face_aspect)
    filters.add_argument("--max-faces", type=int, default=defaults.max_faces)
    filters.add_argument("--exclude-children", action="store_true", help="Boy / Girl を含む画像を除外")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> FetchOptions:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers は 1 以上が必要です")
    if args.retries < 0:
        parser.error("--retries は 0 以上が必要です")
    if args.max_images is not None and args.max_images < 0:
        parser.error("--max-images は 0 以上が必要です")
    try:
        shard_slice([], args.shard, args.shard_count)
        config = oi_filter.FilterConfig(
            min_face_area=args.min_face_area, min_face_aspect=args.min_face_aspect,
            max_face_aspect=args.max_face_aspect, max_faces=args.max_faces,
            exclude_children=args.exclude_children,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return FetchOptions(
        source=args.source, exp=args.exp,
        experiments_dir=args.experiments_dir, output_dir=args.output, splits=tuple(args.split),
        cache_dir=args.cache_dir, attribution_csv=args.attribution_csv, filter_config=config,
        refresh_cache=args.refresh_cache, shard=args.shard, shard_count=args.shard_count,
        max_images=args.max_images, dry_run=args.dry_run,
        workers=args.workers, retries=args.retries,
    )


def format_report(report: FetchReport, output_dir: Path) -> List[str]:
    plan = report.plan
    lines = [
        f"取得元の候補: {plan.n_candidates} 枚（うち取得済み {plan.n_existing} 枚）",
        f"取得対象: {len(plan.targets)} 枚 → {output_dir}",
    ]
    if report.dry_run:
        lines.append(
            f"[dry-run] 取得は行いません。容量の目安: {report.estimated_bytes / (1024 * 1024):.1f} MB"
            f"（1 枚あたり約 {ESTIMATED_BYTES_PER_IMAGE // 1024}KB で見積り）"
        )
        return lines
    s = report.summary
    lines.append(
        f"結果: 取得 {s.downloaded} / スキップ {s.skipped} / 未発見 {s.not_found} / 失敗 {s.failed}"
    )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    make_console_safe()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    options = parse_args(argv)
    try:
        report = run_fetch(options)
    except (ValueError, OSError, csv.Error) as exc:  # requests.RequestException は OSError の派生
        logger.error("%s", exc)
        return 2
    for line in format_report(report, options.output_dir):
        print(line)
    return 0 if report.summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())

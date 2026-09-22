"""クローン直後に学習・評価用データ（samples.db・画像）を復元する。

最新実験の manifest を基準に、3 つのステップを順に実行する:

    fetch     画像を Open Images の公開 S3 から data/images/ に取得
    db        manifest から samples.db を再構築（idx・アノテーション・eye_scale・norm_eye_scale）
    features  FaceMesh を再実行して face_yaw/pitch/roll・lap_var を復元

`--steps` で一部だけ実行できる（順序は常に fetch → db → features）。fetch の後に db を実行するのは、
画像が手元にあれば norm_eye_scale まで復元できるため。

`--source` で「何を対象にするか」を 1 つだけ切り替える（画像の取得と DB の再構築が連動する）:

    manifest（既定）  manifest の画像だけ。DB は manifest から再構築する（アノテーション済みの行だけ）
    all               manifest に加えて、Open Images 全体（--split、既定 validation）から条件に合う画像を追加で取得
                      （--max-images は 1 回の実行で追加する上限。既定 500 枚。手元にある画像は数えないので、繰り返すと
                      次の N 枚を追加する）。DB は manifest の行に加えて、手元にあって行が無い画像に未アノテーションの行
                      （t は NULL）を追記する

安全性:
    - db（manifest）: manifest と一致する DB はそのまま。一致しない DB は拒否し、`--overwrite-db` のときだけ
      `samples.db.bak-<時刻>` にバックアップして置換する
    - db（all）: DB が manifest の行をすべて（同じ idx・同じサンプルで）含めばそのまま。DB のみの行と既存の行の値の違い
      （追記した行・自分のアノテーション）は許す。既存の行の値は上書き・削除せず（NULL の norm_eye_scale・特徴量を埋めるだけ）、追記だけを行う
      （何度実行しても同じ結果）。
      新しい行の idx は現在の最大値の次から採番する（idx はその環境の中だけで意味を持つ番号）
    - features（manifest）: DB の全行が manifest に含まれる場合のみ実行（本来の大規模 DB を誤って上書きしない）
    - features（all）: 顔姿勢が NULL の manifest の行を持つ画像だけを処理し、NULL の列だけを埋める。既存の値は上書きしない
      （lap_var は分析用で NULL でもよいので、lap_var だけが NULL の画像は処理しない）
    - db・features のエラーは以降のステップを中止する。fetch の取得失敗は報告のうえ後続を続ける
      （DB は画像なしで作れ、features は画像の無いものを飛ばす）

復元できないもの: 未アノテーション行・skip_flag=1 の行・image_labels テーブル、MediaPipe のバージョン差による
lap_var・顔姿勢の微小な違い。新規に取得した画像に既存アノテーションは付かない。
`features` と `--source all` の行の追記には mediapipe>=0.10.21,<0.10.30（旧 API）が必要。requirements.txt の
PyTorch は CPU 版のため、GPU で学習する場合は別途 GPU 対応版を入れること。

使い方:
    python tools/data_management/restore_dataset.py
    python tools/data_management/restore_dataset.py --dry-run
    python tools/data_management/restore_dataset.py --steps db features
    python tools/data_management/restore_dataset.py --exp 32
    python tools/data_management/restore_dataset.py --source all --max-images 500
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.data_management import attribution as attribution_mod  # noqa: E402
from tools.data_management import fetch_images, oi_metadata  # noqa: E402
from tools.data_management._console import make_console_safe  # noqa: E402
from tools.data_management.manifest_source import (  # noqa: E402
    ManifestError,
    ManifestRow,
    ManifestSource,
    find_latest_manifest,
    image_ids,
    load_manifest,
)
from tools.sampledb_maintenance import add_new_images, rebuild_samples_db, restore_face_pose  # noqa: E402

logger = logging.getLogger(__name__)

STEPS: tuple[str, ...] = ("fetch", "db", "features")
SOURCES: tuple[str, ...] = ("manifest", "all")
StepRunner = Callable[["RestoreOptions", "ManifestSource", Sequence[ManifestRow]], "StepReport"]


@dataclass(frozen=True)
class RestoreOptions:
    steps: tuple[str, ...] = STEPS
    source: str = "manifest"
    splits: tuple[str, ...] = ("validation",)  # all 用: 追加で検索する split
    max_images: Optional[int] = None  # all 用: 追加取得の上限（None は fetch_images の既定 500）
    cache_dir: Path = oi_metadata.DEFAULT_CACHE_DIR  # all 用: Open Images のメタデータのキャッシュ先
    exp: Optional[str] = None
    experiments_dir: Path = Path("experiments")
    db_path: Path = rebuild_samples_db.DEFAULT_DB_PATH
    images_dir: Path = rebuild_samples_db.DEFAULT_IMAGES_DIR
    attribution_csv: Path = attribution_mod.DEFAULT_ATTRIBUTION_CSV
    overwrite_db: bool = False
    dry_run: bool = False
    tol: float = restore_face_pose.DEFAULT_TOLERANCE
    workers: int = fetch_images.DEFAULT_WORKERS
    retries: int = fetch_images.DEFAULT_RETRIES


@dataclass(frozen=True)
class StepReport:
    name: str
    ok: bool
    lines: tuple[str, ...]


@dataclass(frozen=True)
class RestoreReport:
    exp_id: int
    n_rows: int
    n_images: int
    steps: tuple[StepReport, ...]

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)


# ---------------------------------------------------------------------------
# 各ステップ
# ---------------------------------------------------------------------------

def _step_fetch(options: RestoreOptions, source: ManifestSource, rows: Sequence[ManifestRow]) -> StepReport:
    report = fetch_images.run_fetch(fetch_images.FetchOptions(
        source=options.source,
        exp=source.exp_id,
        image_ids=tuple(image_ids(rows)),  # 読み込み済みの manifest から渡す（読み直さない）
        splits=options.splits,
        max_images=options.max_images,
        cache_dir=options.cache_dir,
        experiments_dir=options.experiments_dir,
        output_dir=options.images_dir,
        attribution_csv=options.attribution_csv,
        dry_run=options.dry_run,
        workers=options.workers,
        retries=options.retries,
    ))
    lines = tuple(fetch_images.format_report(report, options.images_dir))
    return StepReport("fetch", report.dry_run or report.summary.ok, lines)


def _step_db(options: RestoreOptions, source: ManifestSource, rows: Sequence[ManifestRow]) -> StepReport:
    extra_rows_ok = options.source == "all"
    result = rebuild_samples_db.rebuild_samples_db(
        rows, options.db_path,
        images_dir=options.images_dir,
        overwrite=options.overwrite_db,
        dry_run=options.dry_run,
        allow_extra_rows=extra_rows_ok,
    )
    lines = [f"{result.action}{'（dry-run）' if result.dry_run else ''}: {result.detail}"]
    if result.backup_path is not None:
        lines.append(f"バックアップ: {result.backup_path}")
    lines.append(f"{result.n_rows} 行 / {result.n_images} 画像 → {options.db_path}")
    ok = result.action is not rebuild_samples_db.RebuildAction.REFUSE
    if extra_rows_ok and ok:
        lines.extend(_add_new_image_rows(options, rows, result))
    return StepReport("db", ok, tuple(lines))


def _add_new_image_rows(
    options: RestoreOptions,
    rows: Sequence[ManifestRow],
    rebuilt: rebuild_samples_db.RebuildResult,
) -> List[str]:
    """`--source all`: manifest の行を用意した DB に、手元にあって行が無い画像の未アノテーション行を追記する。"""
    # dry-run で DB がこれから作られる/置換される場合、現時点の DB は見ず、manifest の画像を「行あり」とみなす。
    pending = options.dry_run and rebuilt.action in (
        rebuild_samples_db.RebuildAction.CREATE, rebuild_samples_db.RebuildAction.REPLACE)
    added = add_new_images.add_new_images(
        options.db_path, options.images_dir, dry_run=options.dry_run,
        known_image_ids=image_ids(rows) if pending else (), db_pending_rebuild=pending,
    )
    return add_new_images.format_result(added)


def _step_features(options: RestoreOptions, source: ManifestSource, rows: Sequence[ManifestRow]) -> StepReport:
    pending = False
    if options.dry_run and "db" in options.steps:
        # 同じ実行の db ステップがこれから DB を作る/置換するので、現時点の DB は検査できない
        plan = rebuild_samples_db.plan_rebuild(
            rows, options.db_path, overwrite=options.overwrite_db, allow_extra_rows=options.source == "all")
        pending = plan.action in (rebuild_samples_db.RebuildAction.CREATE, rebuild_samples_db.RebuildAction.REPLACE)
    result = restore_face_pose.restore_face_features(
        rows, options.db_path, options.images_dir,
        tol=options.tol, dry_run=options.dry_run, db_pending_rebuild=pending,
        fill_missing_only=options.source == "all",
    )
    return StepReport("features", True, tuple(restore_face_pose.format_result(result)))


_RUNNERS: Dict[str, StepRunner] = {
    "fetch": _step_fetch,
    "db": _step_db,
    "features": _step_features,
}

# 例外・失敗時に以降のステップを中止する対象（fetch の不完全さは後続を妨げない）。
_STOP_ON_FAILURE = frozenset({"db", "features"})
_STEP_ERRORS = (
    ManifestError,
    rebuild_samples_db.RebuildError,
    add_new_images.AddImagesError,
    restore_face_pose.FeatureGuardError,
    restore_face_pose.FaceMeshUnavailableError,
    ValueError,
    OSError,
    sqlite3.Error,
    csv.Error,
)


def run_restore(options: RestoreOptions) -> RestoreReport:
    """manifest を 1 度だけ検証して読み込み、選択されたステップを順に実行する。

    Raises:
        ManifestError: 最新（指定）実験の manifest が無い・復元に使えない形式。
    """
    source = find_latest_manifest(options.experiments_dir, options.exp)
    rows = load_manifest(source)
    logger.info("復元の基準: EXP%03d manifest（%d 行 / %d 画像）", source.exp_id, len(rows), len(image_ids(rows)))

    reports: List[StepReport] = []
    for name in options.steps:
        try:
            report = _RUNNERS[name](options, source, rows)
        except _STEP_ERRORS as exc:
            reports.append(StepReport(name, False, (str(exc),)))
            break
        reports.append(report)
        if not report.ok and name in _STOP_ON_FAILURE:
            break
    return RestoreReport(source.exp_id, len(rows), len(image_ids(rows)), tuple(reports))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> RestoreOptions:
    parser = argparse.ArgumentParser(
        description="最新実験の manifest から samples.db・画像を復元する（fetch → db → features）。"
                    "--source all で Open Images の画像の追加取得と未アノテーション行の追記も行う。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS),
                        help="実行するステップ（順序は常に fetch → db → features。default: 全て）")
    parser.add_argument("--source", choices=SOURCES, default="manifest",
                        help="manifest: manifest の画像と行だけ / all: 加えて Open Images から画像を追加取得し、"
                             "手元の画像の未アノテーション行を追記する (default: manifest)")
    parser.add_argument("--split", nargs="+", choices=oi_metadata.SPLITS, default=None,
                        help="all 用: 追加で検索する split (default: validation)")
    parser.add_argument("--max-images", type=int, default=None, metavar="N",
                        help=f"all 用: 1 回の実行で追加取得する最大枚数（省略時 {fetch_images.DEFAULT_ALL_MAX_IMAGES}）。"
                             "manifest の画像と手元にある画像は数えないので、繰り返すと次の N 枚を追加する")
    parser.add_argument("--cache-dir", type=Path, default=oi_metadata.DEFAULT_CACHE_DIR,
                        help="all 用: Open Images のメタデータのキャッシュ先")
    parser.add_argument("--exp", default=None, help="manifest を使う実験番号（例: 32。default: 最新）")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--db", type=Path, default=rebuild_samples_db.DEFAULT_DB_PATH, dest="db_path")
    parser.add_argument("--images-dir", type=Path, default=rebuild_samples_db.DEFAULT_IMAGES_DIR)
    parser.add_argument("--attribution-csv", type=Path, default=attribution_mod.DEFAULT_ATTRIBUTION_CSV)
    parser.add_argument("--overwrite-db", action="store_true",
                        help="manifest と一致しない既存 DB をバックアップ後に置換する")
    parser.add_argument("--dry-run", action="store_true", help="何も書き込まず、実行内容と件数だけ表示")
    parser.add_argument("--tol", type=float, default=restore_face_pose.DEFAULT_TOLERANCE,
                        help="features の画像同一性の許容値（default: %(default)s）")
    parser.add_argument("--workers", type=int, default=fetch_images.DEFAULT_WORKERS, help="fetch の並列数")
    parser.add_argument("--retries", type=int, default=fetch_images.DEFAULT_RETRIES, help="fetch のリトライ回数")
    args = parser.parse_args(argv)
    if args.tol <= 0:
        parser.error("--tol は正の値が必要です")
    if args.workers < 1:
        parser.error("--workers は 1 以上が必要です")
    if args.retries < 0:
        parser.error("--retries は 0 以上が必要です")
    if args.max_images is not None and args.max_images < 0:
        parser.error("--max-images は 0 以上が必要です")
    if args.source == "manifest" and (args.max_images is not None or args.split is not None):
        # manifest の画像を一部しか取得しないと、DB は全行を作るので中途半端な復元になる
        parser.error("--max-images / --split は --source all のときだけ指定できます"
                     "（--source manifest は manifest の画像をすべて取得する）")

    selected = set(args.steps)
    return RestoreOptions(
        steps=tuple(step for step in STEPS if step in selected),
        source=args.source, splits=tuple(args.split) if args.split else ("validation",),
        max_images=args.max_images, cache_dir=args.cache_dir,
        exp=args.exp, experiments_dir=args.experiments_dir, db_path=args.db_path,
        images_dir=args.images_dir, attribution_csv=args.attribution_csv,
        overwrite_db=args.overwrite_db, dry_run=args.dry_run, tol=args.tol,
        workers=args.workers, retries=args.retries,
    )


def print_report(report: RestoreReport, dry_run: bool) -> None:
    print(f"復元の基準: EXP{report.exp_id:03d} manifest（{report.n_rows} 行 / {report.n_images} 画像）"
          + ("  [dry-run]" if dry_run else ""))
    for step in report.steps:
        print(f"[{step.name}] {'OK' if step.ok else '失敗'}")
        for line in step.lines:
            print(f"  {line}")
    print("完了" if report.ok else "一部のステップが失敗しました（上記を確認してください）")


def main(argv: Optional[Sequence[str]] = None) -> int:
    make_console_safe()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    options = parse_args(argv)
    try:
        report = run_restore(options)
    except ManifestError as exc:
        print(f"[manifest] {exc}")
        return 1
    print_report(report, options.dry_run)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

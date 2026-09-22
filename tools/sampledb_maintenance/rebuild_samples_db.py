"""最新実験の manifest から `data/samples.db` を再構築する。

manifest（experiments/EXP{NNN}_*/manifests/manifest_{train,test}.csv）はアノテーション済みの
サンプルを `idx` 付きで持つので、これを基準に DB を復元する:

    idx / image_path / face_index / part_name / sub_key / point_index / px,py,qx,qy / t / nose_x,nose_y
    + eye_scale      （眼ごとの P/Q バウンディングボックス対角長。backfill_db_features.py と同定義）
    + norm_eye_scale （eye_scale / 画像対角長。画像が手元にある場合のみ。後から画像を取得して再実行すれば補完）

face_yaw / face_pitch / face_roll / lap_var は画像が必要なので restore_face_pose.py が復元する。
復元できないもの: 未アノテーション行・skip_flag=1 の行・image_labels テーブル。

既存の DB を壊さないためのガード:
    - manifest と一致する DB はそのまま（復元済みの特徴量も消さない）
    - 一致しない DB は拒否。`--overwrite-db` を付けたときだけ `samples.db.bak-<時刻>` に
      バックアップしてから置換する
    - ロック中など「壊れているわけではない理由で読めない」DB は、推測で置換せずエラーで止める
    - 比較は読み取り専用。置換は一時ファイルで新 DB を作ってから rename する
    - 置換・新規作成の前に、旧 DB の -wal / -shm / -journal を必ず消す（新 DB に再生されるのを防ぐ）

置換の前に annotator など DB を開いているプロセスは止めておくこと（Windows では置換に失敗して中止する）。

使い方:
    python tools/sampledb_maintenance/rebuild_samples_db.py
    python tools/sampledb_maintenance/rebuild_samples_db.py --dry-run
    python tools/sampledb_maintenance/rebuild_samples_db.py --exp 32 --overwrite-db
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from annotator import db as sample_db  # noqa: E402
from tools.data_management._console import make_console_safe  # noqa: E402
from tools.data_management.manifest_source import (  # noqa: E402
    ManifestError,
    ManifestRow,
    find_latest_manifest,
    is_valid_image_id,
    load_manifest,
)
from tools.sampledb_maintenance import _imageinfo  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/samples.db")
DEFAULT_IMAGES_DIR = Path("data/images")
_FLOAT_TOL = 1e-9
_DB_TIMEOUT_SEC = 30
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_CORE_COLUMNS = "idx, image_path, face_index, part_name, sub_key, point_index, px, py, qx, qy, t"
# 「SQLite / samples.db として壊れている」ことを示すエラー。それ以外（ロック・開けない等）は壊れているとは限らない。
_NOT_A_SAMPLES_DB_MARKERS = ("file is not a database", "no such table", "malformed")


class RebuildError(RuntimeError):
    """DB 再構築の失敗。"""


class DbGuardError(RebuildError):
    """既存 DB が manifest と一致せず、`--overwrite-db` も無いため中止した。"""


class RebuildAction(StrEnum):
    CREATE = "create"
    UNCHANGED = "unchanged"
    REPLACE = "replace"
    REFUSE = "refuse"
    CREATED = "created"    # 実行結果
    REPLACED = "replaced"  # 実行結果


class DbCoreRow(NamedTuple):
    """DB 行のうち manifest と比較する部分（主キー・識別子・座標・t）。"""

    idx: int
    image_path: str
    face_index: int
    part_name: str
    sub_key: str
    point_index: int
    px: float
    py: float
    qx: float
    qy: float
    t: float


@dataclass(frozen=True)
class RebuildPlan:
    action: RebuildAction  # CREATE | UNCHANGED | REPLACE | REFUSE
    detail: str
    backup_needed: bool = False


@dataclass(frozen=True)
class RebuildResult:
    action: RebuildAction  # 実行時: CREATED | UNCHANGED | REPLACED / dry-run: RebuildPlan.action
    detail: str
    n_rows: int
    n_images: int
    dry_run: bool = False
    backup_path: Optional[Path] = None
    n_backfilled: int = 0  # 既存 DB の norm_eye_scale を後から補完した行数


# ---------------------------------------------------------------------------
# manifest → samples
# ---------------------------------------------------------------------------

def _image_diagonal(path: Path) -> Optional[float]:
    """画像ヘッダから対角長 [px] を読む。読めなければ理由を警告して None（画像本体は展開しない）。"""
    try:
        width, height = _imageinfo.read_image_size(path)
    except _imageinfo.ImageSizeError as exc:
        logger.warning("画像サイズを読めないため %s の norm_eye_scale は NULL のままにします: %s", path.name, exc)
        return None
    return math.hypot(width, height)


def _eye_scales(rows: Sequence[ManifestRow]) -> Dict[Tuple[str, int, str], float]:
    """(image_path, face_index, part_name) ごとの P/Q バウンディングボックス対角長（下限 1px）。"""
    bounds: Dict[Tuple[str, int, str], List[float]] = {}
    for row in rows:
        key = (row.image_path, row.face_index, row.part_name)
        xs = (row.px, row.qx)
        ys = (row.py, row.qy)
        current = bounds.get(key)
        if current is None:
            bounds[key] = [min(xs), max(xs), min(ys), max(ys)]
        else:
            current[0] = min(current[0], *xs)
            current[1] = max(current[1], *xs)
            current[2] = min(current[2], *ys)
            current[3] = max(current[3], *ys)
    return {
        key: max(math.hypot(b[1] - b[0], b[3] - b[2]), 1.0) for key, b in bounds.items()
    }


def build_samples(
    rows: Sequence[ManifestRow],
    images_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """manifest 行を samples.db の行（dict）へ変換する。入力は変更しない。"""
    eye_scales = _eye_scales(rows)
    diagonals: Dict[str, Optional[float]] = {}
    if images_dir is not None and not _imageinfo.pillow_available():
        logger.warning("Pillow が無いため norm_eye_scale を計算できません（pip install pillow）")
        images_dir = None
    if images_dir is not None:
        for image_id in {row.image_id for row in rows}:
            path = images_dir / f"{image_id}.jpg"
            diagonals[image_id] = _image_diagonal(path) if path.is_file() else None

    samples: List[Dict[str, Any]] = []
    for row in rows:
        eye_scale = eye_scales[(row.image_path, row.face_index, row.part_name)]
        diagonal = diagonals.get(row.image_id)
        samples.append({
            "idx": row.idx,
            "image_path": row.image_path,
            "face_index": row.face_index,
            "part_name": row.part_name,
            "sub_key": row.sub_key,
            "point_index": row.point_index,
            "px": row.px, "py": row.py, "qx": row.qx, "qy": row.qy,
            "t": row.t,
            "skip_flag": False,
            "face_yaw": None, "face_pitch": None, "face_roll": None,
            "eye_scale": eye_scale,
            "norm_eye_scale": eye_scale / diagonal if diagonal else None,
            "lap_var": None,
            "nose_x": row.nose_x, "nose_y": row.nose_y,
        })
    return samples


# ---------------------------------------------------------------------------
# 既存 DB との比較（読み取り専用）
# ---------------------------------------------------------------------------

def _manifest_core(row: ManifestRow) -> DbCoreRow:
    return DbCoreRow(row.idx, row.image_path, row.face_index, row.part_name, row.sub_key,
                     row.point_index, row.px, row.py, row.qx, row.qy, row.t)


def _same_row(a: DbCoreRow, b: DbCoreRow) -> bool:
    for left, right in zip(a, b, strict=True):
        if isinstance(left, float) or isinstance(right, float):
            if left is None or right is None or not math.isclose(
                    left, right, rel_tol=0.0, abs_tol=_FLOAT_TOL):
                return False
        elif left != right:
            return False
    return True


def _readonly_uri(db_path: Path) -> str:
    return f"{Path(db_path).resolve().as_uri()}?mode=ro"


def _is_not_a_samples_db(exc: sqlite3.Error) -> bool:
    """SQLite として壊れている・samples テーブルが無い、という種類のエラーか（ロック等は False）。"""
    message = str(exc).lower()
    return any(marker in message for marker in _NOT_A_SAMPLES_DB_MARKERS)


def read_db_core(db_path: Path) -> Dict[int, DbCoreRow]:
    """DB の主キー・座標・t を読み取り専用で {idx: row} にして返す。

    Raises:
        sqlite3.DatabaseError: SQLite DB でない / samples テーブルが無い / ロック等で読めない。
    """
    with closing(sqlite3.connect(_readonly_uri(db_path), uri=True)) as conn:
        return {r[0]: DbCoreRow._make(r) for r in conn.execute(f"SELECT {_CORE_COLUMNS} FROM samples")}


@dataclass(frozen=True)
class _DbDiff:
    n_db: int
    n_manifest: int
    n_common: int
    n_mismatched: int  # 厳密: 値が違う行 / identity_only: 識別子（自然キー）が違う行
    identity_only: bool = False

    @property
    def only_db(self) -> int:
        return self.n_db - self.n_common

    @property
    def only_manifest(self) -> int:
        return self.n_manifest - self.n_common

    @property
    def identical(self) -> bool:
        """厳密: 完全一致。identity_only: DB が manifest の行をすべて含む（DB のみの行は許す）。"""
        if self.identity_only:
            return self.n_mismatched == self.only_manifest == 0
        return self.n_mismatched == self.only_db == self.only_manifest == 0

    def describe(self) -> str:
        kind = "識別子の不一致" if self.identity_only else "値の不一致"
        return (
            f"既存 DB {self.n_db} 行 / manifest {self.n_manifest} 行"
            f"（共通 {self.n_common}、DB のみ {self.only_db}、manifest のみ {self.only_manifest}、"
            f"{kind} {self.n_mismatched}）"
        )


def _identity(row: DbCoreRow) -> Tuple[Any, ...]:
    """サンプルの自然キー（image_path, face_index, part_name, sub_key, point_index）。"""
    return row[1:6]


def _compare(
    existing: Dict[int, DbCoreRow],
    rows: Sequence[ManifestRow],
    *,
    identity_only: bool = False,
) -> _DbDiff:
    """既存 DB と manifest を idx で突き合わせる。identity_only なら、共通の idx が同じサンプルかだけを見る。"""
    expected = {row.idx: _manifest_core(row) for row in rows}
    common = existing.keys() & expected.keys()
    if identity_only:
        mismatched = sum(1 for idx in common if _identity(existing[idx]) != _identity(expected[idx]))
    else:
        mismatched = sum(1 for idx in common if not _same_row(existing[idx], expected[idx]))
    return _DbDiff(len(existing), len(expected), len(common), mismatched, identity_only)


def plan_rebuild(
    rows: Sequence[ManifestRow],
    db_path: Path,
    *,
    overwrite: bool = False,
    allow_extra_rows: bool = False,
) -> RebuildPlan:
    """既存 DB と manifest を比較し、何をするかを決める（書き込みはしない）。

    Args:
        allow_extra_rows: True なら、DB が manifest の行をすべて（同じ idx・同じ自然キーで）含んでいれば
            「一致」とみなす。DB のみの行（追記した未アノテーション行・第三者のアノテーション）は許し、
            既存の行の値の違いは見ない（既存の行は変更しないため）。

    Raises:
        RebuildError: 既存 DB が壊れているのではなく、ロック等の理由で読めない。
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return RebuildPlan(RebuildAction.CREATE, "DB を新規作成します")
    if db_path.stat().st_size == 0:
        return RebuildPlan(RebuildAction.REPLACE, "既存ファイルは空です")

    try:
        existing = read_db_core(db_path)
    except sqlite3.DatabaseError as exc:
        if not _is_not_a_samples_db(exc):
            raise RebuildError(
                f"既存 DB を読めません（{exc}）。他のプロセスが使用中の可能性があります。DB は変更しません"
            ) from exc
        detail = f"既存ファイルは samples.db として読めません（{exc}）"
        if overwrite:
            return RebuildPlan(RebuildAction.REPLACE, detail, backup_needed=True)
        return RebuildPlan(
            RebuildAction.REFUSE, f"{detail}。置換するには --overwrite-db を指定してください")

    if not existing:
        return RebuildPlan(RebuildAction.REPLACE, "既存 DB は空です")

    diff = _compare(existing, rows, identity_only=allow_extra_rows)
    if diff.identical:
        if allow_extra_rows:
            return RebuildPlan(
                RebuildAction.UNCHANGED,
                f"既存 DB は manifest の行をすべて含んでいます（DB {len(existing)} 行のうち manifest {len(rows)} 行）")
        return RebuildPlan(
            RebuildAction.UNCHANGED, f"既存 DB は manifest と一致しています（{len(existing)} 行）")

    detail = diff.describe()
    if overwrite:
        return RebuildPlan(RebuildAction.REPLACE, detail, backup_needed=True)
    return RebuildPlan(
        RebuildAction.REFUSE,
        f"{detail}。既存 DB は変更しません。置換するには --overwrite-db を指定してください",
    )


# ---------------------------------------------------------------------------
# norm_eye_scale の後補完
# ---------------------------------------------------------------------------

def backfill_norm_eye_scale(db_path: Path, images_dir: Optional[Path]) -> int:
    """norm_eye_scale が NULL の行のうち、画像が手元にあるものを埋め、埋めた行数を返す。

    fetch が一部失敗した状態で db ステップが NULL のまま作った DB を、画像を取得し直したあとの再実行で補う。
    """
    if images_dir is None or not _imageinfo.pillow_available():
        return 0
    filled = 0
    with closing(sqlite3.connect(db_path, timeout=_DB_TIMEOUT_SEC)) as conn:
        try:
            pending = conn.execute(
                "SELECT DISTINCT image_path FROM samples "
                "WHERE norm_eye_scale IS NULL AND eye_scale IS NOT NULL").fetchall()
            for (image_path,) in pending:
                image_id = Path(image_path).stem
                image = images_dir / f"{image_id}.jpg"
                diagonal = _image_diagonal(image) if is_valid_image_id(image_id) and image.is_file() else None
                if diagonal:
                    filled += conn.execute(
                        "UPDATE samples SET norm_eye_scale = eye_scale / ? "
                        "WHERE image_path = ? AND norm_eye_scale IS NULL AND eye_scale IS NOT NULL",
                        (diagonal, image_path)).rowcount
            conn.commit()
        except sqlite3.OperationalError as exc:  # eye_scale 列が無い古い DB など
            logger.warning("norm_eye_scale を補完できませんでした: %s", exc)
            return 0
    return filled


# ---------------------------------------------------------------------------
# バックアップ・置換
# ---------------------------------------------------------------------------

def _backup_path(db_path: Path, now: datetime) -> Path:
    base = db_path.with_name(f"{db_path.name}.bak-{now:%Y%m%d-%H%M%S}")
    candidate, n = base, 0
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.name}-{n}")
    return candidate


def _unlink_sidecars(path: Path) -> None:
    for suffix in _SIDECAR_SUFFIXES:
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _backup(db_path: Path, dest: Path) -> None:
    """一貫したスナップショット（WAL 内の確定済みデータを含む）を `dest` に作る。

    SQLite として壊れているファイルだけは、本体と -wal / -shm をそのままコピーする。
    それ以外の失敗（ロック・ディスクエラー等）は、置換を進めずエラーにする。
    """
    try:
        with closing(sqlite3.connect(_readonly_uri(db_path), uri=True)) as source, \
                closing(sqlite3.connect(dest)) as target:
            source.backup(target)
    except sqlite3.DatabaseError as exc:
        dest.unlink(missing_ok=True)
        if not _is_not_a_samples_db(exc):
            raise RebuildError(f"バックアップに失敗しました（{exc}）。DB は変更しません") from exc
        shutil.copy2(db_path, dest)
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = Path(f"{db_path}{suffix}")
            if sidecar.exists():
                shutil.copy2(sidecar, Path(f"{dest}{suffix}"))


def _write_new_db(samples: List[Dict[str, Any]], path: Path) -> None:
    sample_db.init_db(path)
    sample_db.replace_all_samples(samples, path)
    # annotator.db は `with _connect() as conn` で接続を close せず、循環参照のため GC まで
    # ハンドルが残る。Windows では開いたままだと rename/unlink できないので先に解放する。
    gc.collect()
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _unlink_sidecars(path)


def _install_new_db(samples: List[Dict[str, Any]], db_path: Path, *, creating: bool) -> None:
    """新 DB を一時ファイルに作り、旧 DB の付随ファイルを消してから rename で置き換える。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(f"{db_path.name}.rebuild-{os.getpid()}.tmp")
    try:
        _write_new_db(samples, tmp)
        if creating and db_path.exists():
            raise RebuildError(f"DB の作成中に別のプロセスが {db_path} を作りました。上書きせず中止します")
        _unlink_sidecars(db_path)  # 本体が無くても、孤立した -wal 等が新 DB に再生されないよう必ず消す
        os.replace(tmp, db_path)
    except OSError as exc:
        raise RebuildError(
            f"DB を置換できませんでした（annotator 等が {db_path} を開いていませんか）: {exc}"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
        _unlink_sidecars(tmp)


def rebuild_samples_db(
    rows: Sequence[ManifestRow],
    db_path: Path,
    *,
    images_dir: Optional[Path],
    overwrite: bool = False,
    dry_run: bool = False,
    allow_extra_rows: bool = False,
    now: Callable[[], datetime] = datetime.now,
) -> RebuildResult:
    """manifest から `db_path` を再構築する。

    Args:
        allow_extra_rows: `plan_rebuild` を参照。既存 DB を「manifest の行をすべて含む」で判定する。

    Raises:
        RebuildError: manifest が空 / 既存 DB を安全に扱えない / 置換に失敗（他プロセスが DB を開いている等）。
        DbGuardError: 既存 DB が manifest と一致せず `overwrite` も無い。
    """
    if not rows:
        raise RebuildError("manifest が空です")
    db_path = Path(db_path)
    plan = plan_rebuild(rows, db_path, overwrite=overwrite, allow_extra_rows=allow_extra_rows)
    n_images = len({row.image_id for row in rows})
    base = RebuildResult(plan.action, plan.detail, len(rows), n_images, dry_run=dry_run)

    if dry_run:
        return base
    if plan.action is RebuildAction.REFUSE:
        raise DbGuardError(plan.detail)
    if plan.action is RebuildAction.UNCHANGED:
        return replace(base, n_backfilled=backfill_norm_eye_scale(db_path, images_dir))

    backup: Optional[Path] = None
    if plan.backup_needed:
        backup = _backup_path(db_path, now())
        try:
            _backup(db_path, backup)
        except RebuildError:
            raise
        except (sqlite3.Error, OSError) as exc:  # ディスクエラー等。バックアップできない置換はしない
            raise RebuildError(f"バックアップに失敗しました（{exc}）。DB は変更しません") from exc
        logger.info("既存 DB をバックアップしました: %s", backup)

    creating = plan.action is RebuildAction.CREATE
    _install_new_db(build_samples(rows, images_dir), db_path, creating=creating)
    action = RebuildAction.CREATED if creating else RebuildAction.REPLACED
    logger.info("samples.db を再構築しました (%s): %d 行 / %d 画像 → %s", action, len(rows), n_images, db_path)
    return replace(base, action=action, backup_path=backup)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="最新実験の manifest から samples.db を再構築する。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exp", default=None, help="manifest を使う実験番号（既定: 最新）")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="出力 DB (default: data/samples.db)")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="norm_eye_scale 算出用の画像ディレクトリ (default: data/images)")
    parser.add_argument("--overwrite-db", action="store_true",
                        help="manifest と一致しない既存 DB をバックアップ後に置換する")
    parser.add_argument("--dry-run", action="store_true", help="書き込まず実行内容だけ表示")
    args = parser.parse_args(argv)

    make_console_safe()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        source = find_latest_manifest(args.experiments_dir, args.exp)
        rows = load_manifest(source)
        result = rebuild_samples_db(
            rows, args.db, images_dir=args.images_dir,
            overwrite=args.overwrite_db, dry_run=args.dry_run,
        )
    except (ManifestError, RebuildError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("EXP%03d manifest: %s%s", source.exp_id, result.action, "（dry-run）" if result.dry_run else "")
    logger.info("%s", result.detail)
    if result.n_backfilled:
        logger.info("norm_eye_scale を補完した行: %d", result.n_backfilled)
    return 1 if result.action is RebuildAction.REFUSE else 0


if __name__ == "__main__":
    sys.exit(main())

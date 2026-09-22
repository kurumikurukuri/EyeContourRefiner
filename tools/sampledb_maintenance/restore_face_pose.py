"""samples.db の face_yaw / face_pitch / face_roll / lap_var を FaceMesh の再実行で復元する。

manifest から再構築した DB（rebuild_samples_db.py）には画像由来の特徴量が無い。画像を
MediaPipe FaceMesh に通し直して埋める（annotator/preprocessor.py・build_image_registry.py と同じ定義）:

    face_yaw/pitch/roll : 顔ごとの姿勢（annotator.face_pose.compute_face_orientation）
    lap_var             : 検出した全ての目の領域のラプラシアン分散の最大値（画像単位。images.db の max_lap_var）

画像同一性の検証:
    元の P/Q 座標そのものは MediaPipe のバージョン差で完全には再現できないため、座標群の位置関係で
    「今の画像がアノテーション時と同じ画像か」を照合する。眼ごとに
        manifest の P/Q 点群の重心  と  FaceMesh の目の輪郭ランドマークの重心
    の距離を、その眼の対角長（P/Q バウンディングボックス）で正規化し、両目の最悪値を判定値にする
    （既定の許容値 0.25）。同一画像なら小さく、別画像なら大きくなる。照合できなかった画像には
    face_pose・lap_var のどちらも書き込まない。

誤上書きガード:
    DB の全行が manifest に含まれる場合のみ実行する。本来の大規模 samples.db（未アノテーション行を含む）に、
    復元用の小さな manifest 由来の値を適用しないため。なお対象の画像は毎回再計算して上書きする（冪等）。

fill_missing_only（restore_dataset --source all 用）:
    DB に manifest 以外の行（追記した未アノテーション行・第三者のアノテーション）があってもよい。書き込むのは
    manifest の行のうち特徴量が NULL の列だけで、既存の値は上書きしない（同じ顔の manifest 外の行にも書かない）。
    顔姿勢が揃っている画像は FaceMesh に通さず、通す画像が無ければ FaceMesh（mediapipe）も使わない。
    lap_var は分析用の列（学習には使わない）で NULL でもよいので、揃っているかの判定には使わない（lap_var だけが
    NULL の画像は処理しない。顔姿勢が NULL で処理する画像では、ついでに埋める）。
    ガードは「manifest の idx が別のサンプルに使われていない」ことだけを見る（識別子のすり替えは拒否）。
    顔姿勢を算出できない画像は NULL のまま残るので、実行のたびに FaceMesh に通される（結果は変わらない）。

画像の安全性: 展開前に Pillow でヘッダを読み、`MAX_IMAGE_PIXELS` を超える画素数の画像は処理しない。

要件: MediaPipe の旧 API（`mp.solutions`）を使うため mediapipe>=0.10.21,<0.10.30。

使い方:
    python tools/sampledb_maintenance/restore_face_pose.py
    python tools/sampledb_maintenance/restore_face_pose.py --dry-run --tol 0.25
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from annotator import db as sample_db  # noqa: E402
from annotator.face_pose import compute_face_orientation  # noqa: E402
from tools.data_management._console import make_console_safe  # noqa: E402
from tools.data_management.manifest_source import (  # noqa: E402
    ManifestError,
    ManifestRow,
    find_latest_manifest,
    load_manifest,
)
from tools.sampledb_maintenance import _imageinfo  # noqa: E402
from tools.sampledb_maintenance.rebuild_samples_db import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_IMAGES_DIR,
    read_db_core,
)

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.25
MAX_IMAGE_PIXELS = 200_000_000  # これを超える画素数の画像は FaceMesh に渡さない（RAM 枯渇の防止）
_MARGIN_RATIO = 0.2  # lap_var の目領域バウンディングボックスに足す余白（build_image_registry と同じ）
_COMMIT_EVERY = 100
_DB_TIMEOUT_SEC = 30
_MEDIAPIPE_REQUIREMENT = "mediapipe>=0.10.21,<0.10.30"

# FaceMesh の目の輪郭ランドマーク（build_image_registry の _LEFT/_RIGHT_EYE_CONTOUR と同一）。
# eyeliner_left = 画像左側の目（P/Q 表が 33 番周辺のランドマークを使う側）。
EYE_CONTOUR_LANDMARKS: Mapping[str, Tuple[int, ...]] = MappingProxyType({
    "eyeliner_left": (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7),
    "eyeliner_right": (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382),
})

FaceMeshFactory = Callable[[], AbstractContextManager]
DbUpdate = Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str, int]


class FeatureGuardError(RuntimeError):
    """DB が manifest の範囲を超えている等、復元を実行してはいけない状態。"""


class FaceMeshUnavailableError(RuntimeError):
    """MediaPipe（旧 API）が使えない。"""


class ImageStatus(StrEnum):
    RESTORED = "restored"
    UNVERIFIED = "unverified"
    NO_FACE = "no_face"
    MISSING_IMAGE = "missing_image"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class ImageOutcome:
    image_id: str
    status: ImageStatus
    score: Optional[float] = None


@dataclass(frozen=True)
class FeatureRestoreResult:
    n_images: int
    n_verified: int = 0
    n_unverified: int = 0
    n_no_face: int = 0
    n_missing_image: int = 0
    n_unreadable: int = 0
    n_rows_updated: int = 0
    dry_run: bool = False
    outcomes: Tuple[ImageOutcome, ...] = ()
    n_skipped_complete: int = 0  # fill_missing_only: 顔姿勢が揃っていて処理しなかった画像


@dataclass(frozen=True)
class EyeReference:
    cx: float
    cy: float
    scale: float


@dataclass(frozen=True)
class DetectedFace:
    eye_centroids: Dict[str, Tuple[float, float]]
    pose: Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class ImageAnalysis:
    faces: Tuple[DetectedFace, ...]
    lap_var: Optional[float]


# ---------------------------------------------------------------------------
# FaceMesh
# ---------------------------------------------------------------------------

def create_face_mesh() -> Any:
    """アノテーション時と同じ設定の FaceMesh を作る。

    Raises:
        FaceMeshUnavailableError: mediapipe が無い、または旧 API（mp.solutions）が無い版。
    """
    try:
        import mediapipe as mp  # noqa: PLC0415  重い依存なので features を実行するときだけ読み込む
    except ImportError as exc:
        raise FaceMeshUnavailableError(
            f"mediapipe が見つかりません。`pip install \"{_MEDIAPIPE_REQUIREMENT}\"` "
            "でインストールしてください（0.10.30 以降は旧 API が無いため不可）"
        ) from exc
    face_mesh_module = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_module is None:
        raise FaceMeshUnavailableError(
            f"mediapipe {getattr(mp, '__version__', '?')} には旧 API（mp.solutions）がありません。"
            f"{_MEDIAPIPE_REQUIREMENT} が必要です"
        )
    return face_mesh_module.FaceMesh(
        static_image_mode=True, max_num_faces=10, refine_landmarks=True)


def eye_lap_var(rgb: np.ndarray, points: np.ndarray) -> Optional[float]:
    """目の輪郭点（整数画素）を囲む領域（+20% 余白）のラプラシアン分散。領域が空なら None。"""
    img_h, img_w = rgb.shape[:2]
    x, y, w, h = cv2.boundingRect(points)
    margin = int(min(w, h) * _MARGIN_RATIO)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_w - x, w + 2 * margin)
    h = min(img_h - y, h + 2 * margin)
    if w <= 0 or h <= 0:
        return None
    gray = cv2.cvtColor(rgb[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_image(rgb: np.ndarray, face_mesh: Any) -> Optional[ImageAnalysis]:
    """画像を FaceMesh に通し、顔ごとの目の重心・姿勢と画像の lap_var を返す。顔が無ければ None。"""
    img_h, img_w = rgb.shape[:2]
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    faces: List[DetectedFace] = []
    lap_vars: List[float] = []
    for face_landmarks in result.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        xy = np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks], dtype=np.float64)
        xyz = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks], dtype=np.float64)
        centroids: Dict[str, Tuple[float, float]] = {}
        for part, contour in EYE_CONTOUR_LANDMARKS.items():
            eye = xy[list(contour)]
            centroids[part] = (float(eye[:, 0].mean()), float(eye[:, 1].mean()))
            value = eye_lap_var(rgb, eye.astype(np.int32))
            if value is not None:
                lap_vars.append(value)
        faces.append(DetectedFace(centroids, compute_face_orientation(xyz, img_w, img_h)))
    return ImageAnalysis(tuple(faces), max(lap_vars) if lap_vars else None)


# ---------------------------------------------------------------------------
# 画像同一性の照合
# ---------------------------------------------------------------------------

def build_references(rows: Sequence[ManifestRow]) -> Dict[int, Dict[str, EyeReference]]:
    """1 画像分の manifest 行から {face_index: {part_name: 眼の基準（P/Q 重心・対角長）}}。"""
    groups: Dict[Tuple[int, str], List[ManifestRow]] = defaultdict(list)
    for row in rows:
        groups[(row.face_index, row.part_name)].append(row)

    references: Dict[int, Dict[str, EyeReference]] = defaultdict(dict)
    for (face_index, part), eye_rows in groups.items():
        xs = [v for r in eye_rows for v in (r.px, r.qx)]
        ys = [v for r in eye_rows for v in (r.py, r.qy)]
        scale = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
        references[face_index][part] = EyeReference(sum(xs) / len(xs), sum(ys) / len(ys), scale)
    return dict(references)


def face_score(refs: Dict[str, EyeReference], face: DetectedFace) -> float:
    """注釈済みの顔と検出顔の不一致度。両目の（重心距離 / 眼の対角長）の最悪値。"""
    return max(
        math.hypot(face.eye_centroids[part][0] - ref.cx, face.eye_centroids[part][1] - ref.cy) / ref.scale
        for part, ref in refs.items()
    )


def match_faces(
    references: Dict[int, Dict[str, EyeReference]],
    faces: Sequence[DetectedFace],
) -> Tuple[Optional[Dict[int, int]], Optional[float]]:
    """注釈済みの顔（face_index）と検出顔を、不一致度の小さい順に 1 対 1 で対応付ける。

    Returns:
        (face_index → 検出顔の位置, 最悪の不一致度)。全ての注釈済み顔に検出顔を割り当てられない場合は (None, None)。
    """
    pairs = sorted(
        (face_score(refs, face), face_index, position)
        for face_index, refs in references.items()
        for position, face in enumerate(faces)
    )
    assignment: Dict[int, int] = {}
    used: set[int] = set()
    worst = 0.0
    for score, face_index, position in pairs:
        if face_index in assignment or position in used:
            continue
        assignment[face_index] = position
        used.add(position)
        worst = max(worst, score)
    if len(assignment) < len(references):
        return None, None
    return assignment, worst


# ---------------------------------------------------------------------------
# ガード・復元
# ---------------------------------------------------------------------------

def assert_db_within_manifest(
    rows: Sequence[ManifestRow],
    db_path: Path,
    *,
    allow_extra_rows: bool = False,
) -> FrozenSet[int]:
    """DB の全行が manifest に含まれることを検証する（読み取り専用）。DB の idx 集合を返す。

    Args:
        allow_extra_rows: True なら manifest に無い行があってもよい（識別子のすり替えだけを拒否する）。

    Raises:
        FeatureGuardError: DB が無い・読めない・空、または manifest に無い行を含む（allow_extra_rows 以外）、
            または manifest の idx が別のサンプルに使われている。
    """
    if not Path(db_path).exists():
        raise FeatureGuardError(
            f"samples.db が見つかりません: {db_path}（先に db ステップで manifest から再構築してください）")
    try:
        existing = read_db_core(db_path)
    except sqlite3.DatabaseError as exc:
        raise FeatureGuardError(f"samples.db を読めません: {exc}") from exc
    if not existing:
        raise FeatureGuardError(f"samples.db が空です: {db_path}（先に db ステップを実行してください）")

    identities = {
        r.idx: (r.image_path, r.face_index, r.part_name, r.sub_key, r.point_index) for r in rows
    }
    outside = [idx for idx in existing if idx not in identities]
    swapped = [idx for idx, row in existing.items()
               if idx in identities
               and (row.image_path, row.face_index, row.part_name, row.sub_key, row.point_index) != identities[idx]]
    if allow_extra_rows and swapped:
        raise FeatureGuardError(
            f"samples.db（{len(existing)} 行）に、manifest と識別子が食い違う行が {len(swapped)} 行あります。"
            "manifest の idx が別のサンプルに使われている DB には書き込まないため中止しました")
    if not allow_extra_rows and (outside or swapped):
        raise FeatureGuardError(
            f"samples.db（{len(existing)} 行）に manifest に無い行が {len(outside)} 行、"
            f"manifest と識別子が食い違う行が {len(swapped)} 行あります。特徴量の復元は、DB の全行が"
            "manifest に含まれる場合（manifest から再構築した DB）にだけ実行します。既存のアノテーション DB を"
            "上書きしないため中止しました"
        )
    return frozenset(existing)


def _images_lacking_features(db_path: Path, manifest_idx: FrozenSet[int]) -> FrozenSet[str]:
    """manifest の行のうち、顔姿勢（yaw / pitch / roll）のどれかが NULL の行を持つ画像の ID。読み取り専用。

    lap_var は分析用の列（学習には使わない）で NULL でもよいので、ここでは見ない。
    """
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return frozenset(
            Path(image_path).stem
            for idx, image_path in conn.execute(
                "SELECT idx, image_path FROM samples "
                "WHERE face_yaw IS NULL OR face_pitch IS NULL OR face_roll IS NULL")
            if idx in manifest_idx)


def _read_rgb(path: Path) -> Union[np.ndarray, ImageStatus]:
    """画像を RGB で読む。読めない場合は理由（ImageStatus）を返す。巨大な画像はデコード前に拒否する。"""
    if not path.is_file():
        return ImageStatus.MISSING_IMAGE
    if _imageinfo.pillow_available():
        try:
            width, height = _imageinfo.read_image_size(path)
        except _imageinfo.ImageSizeError as exc:
            logger.warning("画像ヘッダを読めません %s: %s", path.name, exc)
            return ImageStatus.UNREADABLE
        if width * height > MAX_IMAGE_PIXELS:
            logger.warning("画素数が上限を超えるため処理しません %s: %d x %d", path.name, width, height)
            return ImageStatus.UNREADABLE
    try:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return ImageStatus.UNREADABLE
    if bgr is None:
        return ImageStatus.UNREADABLE
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _restore_image(
    image_id: str,
    rows: Sequence[ManifestRow],
    images_dir: Path,
    face_mesh: Any,
    tol: float,
) -> Tuple[ImageOutcome, List[DbUpdate]]:
    rgb = _read_rgb(images_dir / f"{image_id}.jpg")
    if isinstance(rgb, ImageStatus):
        return ImageOutcome(image_id, rgb), []
    analysis = analyze_image(rgb, face_mesh)
    if analysis is None:
        return ImageOutcome(image_id, ImageStatus.NO_FACE), []

    assignment, worst = match_faces(build_references(rows), analysis.faces)
    if assignment is None or worst is None or worst > tol:
        return ImageOutcome(image_id, ImageStatus.UNVERIFIED, worst), []

    image_path = rows[0].image_path
    updates: List[DbUpdate] = []
    for face_index, position in sorted(assignment.items()):
        yaw, pitch, roll = analysis.faces[position].pose or (None, None, None)
        updates.append((yaw, pitch, roll, analysis.lap_var, image_path, face_index))
    return ImageOutcome(image_id, ImageStatus.RESTORED, worst), updates


def _group_by_image(rows: Sequence[ManifestRow], db_idx: FrozenSet[int]) -> Dict[str, List[ManifestRow]]:
    by_image: Dict[str, List[ManifestRow]] = defaultdict(list)
    for row in rows:
        if row.idx in db_idx:
            by_image[row.image_id].append(row)
    return by_image


_OVERWRITE_SQL = (
    "UPDATE samples SET face_yaw = ?, face_pitch = ?, face_roll = ?, lap_var = ? "
    "WHERE image_path = ? AND face_index = ?")


def _fill_missing_sql(n_idx: int) -> str:
    """既存の値は残し、NULL の列だけを埋める。対象は `idx` で指定した行（manifest の行）だけ。

    WHERE の最後の条件で、実際に NULL を埋める行だけが更新行数に数えられる。
    """
    placeholders = ",".join("?" * n_idx)
    return (
        "UPDATE samples SET face_yaw = COALESCE(face_yaw, ?), face_pitch = COALESCE(face_pitch, ?), "
        "face_roll = COALESCE(face_roll, ?), lap_var = COALESCE(lap_var, ?) "
        f"WHERE image_path = ? AND face_index = ? AND idx IN ({placeholders}) "
        "AND (face_yaw IS NULL OR face_pitch IS NULL OR face_roll IS NULL OR lap_var IS NULL)")


def _write_updates(
    conn: sqlite3.Connection,
    updates: Sequence[DbUpdate],
    *,
    idx_by_face: Optional[Mapping[Tuple[str, int], Sequence[int]]] = None,
) -> int:
    """顔ごとの値を書き込み、更新した行数を返す。

    `idx_by_face` が None なら (image_path, face_index) の全行を上書きする。指定があれば（fill_missing_only）、
    その顔のうち `idx` に含まれる行の NULL の列だけを埋める。
    """
    updated = 0
    for update in updates:
        if idx_by_face is None:
            updated += conn.execute(_OVERWRITE_SQL, update).rowcount
            continue
        *values, image_path, face_index = update
        idxs = idx_by_face.get((image_path, face_index), ())
        if idxs:
            updated += conn.execute(
                _fill_missing_sql(len(idxs)), (*values, image_path, face_index, *idxs)).rowcount
    return updated


def _summarize(outcomes: Sequence[ImageOutcome], rows_updated: int) -> FeatureRestoreResult:
    def count(status: ImageStatus) -> int:
        return sum(1 for o in outcomes if o.status == status)

    return FeatureRestoreResult(
        n_images=len(outcomes), n_verified=count(ImageStatus.RESTORED),
        n_unverified=count(ImageStatus.UNVERIFIED), n_no_face=count(ImageStatus.NO_FACE),
        n_missing_image=count(ImageStatus.MISSING_IMAGE), n_unreadable=count(ImageStatus.UNREADABLE),
        n_rows_updated=rows_updated, outcomes=tuple(outcomes),
    )


def restore_face_features(
    rows: Sequence[ManifestRow],
    db_path: Path,
    images_dir: Path,
    *,
    tol: float = DEFAULT_TOLERANCE,
    dry_run: bool = False,
    db_pending_rebuild: bool = False,
    fill_missing_only: bool = False,
    face_mesh_factory: Optional[FaceMeshFactory] = None,
) -> FeatureRestoreResult:
    """FaceMesh を再実行して DB の face_yaw/pitch/roll・lap_var を復元する。

    Args:
        db_pending_rebuild: dry-run 専用。直前の db ステップがこれから DB を作る/置換するため、
            現時点の DB は検査せず、manifest 全体を対象とした件数だけ報告する。
        fill_missing_only: DB に manifest 以外の行があってもよい。特徴量が NULL の列だけを埋め、既存の値は
            上書きしない。顔姿勢が揃っている画像は FaceMesh に通さない（lap_var の NULL は見ない）。

    Raises:
        FeatureGuardError: DB の全行が manifest に含まれない等（DB は変更しない）。
        FaceMeshUnavailableError: mediapipe（旧 API）が使えない。
        ValueError: `db_pending_rebuild` を dry-run 以外で指定した。
    """
    if db_pending_rebuild:
        if not dry_run:
            raise ValueError("db_pending_rebuild は dry_run 専用です")
        return FeatureRestoreResult(n_images=len({r.image_id for r in rows}), dry_run=True)

    by_image = _group_by_image(rows, assert_db_within_manifest(rows, db_path, allow_extra_rows=fill_missing_only))
    n_skipped = 0
    if fill_missing_only:
        lacking = _images_lacking_features(db_path, frozenset(r.idx for r in rows))
        n_skipped = sum(1 for image_id in by_image if image_id not in lacking)
        by_image = {image_id: image_rows for image_id, image_rows in by_image.items() if image_id in lacking}
    if dry_run:
        return FeatureRestoreResult(n_images=len(by_image), dry_run=True, n_skipped_complete=n_skipped)
    if not by_image:  # 何もすることが無いなら、DB にも FaceMesh（mediapipe）にも触れない
        return replace(_summarize([], 0), n_skipped_complete=n_skipped)

    idx_by_face: Optional[Dict[Tuple[str, int], List[int]]] = None
    if fill_missing_only:  # manifest の行だけに書く（同じ顔の manifest 外の行には書かない）
        idx_by_face = defaultdict(list)
        for image_rows in by_image.values():
            for row in image_rows:
                idx_by_face[(row.image_path, row.face_index)].append(row.idx)

    sample_db.init_db(db_path)  # 不足カラムがあれば補う（冪等）
    gc.collect()  # annotator.db が閉じない接続を解放してから書き込む
    outcomes: List[ImageOutcome] = []
    updated = 0
    with closing(sqlite3.connect(db_path, timeout=_DB_TIMEOUT_SEC)) as conn:
        with (face_mesh_factory or create_face_mesh)() as face_mesh:
            for done, (image_id, image_rows) in enumerate(sorted(by_image.items()), start=1):
                outcome, updates = _restore_image(image_id, image_rows, images_dir, face_mesh, tol)
                outcomes.append(outcome)
                updated += _write_updates(conn, updates, idx_by_face=idx_by_face)
                if done % _COMMIT_EVERY == 0:
                    conn.commit()
                    logger.info("  %d / %d 画像", done, len(by_image))
        conn.commit()
    return replace(_summarize(outcomes, updated), n_skipped_complete=n_skipped)


def format_result(result: FeatureRestoreResult) -> List[str]:
    skipped = ([f"顔姿勢が揃っていて飛ばした画像: {result.n_skipped_complete} 枚"]
               if result.n_skipped_complete else [])
    if result.dry_run:
        return [f"[dry-run] FaceMesh の対象画像: {result.n_images} 枚（実行・書き込みは行いません）", *skipped]
    lines = [
        f"画像 {result.n_images} 枚: 復元 {result.n_verified} / 照合不一致 {result.n_unverified} / "
        f"顔未検出 {result.n_no_face} / 画像なし {result.n_missing_image} / 読込不可 {result.n_unreadable}",
        f"更新した行: {result.n_rows_updated}",
        *skipped,
    ]
    unverified = [o for o in result.outcomes if o.status == ImageStatus.UNVERIFIED]
    for outcome in unverified[:10]:
        score = "顔数不足" if outcome.score is None else f"score={outcome.score:.3f}"
        lines.append(f"  照合不一致（書き込まず）: {outcome.image_id} {score}")
    if len(unverified) > 10:
        lines.append(f"  ... ほか {len(unverified) - 10} 枚")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FaceMesh を再実行して samples.db の face_yaw/pitch/roll・lap_var を復元する。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exp", default=None, help="manifest を使う実験番号（既定: 最新）")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE,
                        help=f"画像同一性の許容値（正規化重心距離の最悪値。default: {DEFAULT_TOLERANCE}）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.tol <= 0:
        parser.error("--tol は正の値が必要です")

    make_console_safe()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        rows = load_manifest(find_latest_manifest(args.experiments_dir, args.exp))
        result = restore_face_features(
            rows, args.db, args.images_dir, tol=args.tol, dry_run=args.dry_run)
    except (ManifestError, FeatureGuardError, FaceMeshUnavailableError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return 1
    for line in format_result(result):
        logger.info("%s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

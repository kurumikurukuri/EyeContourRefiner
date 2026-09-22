"""手元にあって samples.db に行が無い画像に、FaceMesh で P/Q 候補の行（未アノテーション）を追記する。

`restore_dataset.py --source all` の db ステップが、manifest の行を用意した後に呼ぶ。

    対象   `data/images/` にある jpg のうち、samples.db に 1 行も無い画像（ID が不正なファイルは除く）
    追記   P/Q 座標・顔番号・顔姿勢・鼻先・eye_scale・norm_eye_scale・lap_var。`t` は NULL（未アノテーション）
    idx    DB の現在の最大値の次から、画像 ID の昇順に画像ごとの連続ブロックで採番する（書き込みロックを取ってから決める）

追記のみ: 既存の行は変更・削除しない。画像ごとに 1 つのトランザクションで書くので、途中で止まっても中途半端な
画像は残らず、再実行すれば残りの画像だけが追記される（何度実行しても同じ結果になる）。

`idx` はその環境の中だけで意味を持つ番号。環境をまたぐ突き合わせは自然キー
（image_path, face_index, part_name, sub_key, point_index）で行うこと。

顔が検出できなかった画像には行が入らないため、実行のたびに解析し直される（結果は変わらない）。

P/Q・顔番号・姿勢・鼻先は annotator/preprocessor.py と同じ定義、lap_var は restore_face_pose.py と同じ定義。
要件: MediaPipe の旧 API（`mp.solutions`）を使うため mediapipe>=0.10.21,<0.10.30。
"""
from __future__ import annotations

import gc
import logging
import math
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from annotator import db as sample_db  # noqa: E402
from annotator.face_pose import compute_face_orientation  # noqa: E402
from annotator.preprocessor import EYELINER_SAMPLES_MAP, _match_face_order  # noqa: E402
from tools.data_management.manifest_source import is_valid_image_id  # noqa: E402
from tools.sampledb_maintenance import restore_face_pose as rfp  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_PATH_TEMPLATE = "data/images/{}.jpg"  # manifest・samples.db と同じ形式
_DB_TIMEOUT_SEC = 30
_LOG_EVERY = 100
_NOSE_TIP_LANDMARK = 4  # MediaPipe face mesh の鼻先（annotator/preprocessor.py と同じ）


class AddImagesError(RuntimeError):
    """samples.db が無い・読めない等、追記を実行できない。"""


@dataclass(frozen=True)
class AddImagesResult:
    n_candidates: int  # 行が無い手元の画像の枚数
    n_added_images: int = 0
    n_added_rows: int = 0
    n_no_face: int = 0
    n_unreadable: int = 0
    first_idx: Optional[int] = None
    last_idx: Optional[int] = None
    dry_run: bool = False


# ---------------------------------------------------------------------------
# 対象画像
# ---------------------------------------------------------------------------

def _image_ids_in_db(db_path: Path) -> set[str]:
    """samples.db に行がある画像の ID（DB が無ければ空）。読み取り専用。"""
    if not db_path.exists():
        return set()
    try:
        with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            return {Path(p).stem for (p,) in conn.execute("SELECT DISTINCT image_path FROM samples")}
    except sqlite3.DatabaseError as exc:
        raise AddImagesError(f"samples.db を読めません: {exc}") from exc


def _image_ids_on_disk(images_dir: Path) -> List[str]:
    if not images_dir.is_dir():
        return []
    return sorted({p.stem for p in images_dir.glob("*.jpg") if is_valid_image_id(p.stem)})


def list_images_without_rows(
    db_path: Path,
    images_dir: Path,
    known_image_ids: Collection[str] = (),
    *,
    db_pending_rebuild: bool = False,
) -> List[str]:
    """手元にあって samples.db に行が無い画像の ID を昇順で返す。

    Args:
        known_image_ids: 行があるものとして扱う画像（dry-run で、同じ実行の直前の手順がこれから作る行の分）。
        db_pending_rebuild: 直前の手順が DB を作る/置換する予定なので、現時点の DB は見ず、
            `known_image_ids` だけを「行がある」とみなす。
    """
    present = set(known_image_ids) if db_pending_rebuild else _image_ids_in_db(Path(db_path)) | set(known_image_ids)
    return [i for i in _image_ids_on_disk(Path(images_dir)) if i not in present]


# ---------------------------------------------------------------------------
# 1 枚の画像 → サンプル
# ---------------------------------------------------------------------------

def _image_lap_var(rgb: np.ndarray, xy_list: List[np.ndarray]) -> Optional[float]:
    """全ての顔・両目の領域のラプラシアン分散の最大値（restore_face_pose.analyze_image と同じ定義）。"""
    values = []
    for xy in xy_list:
        for contour in rfp.EYE_CONTOUR_LANDMARKS.values():
            value = rfp.eye_lap_var(rgb, xy[list(contour)].astype(np.int32))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _with_eye_scale(samples: List[Dict[str, Any]], diagonal: float) -> None:
    """眼ごとの eye_scale（P/Q バウンディングボックス対角長、下限 1px）と norm_eye_scale を、作ったばかりの行に書く。"""
    bounds: Dict[Tuple[int, str], List[float]] = {}
    for s in samples:
        xs, ys = (s["px"], s["qx"]), (s["py"], s["qy"])
        b = bounds.setdefault((s["face_index"], s["part_name"]), [min(xs), max(xs), min(ys), max(ys)])
        b[0], b[1] = min(b[0], *xs), max(b[1], *xs)
        b[2], b[3] = min(b[2], *ys), max(b[3], *ys)
    for s in samples:
        b = bounds[(s["face_index"], s["part_name"])]
        s["eye_scale"] = max(math.hypot(b[1] - b[0], b[3] - b[2]), 1.0)
        s["norm_eye_scale"] = s["eye_scale"] / diagonal


def extract_samples(rgb: np.ndarray, face_mesh: Any, image_id: str) -> Optional[List[Dict[str, Any]]]:
    """1 枚の画像から全ての顔・目の P/Q 行（未アノテーション、idx なし）を作る。顔が検出できなければ None。"""
    height, width = rgb.shape[:2]
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    coords_list: List[np.ndarray] = []
    xyz_list: List[np.ndarray] = []
    for face_landmarks in result.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        coords_list.append(np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32))
        xyz_list.append(np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks], dtype=np.float64))

    lap_var = _image_lap_var(rgb, [xyz[:, :2] for xyz in xyz_list])
    face_indices = _match_face_order(coords_list, {})  # 既存の行が無いので x 座標昇順
    image_path = IMAGE_PATH_TEMPLATE.format(image_id)

    samples: List[Dict[str, Any]] = []
    for coords, xyz, face_index in zip(coords_list, xyz_list, face_indices, strict=True):
        yaw, pitch, roll = compute_face_orientation(xyz, width, height) or (None, None, None)
        nose_x, nose_y = float(coords[_NOSE_TIP_LANDMARK][0]), float(coords[_NOSE_TIP_LANDMARK][1])
        for part_name, sub_map in EYELINER_SAMPLES_MAP.items():
            for sub_key, entries in sub_map.items():
                for point_index, (_, p_idx, q_idx) in enumerate(entries):
                    px, py = coords[p_idx]
                    qx, qy = coords[q_idx]
                    samples.append({
                        "image_path": image_path, "face_index": face_index,
                        "part_name": part_name, "sub_key": sub_key, "point_index": point_index,
                        "px": float(px), "py": float(py), "qx": float(qx), "qy": float(qy),
                        "t": None, "skip_flag": False,
                        "face_yaw": yaw, "face_pitch": pitch, "face_roll": roll,
                        "lap_var": lap_var, "nose_x": nose_x, "nose_y": nose_y,
                    })
    _with_eye_scale(samples, math.hypot(width, height))
    return samples


# ---------------------------------------------------------------------------
# 追記
# ---------------------------------------------------------------------------

def add_new_images(
    db_path: Path,
    images_dir: Path,
    *,
    dry_run: bool = False,
    face_mesh_factory: Optional[Any] = None,
    known_image_ids: Collection[str] = (),
    db_pending_rebuild: bool = False,
) -> AddImagesResult:
    """行が無い手元の画像に、未アノテーションの行を追記する（既存の行は変更しない）。

    Args:
        dry_run: 対象の枚数だけ数える（FaceMesh も書き込みも行わない）。DB が無くてもよい。
        known_image_ids: 行があるものとして扱う画像（dry-run で、直前の手順がこれから作る行の分）。
        db_pending_rebuild: dry-run 専用。直前の手順がこれから DB を作る/置換するため、現時点の DB は見ず、
            `known_image_ids` だけを「行がある」とみなす。

    Raises:
        AddImagesError: DB が無い（dry-run 以外）・読めない。
        FaceMeshUnavailableError: mediapipe（旧 API）が使えない。追記する画像がある場合だけ。
        ValueError: `db_pending_rebuild` を dry-run 以外で指定した。
    """
    if db_pending_rebuild and not dry_run:
        raise ValueError("db_pending_rebuild は dry_run 専用です")
    db_path, images_dir = Path(db_path), Path(images_dir)
    if not db_path.exists() and not dry_run:
        raise AddImagesError(f"samples.db が見つかりません: {db_path}（先に manifest から再構築してください）")
    candidates = list_images_without_rows(
        db_path, images_dir, known_image_ids, db_pending_rebuild=db_pending_rebuild)
    if dry_run or not candidates:
        return AddImagesResult(n_candidates=len(candidates), dry_run=dry_run)

    added_images = added_rows = no_face = unreadable = 0
    first_idx = last_idx = None
    insert_sql = (f"INSERT INTO samples ({sample_db._INSERT_COLUMNS}) "
                  f"VALUES ({sample_db._INSERT_PLACEHOLDERS})")
    with (face_mesh_factory or rfp.create_face_mesh)() as face_mesh:  # mediapipe が使えないなら DB に触れる前に失敗する
        sample_db.init_db(db_path)  # 不足カラムがあれば補う（冪等）
        gc.collect()  # annotator.db が閉じない接続を解放してから書き込む
        with closing(sqlite3.connect(db_path, timeout=_DB_TIMEOUT_SEC)) as conn:
            for done, image_id in enumerate(candidates, start=1):
                rgb = rfp._read_rgb(images_dir / f"{image_id}.jpg")
                if isinstance(rgb, rfp.ImageStatus):
                    unreadable += 1
                    continue
                samples = extract_samples(rgb, face_mesh, image_id)
                if samples is None:
                    no_face += 1
                    continue
                with conn:  # 画像単位のトランザクション（例外なら 1 行も入らない）
                    # 書き込みロックを取ってから idx を決める（別のプロセスが同時に行を足しても衝突しない）
                    conn.execute("BEGIN IMMEDIATE")
                    start_idx = (conn.execute("SELECT MAX(idx) FROM samples").fetchone()[0] or 0) + 1
                    rows = [sample_db._sample_to_row({**s, "idx": start_idx + k}) for k, s in enumerate(samples)]
                    conn.executemany(insert_sql, rows)
                first_idx = start_idx if first_idx is None else first_idx
                last_idx = start_idx + len(rows) - 1
                added_images += 1
                added_rows += len(rows)
                if done % _LOG_EVERY == 0:
                    logger.info("  %d / %d 画像", done, len(candidates))
    return AddImagesResult(
        n_candidates=len(candidates), n_added_images=added_images, n_added_rows=added_rows,
        n_no_face=no_face, n_unreadable=unreadable, first_idx=first_idx, last_idx=last_idx,
    )


def format_result(result: AddImagesResult) -> List[str]:
    if result.dry_run:
        return [f"[dry-run] 行を追記する画像の候補: {result.n_candidates} 枚"
                "（手元にある画像のみ。これから取得する画像は含まない。FaceMesh の実行・書き込みは行いません）"]
    if result.n_candidates == 0:
        return ["行が無い手元の画像はありません（追記なし）"]
    lines = [
        f"追記: {result.n_added_images} 画像 / {result.n_added_rows} 行"
        + (f"（idx {result.first_idx}〜{result.last_idx}）" if result.first_idx is not None else ""),
        f"顔未検出 {result.n_no_face} / 読込不可 {result.n_unreadable}（いずれも行は追記しない）",
    ]
    return lines

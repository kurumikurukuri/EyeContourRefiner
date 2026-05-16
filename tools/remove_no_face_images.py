"""
画像レジストリ DB (data/images.db) を参照し、条件に合致する画像を削除する。

===============================================================================
前提
===============================================================================
事前に `python tools/build_image_registry.py` で images.db を構築しておくこと。
images.db には全画像の has_face / max_eye_scale / max_lap_var / min_face_yaw が
格納されており、フィルタは SQL 一発で完了する（MediaPipe 不要）。

===============================================================================
フィルタ
===============================================================================
  --no-face              顔が 1 つも検出されていない
  --min-eye-scale PX     max_eye_scale < 閾値
  --max-face-yaw DEG     min_face_yaw > 閾値（全顔が横顔すぎる）
  --max-face-pitch DEG   min_face_pitch > 閾値（全顔が上向き/下向きすぎる）
  --max-face-roll DEG    min_face_roll > 閾値（全顔が傾きすぎる）
  --max-faces N          n_faces > 閾値（複数顔を除去）
  --min-lap-var VAR      max_lap_var < 閾値
  --ellipse              楕円フィルタ (scale/66)^2 + (lap/240)^2 < 1
  --ellipse-a A          楕円の scale 軸 (default: 66)
  --ellipse-b B          楕円の lap 軸 (default: 240)
  --max-eye-brightness B max_eye_brightness < 閾値（暗すぎ・サングラス検出: max(全眼) < B）
  --min-eye-brightness B min_eye_brightness > 閾値（明るすぎ・露出過多検出: min(全眼) > B）
                         --max + --min でバンドパスフィルタとして使用可

===============================================================================
使い方
===============================================================================
  # dry-run
  python tools/remove_no_face_images.py --no-face --ellipse --dry-run
  
  #条件
  python tools/remove_no_face_images.py --no-face --ellipse --min-eye-scale 39 --max-face-yaw 45 --max-eye-brightness 20 --min-eye-brightness 185

  # 実際に削除


  # 強制（アノテーション済み含む）
  python tools/remove_no_face_images.py --no-face --ellipse --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from annotator import db as _db  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "data" / "images"
SAMPLES_DB = ROOT / "data" / "samples.db"
REGISTRY_DB = ROOT / "data" / "images.db"
ATTRIBUTION_CSV = ROOT / "data" / "open_images_attribution.csv"


# ---------------------------------------------------------------------------
# Filter queries
# ---------------------------------------------------------------------------

def _build_where(
    no_face: bool = False,
    min_eye_scale: float | None = None,
    max_face_yaw: float | None = None,
    max_face_pitch: float | None = None,
    max_face_roll: float | None = None,
    max_n_faces: int | None = None,
    min_lap_var: float | None = None,
    ellipse_a: float | None = None,
    ellipse_b: float | None = None,
    max_eye_brightness: float | None = None,
    min_eye_brightness: float | None = None,
) -> str:
    """レジストリ DB に対する WHERE 句を構築する。"""
    conditions: list[str] = []

    if no_face:
        conditions.append("has_face = 0")

    if min_eye_scale is not None:
        conditions.append(f"(max_eye_scale IS NULL OR max_eye_scale < {min_eye_scale})")

    if max_face_yaw is not None:
        conditions.append(f"(min_face_yaw IS NOT NULL AND min_face_yaw > {max_face_yaw})")

    if max_face_pitch is not None:
        conditions.append(f"(min_face_pitch IS NOT NULL AND min_face_pitch > {max_face_pitch})")

    if max_face_roll is not None:
        conditions.append(f"(min_face_roll IS NOT NULL AND min_face_roll > {max_face_roll})")

    if max_n_faces is not None:
        conditions.append(f"(n_faces IS NOT NULL AND n_faces > {max_n_faces})")

    if min_lap_var is not None:
        conditions.append(f"(max_lap_var IS NULL OR max_lap_var < {min_lap_var})")

    if ellipse_a is not None and ellipse_b is not None:
        conditions.append(
            f"(max_eye_scale IS NULL OR max_lap_var IS NULL OR "
            f"(max_eye_scale / {ellipse_a}) * (max_eye_scale / {ellipse_a}) + "
            f"(max_lap_var / {ellipse_b}) * (max_lap_var / {ellipse_b}) < 1)"
        )

    if max_eye_brightness is not None:
        # max(全眼) < threshold: 最も明るい目でも暗い → サングラス
        conditions.append(
            f"(has_face = 1 AND "
            f"(max_eye_brightness IS NULL OR max_eye_brightness < {max_eye_brightness}))"
        )

    if min_eye_brightness is not None:
        # min(全眼) > threshold: 最も暗い目でも明るい → 露出過多
        conditions.append(
            f"(has_face = 1 AND "
            f"(min_eye_brightness IS NOT NULL AND min_eye_brightness > {min_eye_brightness}))"
        )

    if not conditions:
        return "1 = 0"  # no filter → empty result
    return "(" + ") OR (".join(conditions) + ")"


def query_delete_candidates(
    conn: sqlite3.Connection,
    no_face: bool = False,
    min_eye_scale: float | None = None,
    max_face_yaw: float | None = None,
    max_face_pitch: float | None = None,
    max_face_roll: float | None = None,
    max_n_faces: int | None = None,
    min_lap_var: float | None = None,
    ellipse_a: float | None = None,
    ellipse_b: float | None = None,
    max_eye_brightness: float | None = None,
    min_eye_brightness: float | None = None,
) -> list[dict]:
    """削除候補の image_path リストを返す。"""
    where = _build_where(
        no_face=no_face,
        min_eye_scale=min_eye_scale,
        max_face_yaw=max_face_yaw,
        max_face_pitch=max_face_pitch,
        max_face_roll=max_face_roll,
        max_n_faces=max_n_faces,
        min_lap_var=min_lap_var,
        ellipse_a=ellipse_a,
        ellipse_b=ellipse_b,
        max_eye_brightness=max_eye_brightness,
        min_eye_brightness=min_eye_brightness,
    )
    sql = (
        f"SELECT image_path, has_face, max_eye_scale, max_lap_var, "
        f"min_face_yaw, min_face_pitch, min_face_roll, n_faces, "
        f"min_eye_brightness FROM images WHERE {where}"
    )
    log.info("query: %s", sql)
    rows = conn.execute(sql).fetchall()
    return [
        {
            "image_path": r[0],
            "has_face": r[1],
            "max_eye_scale": r[2],
            "max_lap_var": r[3],
            "min_face_yaw": r[4],
            "min_face_pitch": r[5],
            "min_face_roll": r[6],
            "n_faces": r[7],
            "min_eye_brightness": r[8],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Safety / data cleanup
# ---------------------------------------------------------------------------

def _get_annotated_db_paths() -> set[str]:
    """samples.db で t IS NOT NULL の image_path の basename セット。"""
    if not SAMPLES_DB.exists():
        return set()
    return {Path(s["image_path"]).name for s in _db.get_annotated_samples(SAMPLES_DB)}


def _remove_from_samples_db(image_paths: list[str], dry_run: bool) -> int:
    """image_paths (basename) を samples.db から削除する（バッチ処理）。"""
    if not SAMPLES_DB.exists() or not image_paths:
        return 0
    full_paths = [f"data/images/{p}" for p in image_paths]
    log.info("samples.db: %d 画像のエントリを削除予定", len(full_paths))
    if not dry_run:
        batch_size = 500
        for i in range(0, len(full_paths), batch_size):
            batch = full_paths[i:i + batch_size]
            _db.delete_samples_by_image_paths(batch, SAMPLES_DB)
        log.info("samples.db: 削除完了")
    return len(full_paths)


def _remove_from_registry(conn: sqlite3.Connection, image_paths: list[str], dry_run: bool) -> int:
    if not image_paths:
        return 0
    log.info("registry: %d 画像を削除予定", len(image_paths))
    if not dry_run:
        conn.executemany(
            "DELETE FROM images WHERE image_path = ?",
            [(p,) for p in image_paths],
        )
        conn.commit()
    return len(image_paths)


def _remove_from_attribution_csv(stems: set[str], dry_run: bool) -> int:
    if not ATTRIBUTION_CSV.exists():
        return 0
    with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept = [r for r in rows if r.get("image_id", "") not in stems]
    removed = len(rows) - len(kept)
    log.info("attribution.csv: %d 行削除 (残り %d)", removed, len(kept))

    if not dry_run and removed > 0:
        ATTRIBUTION_CSV.write_text(
            "\n".join(
                [",".join(fieldnames)]
                + [",".join(r.get(f, "") for f in fieldnames) for r in kept]
            ),
            encoding="utf-8",
        )
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dry_run: bool = False,
    force: bool = False,
    no_face: bool = False,
    min_eye_scale: float | None = None,
    max_face_yaw: float | None = None,
    max_face_pitch: float | None = None,
    max_face_roll: float | None = None,
    max_n_faces: int | None = None,
    min_lap_var: float | None = None,
    ellipse: bool = False,
    ellipse_a: float = 66,
    ellipse_b: float = 240,
    max_eye_brightness: float | None = None,
    min_eye_brightness: float | None = None,
) -> None:
    if not REGISTRY_DB.exists():
        log.error(
            "registry DB が見つかりません: %s\n"
            "  先に `python tools/build_image_registry.py` を実行してください。",
            REGISTRY_DB,
        )
        return

    # ellispe flag → set a, b
    ea = ellipse_a if ellipse else None
    eb = ellipse_b if ellipse else None

    conn = sqlite3.connect(str(REGISTRY_DB))
    conn.execute("PRAGMA read_uncommitted = true")

    # --- query ---
    candidates = query_delete_candidates(
        conn,
        no_face=no_face,
        min_eye_scale=min_eye_scale,
        max_face_yaw=max_face_yaw,
        max_face_pitch=max_face_pitch,
        max_face_roll=max_face_roll,
        max_n_faces=max_n_faces,
        min_lap_var=min_lap_var,
        ellipse_a=ea,
        ellipse_b=eb,
        max_eye_brightness=max_eye_brightness,
        min_eye_brightness=min_eye_brightness,
    )

    if not candidates:
        log.info("削除候補はありません。")
        conn.close()
        return

    image_paths = [c["image_path"] for c in candidates]
    log.info("削除候補: %d 枚", len(image_paths))

    # detail
    for c in candidates[:20]:
        log.info(
            "  %s  face=%d  scale=%s  lap=%s  yaw=%s  pitch=%s  roll=%s  n=%s  bright=%s",
            c["image_path"],
            c["has_face"],
            f'{c["max_eye_scale"]:.0f}' if c["max_eye_scale"] is not None else "-",
            f'{c["max_lap_var"]:.0f}' if c["max_lap_var"] is not None else "-",
            f'{c["min_face_yaw"]:.1f}' if c["min_face_yaw"] is not None else "-",
            f'{c["min_face_pitch"]:.1f}' if c["min_face_pitch"] is not None else "-",
            f'{c["min_face_roll"]:.1f}' if c["min_face_roll"] is not None else "-",
            str(c["n_faces"]) if c["n_faces"] is not None else "-",
            f'{c["min_eye_brightness"]:.0f}' if c["min_eye_brightness"] is not None else "-",
        )
    if len(candidates) > 20:
        log.info("  ... and %d more", len(candidates) - 20)

    # --- safety check ---
    annotated = _get_annotated_db_paths()
    annotated_in_delete = [p for p in image_paths if p in annotated]
    if annotated_in_delete and not force:
        log.error(
            "削除候補に %d 枚のアノテーション済み画像が含まれています。",
            len(annotated_in_delete),
        )
        for p in sorted(annotated_in_delete):
            log.error("  %s", Path(p).name)
        log.error("--force を付けると強制的に削除できます。")
        conn.close()
        return

    if annotated_in_delete:
        log.warning(
            "%d 枚のアノテーション済み画像が削除されます (--force)。",
            len(annotated_in_delete),
        )

    # --- execute ---
    stems = {Path(p).stem for p in image_paths}
    _remove_from_samples_db(image_paths, dry_run)
    _remove_from_registry(conn, image_paths, dry_run)
    _remove_from_attribution_csv(stems, dry_run)

    if dry_run:
        log.info("[dry-run] %d 枚の画像は削除されません。", len(image_paths))
    else:
        deleted = 0
        for p in image_paths:
            fp = IMAGES_DIR / p
            if fp.exists():
                fp.unlink()
                deleted += 1
        log.info("完了: %d 枚の画像を削除しました。", deleted)

    conn.close()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="レジストリ DB 参照で画像を削除する")
    parser.add_argument("--no-face", action="store_true", help="顔なし画像を削除")
    parser.add_argument("--min-eye-scale", type=float, default=None, metavar="PX")
    parser.add_argument("--max-face-yaw", type=float, default=None, metavar="DEG")
    parser.add_argument("--max-face-pitch", type=float, default=None, metavar="DEG")
    parser.add_argument("--max-face-roll", type=float, default=None, metavar="DEG")
    parser.add_argument("--max-faces", type=int, default=None, metavar="N",
                        help="n_faces > 閾値の画像を削除（複数顔）")
    parser.add_argument("--min-lap-var", type=float, default=None, metavar="VAR")
    parser.add_argument("--ellipse", action="store_true", help="楕円フィルタ有効")
    parser.add_argument("--ellipse-a", type=float, default=66, metavar="A")
    parser.add_argument("--ellipse-b", type=float, default=240, metavar="B")
    parser.add_argument(
        "--max-eye-brightness", type=float, default=None, metavar="BRIGHT",
        help="min_eye_brightness < 閾値の画像を削除（暗すぎ・サングラス検出、推奨: 80）",
    )
    parser.add_argument(
        "--min-eye-brightness", type=float, default=None, metavar="BRIGHT",
        help="min_eye_brightness > 閾値の画像を削除（明るすぎ・露出過多検出、推奨: 220）",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    main(
        dry_run=args.dry_run,
        force=args.force,
        no_face=args.no_face,
        min_eye_scale=args.min_eye_scale,
        max_face_yaw=args.max_face_yaw,
        max_face_pitch=args.max_face_pitch,
        max_face_roll=args.max_face_roll,
        max_n_faces=args.max_faces,
        min_lap_var=args.min_lap_var,
        ellipse=args.ellipse,
        ellipse_a=args.ellipse_a,
        ellipse_b=args.ellipse_b,
        max_eye_brightness=args.max_eye_brightness,
        min_eye_brightness=args.min_eye_brightness,
    )

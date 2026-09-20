"""samples.db に eye_scale / norm_eye_scale / lap_var を一括バックフィルする。

eye_scale    : (image_path, face_index, part_name) ごとのPQバウンディングボックス対角長
               ── PQ座標からSQL集約で計算（画像ファイル不要）
norm_eye_scale: eye_scale / sqrt(img_w^2 + img_h^2)
               ── PILで画像ヘッダを読んで計算
lap_var      : 目領域ラプラシアン分散（ブレ指標）
               ── images.db の max_lap_var を per-image として転用

Usage:
    python tools/imagesdb_maintenance/backfill_db_features.py [--db data/samples.db] [--images-db data/images.db]
    python tools/imagesdb_maintenance/backfill_db_features.py --skip-dims  # norm_eye_scale を省略（高速）
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

SAMPLES_DB  = Path("data/samples.db")
IMAGES_DB   = Path("data/images.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_NEW_COLUMNS = [
    ("eye_scale",      "REAL"),
    ("norm_eye_scale", "REAL"),
    ("lap_var",        "REAL"),
]


def ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
    for col, typ in _NEW_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {col} {typ}")
            log.info("Added column: %s %s", col, typ)
    conn.commit()


# ---------------------------------------------------------------------------
# Step 1: eye_scale per (image_path, face_index, part_name) from PQ coords
# ---------------------------------------------------------------------------

_SQL_EYE_SCALE = """
SELECT image_path, face_index, part_name,
    SQRT(POWER(
        MAX(CASE WHEN px > qx THEN px ELSE qx END) -
        MIN(CASE WHEN px < qx THEN px ELSE qx END), 2
    ) + POWER(
        MAX(CASE WHEN py > qy THEN py ELSE qy END) -
        MIN(CASE WHEN py < qy THEN py ELSE qy END), 2
    )) AS eye_scale
FROM samples
GROUP BY image_path, face_index, part_name
"""


def compute_eye_scale_per_group(
    conn: sqlite3.Connection,
) -> dict[tuple[str, int, str], float]:
    log.info("Computing eye_scale per group (SQL aggregation)...")
    rows = conn.execute(_SQL_EYE_SCALE).fetchall()
    result: dict[tuple[str, int, str], float] = {}
    for image_path, face_index, part_name, es in rows:
        result[(image_path, int(face_index), part_name)] = max(float(es), 1.0)
    log.info("  %d groups computed", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 2: image diagonal from PIL (parallel header-only read)
# ---------------------------------------------------------------------------

def _image_size(image_path_str: str) -> tuple[str, float]:
    """Returns (image_path, diagonal_px). diagonal=0.0 on error."""
    try:
        from PIL import Image
        with Image.open(image_path_str) as im:
            w, h = im.size
        return image_path_str, math.sqrt(w * w + h * h)
    except Exception:
        return image_path_str, 0.0


def load_image_diagonals(
    image_paths: list[str],
    workers: int = 8,
) -> dict[str, float]:
    log.info("Reading image dimensions for %d images...", len(image_paths))
    result: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_image_size, p): p for p in image_paths}
        done = 0
        for f in as_completed(futures):
            ip, diag = f.result()
            if diag > 0:
                result[ip] = diag
            done += 1
            if done % 20_000 == 0:
                log.info("  %d / %d", done, len(image_paths))
    log.info("  %d images with valid dimensions", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 3: lap_var from images.db (max_lap_var per image)
# ---------------------------------------------------------------------------

def load_lap_var(images_db: Path) -> dict[str, float]:
    """Returns {image_filename: max_lap_var}. Key = basename only (e.g. 'abc.jpg')."""
    log.info("Loading lap_var from %s...", images_db)
    conn = sqlite3.connect(str(images_db))
    rows = conn.execute(
        "SELECT image_path, max_lap_var FROM images WHERE max_lap_var IS NOT NULL"
    ).fetchall()
    conn.close()
    # images.db の image_path はファイル名のみ（ディレクトリなし）
    result = {row[0]: float(row[1]) for row in rows}
    log.info("  %d entries loaded", len(result))
    return result


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def backfill(
    samples_db: Path = SAMPLES_DB,
    images_db: Path  = IMAGES_DB,
    skip_dims: bool  = False,
    overwrite: bool  = False,
) -> None:
    conn = sqlite3.connect(str(samples_db), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")  # 128 MB

    ensure_columns(conn)

    # どの行を更新するか確認
    if not overwrite:
        need = conn.execute(
            "SELECT COUNT(*) FROM samples WHERE eye_scale IS NULL OR lap_var IS NULL"
        ).fetchone()[0]
        log.info("Rows needing update: %d", need)
        if need == 0:
            log.info("All rows already populated. Use --overwrite to force.")
            conn.close()
            return

    # eye_scale
    eye_scale_map = compute_eye_scale_per_group(conn)

    # norm_eye_scale（画像サイズ必要）
    diag_map: dict[str, float] = {}
    if not skip_dims:
        unique_paths = sorted({k[0] for k in eye_scale_map})
        diag_map = load_image_diagonals(unique_paths)

    # lap_var
    lap_var_map: dict[str, float] = {}
    if images_db.exists():
        lap_var_raw = load_lap_var(images_db)
        # samples.db の image_path は 'data/images/xxx.jpg' 形式 → basename で照合
        lap_var_map = {Path(k).name if '/' not in k and '\\' not in k else k: v
                       for k, v in lap_var_raw.items()}
    else:
        log.warning("images.db not found, lap_var will be NULL")

    # UPDATE データ組み立て
    log.info("Building update records...")
    update_rows: list[tuple] = []
    for (image_path, face_index, part_name), eye_scale in eye_scale_map.items():
        diag = diag_map.get(image_path, 0.0)
        norm_eye_scale = eye_scale / diag if diag > 0 else None
        basename = Path(image_path).name
        lap_var = lap_var_map.get(basename)
        update_rows.append((eye_scale, norm_eye_scale, lap_var, image_path, face_index, part_name))

    log.info("Updating %d groups (%d rows each on average)...",
             len(update_rows),
             conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] // max(len(update_rows), 1))

    conn.executemany(
        "UPDATE samples SET eye_scale=?, norm_eye_scale=?, lap_var=? "
        "WHERE image_path=? AND face_index=? AND part_name=?",
        update_rows,
    )
    conn.commit()

    # 確認
    filled = conn.execute(
        "SELECT COUNT(*) FROM samples WHERE eye_scale IS NOT NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    log.info("Done. eye_scale filled: %d / %d rows", filled, total)
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",         default=str(SAMPLES_DB), help="samples.db path")
    parser.add_argument("--images-db",  default=str(IMAGES_DB),  help="images.db path")
    parser.add_argument("--skip-dims",  action="store_true",      help="skip norm_eye_scale (no image reads)")
    parser.add_argument("--overwrite",  action="store_true",      help="update even if already filled")
    args = parser.parse_args()

    backfill(
        samples_db=Path(args.db),
        images_db=Path(args.images_db),
        skip_dims=args.skip_dims,
        overwrite=args.overwrite,
    )

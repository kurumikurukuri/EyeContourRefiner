"""
MediaPipe で顔が検出できなかった画像、および eye_scale が小さすぎる画像を削除するスクリプト。

===============================================================================
概要
===============================================================================
このスクリプトは、画像データセットの品質管理（クリーニング）を目的としています。

主な役割:
  - 顔が写っていない画像を除外
  - 顔はあるが目が小さすぎて学習用途に不向きな画像を除外
  - 削除対象画像に対応する DB / attribution 情報も同期削除

これにより、顔解析モデルや目関連モデルの学習データ品質を向上できます。

-------------------------------------------------------------------------------
処理フロー
-------------------------------------------------------------------------------
1. 指定ディレクトリ内の画像を列挙
2. MediaPipe FaceDetection + FaceMesh で顔検出
3. 顔が検出できない画像を削除候補に追加
4. （--min-eye-scale 指定時）
   DB 内の eye_scale 情報を参照し、
   「画像内で最も大きい eye_scale ですら閾値未満」の画像を削除候補に追加
5. samples.db から該当サンプル削除
6. open_images_attribution.csv から該当 image_id 削除
7. dry-run でなければ実ファイル削除

-------------------------------------------------------------------------------
使い方
-------------------------------------------------------------------------------
# 顔なし画像のみ確認
python tools/remove_no_face_images.py --dry-run

# 顔なし + eye_scale フィルタ確認
python tools/remove_no_face_images.py --min-eye-scale 30 --dry-run

# 実際に削除
python tools/remove_no_face_images.py --min-eye-scale 30

# 別ディレクトリ指定
python tools/remove_no_face_images.py --images path/to/images/ --min-eye-scale 30
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from annotator import db as _db  # noqa: E402

log = logging.getLogger(__name__)

ROOT            = Path(__file__).parent.parent
IMAGES_DIR      = ROOT / "data" / "images"
DB_PATH         = ROOT / "data" / "samples.db"
ATTRIBUTION_CSV = ROOT / "data" / "open_images_attribution.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def _has_face(image_path: Path) -> bool:
    """MediaPipe FaceMesh で顔が 1 人以上検出できるか確認する。"""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe が見つかりません。`pip install mediapipe` でインストールしてください。"
        ) from exc

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        log.warning("画像を読み込めません: %s", image_path)
        return False

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as detector:
        det = detector.process(rgb)
        if not det.detections:
            log.info("  顔なし: %s", image_path.name)
            return False

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    detected = bool(results.multi_face_landmarks)
    if not detected:
        log.info("  顔なし: %s", image_path.name)
    return detected


def scan_images(images_dir: Path) -> tuple[list[Path], list[Path]]:
    """images_dir 内の画像を走査し (顔あり, 顔なし) に分類して返す。"""
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        log.warning("画像が見つかりません: %s", images_dir)
        return [], []

    log.info("%d 枚の画像を検査します...", len(image_paths))
    with_face: list[Path] = []
    no_face:   list[Path] = []

    for i, p in enumerate(image_paths, 1):
        log.info("[%d/%d] %s", i, len(image_paths), p.name)
        if _has_face(p):
            with_face.append(p)
        else:
            no_face.append(p)

    return with_face, no_face


# ---------------------------------------------------------------------------
# eye_scale filter
# ---------------------------------------------------------------------------

def find_small_eye_images(min_eye_scale: float) -> list[str]:
    """DB から全画像の最大 eye_scale を計算し、閾値未満の image_path リストを返す。

    各画像について全 (face_index, part_name) の最大 eye_scale を代表値とする。
    最大値でも閾値未満 = その画像に含まれる全ての目が小さすぎる。
    """
    max_scale_per_image = _db.get_max_eye_scale_per_image(DB_PATH)
    return [
        image_path
        for image_path, scale in max_scale_per_image.items()
        if scale < min_eye_scale
    ]


# ---------------------------------------------------------------------------
# Data cleanup helpers
# ---------------------------------------------------------------------------

def _remove_image_paths_from_db(image_paths: list[str], dry_run: bool) -> int:
    """DB から指定 image_path のエントリを削除する。削除件数を返す。"""
    if not DB_PATH.exists() or not image_paths:
        return 0
    log.info("DB: %d 画像のエントリを削除予定", len(image_paths))
    if not dry_run:
        _db.delete_samples_by_image_paths(image_paths, DB_PATH)
    return len(image_paths)


def _remove_from_attribution_csv(stems: set[str], dry_run: bool) -> int:
    """open_images_attribution.csv から該当画像の行を削除する。削除件数を返す。"""
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
        with ATTRIBUTION_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(images_dir: Path, dry_run: bool, min_eye_scale: float | None) -> None:
    # DB-stored image_path → absolute file Path
    to_delete: dict[str, Path] = {}

    # --- Face check ---
    if not images_dir.exists():
        log.warning("画像ディレクトリが見つかりません: %s (face check をスキップ)", images_dir)
    else:
        _, no_face = scan_images(images_dir)
        log.info("顔なし: %d 枚", len(no_face))
        if no_face and DB_PATH.exists():
            abs_to_db: dict[Path, str] = {
                (ROOT / s["image_path"]).resolve(): s["image_path"]
                for s in _db.load_all_samples(DB_PATH)
            }
            for p in no_face:
                db_path = abs_to_db.get(p.resolve())
                if db_path:
                    to_delete[db_path] = p.resolve()
                else:
                    to_delete[str(p.resolve())] = p.resolve()

    # --- eye_scale filter ---
    if min_eye_scale is not None:
        small_paths = find_small_eye_images(min_eye_scale)
        log.info("eye_scale < %.1fpx (最大値基準): %d 枚", min_eye_scale, len(small_paths))
        for rel_path in small_paths:
            to_delete.setdefault(rel_path, (ROOT / rel_path).resolve())

    if not to_delete:
        log.info("削除対象の画像はありません。")
        return

    log.info("---")
    log.info("削除対象: 合計 %d 枚", len(to_delete))
    for db_path in sorted(to_delete):
        log.info("  %s", Path(db_path).name)

    stems = {Path(p).stem for p in to_delete}
    _remove_image_paths_from_db(list(to_delete.keys()), dry_run)
    _remove_from_attribution_csv(stems, dry_run)

    if dry_run:
        log.info("[dry-run] %d 枚の画像は削除されません。", len(to_delete))
    else:
        deleted = 0
        for abs_path in to_delete.values():
            if abs_path.exists():
                abs_path.unlink()
                log.info("削除: %s", abs_path.name)
                deleted += 1
        log.info("完了: %d 枚の画像を削除しました。", deleted)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="顔なし画像・eye_scale が小さい画像を削除する"
    )
    parser.add_argument(
        "--images", type=Path, default=IMAGES_DIR,
        help=f"検査する画像ディレクトリ (default: {IMAGES_DIR})",
    )
    parser.add_argument(
        "--min-eye-scale", type=float, default=None, metavar="PX",
        help="全眼の最大 eye_scale がこの値未満の画像を削除対象に追加する（ピクセル単位）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="削除対象を表示するだけで実際には削除しない",
    )
    args = parser.parse_args()
    main(images_dir=args.images, dry_run=args.dry_run, min_eye_scale=args.min_eye_scale)

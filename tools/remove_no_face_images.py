"""
MediaPipe で顔が検出できなかった画像を削除するスクリプト。

以下を一括処理します:
  1. data/images/ 内の全画像に MediaPipe FaceMesh を適用
  2. 顔が 1 人も検出できなかった画像を特定
  3. 該当画像ファイルを削除
  4. samples.json / manifest.csv / open_images_attribution.csv から該当エントリを除去

使い方:
    # 確認のみ（削除しない）
    python scripts/remove_no_face_images.py  --dry-run

    # 実際に削除する
    python scripts/remove_no_face_images.py

    # 別ディレクトリを対象にする
    python scripts/remove_no_face_images.py --images path/to/images/
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

ROOT = Path(__file__).parent.parent
IMAGES_DIR   = ROOT / "data" / "images"
DB_PATH      = ROOT / "data" / "samples.db"
MANIFEST_CSV = ROOT / "data" / "manifest.csv"
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
    mp_face_mesh = mp.solutions.face_mesh   

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.0
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
# Data cleanup helpers
# ---------------------------------------------------------------------------

def _remove_from_db(no_face_paths: set[Path], dry_run: bool) -> int:
    """DB から顔なし画像のエントリを削除する。削除件数を返す。"""
    if not DB_PATH.exists():
        return 0

    # DB に記録されている image_path の中で no_face_paths に含まれるものを抽出
    all_samples = _db.load_all_samples(DB_PATH)
    target_paths: list[str] = []
    for s in all_samples:
        img_abs = (ROOT / s["image_path"]).resolve()
        if img_abs in no_face_paths:
            target_paths.append(s["image_path"])

    removed = len(set(target_paths))
    log.info("DB: %d 画像のエントリを削除予定", removed)

    if not dry_run and target_paths:
        _db.delete_samples_by_image_paths(list(set(target_paths)), DB_PATH)
    return removed


def _remove_from_manifest_csv(no_face_stems: set[str], dry_run: bool) -> int:
    """manifest.csv から顔なし画像の行を削除する。削除件数を返す。"""
    if not MANIFEST_CSV.exists():
        return 0

    with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept = [r for r in rows if Path(r["image_path"]).stem not in no_face_stems]
    removed = len(rows) - len(kept)
    log.info("manifest.csv: %d 行削除 (残り %d)", removed, len(kept))

    if not dry_run and removed > 0:
        with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def _remove_from_attribution_csv(no_face_stems: set[str], dry_run: bool) -> int:
    """open_images_attribution.csv から顔なし画像の行を削除する。削除件数を返す。"""
    if not ATTRIBUTION_CSV.exists():
        return 0

    with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept = [r for r in rows if r.get("image_id", "") not in no_face_stems]
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

def main(images_dir: Path, dry_run: bool) -> None:
    if not images_dir.exists():
        log.error("画像ディレクトリが見つかりません: %s", images_dir)
        return

    with_face, no_face = scan_images(images_dir)

    log.info("---")
    log.info("顔あり: %d 枚 / 顔なし: %d 枚", len(with_face), len(no_face))

    if not no_face:
        log.info("削除対象の画像はありません。")
        return

    log.info("削除対象:")
    for p in no_face:
        log.info("  %s", p.name)

    no_face_abs   = {p.resolve() for p in no_face}
    no_face_stems = {p.stem for p in no_face}

    # データファイルから除去
    _remove_from_db(no_face_abs, dry_run)
    _remove_from_manifest_csv(no_face_stems, dry_run)
    _remove_from_attribution_csv(no_face_stems, dry_run)

    # 画像ファイル削除
    if dry_run:
        log.info("[dry-run] %d 枚の画像は削除されません。", len(no_face))
    else:
        for p in no_face:
            p.unlink()
            log.info("削除: %s", p.name)
        log.info("完了: %d 枚の画像を削除しました。", len(no_face))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="MediaPipe で顔が検出できなかった画像を削除する"
    )
    parser.add_argument(
        "--images", type=Path, default=IMAGES_DIR,
        help=f"検査する画像ディレクトリ (default: {IMAGES_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="削除対象を表示するだけで実際には削除しない",
    )
    args = parser.parse_args()
    main(images_dir=args.images, dry_run=args.dry_run)

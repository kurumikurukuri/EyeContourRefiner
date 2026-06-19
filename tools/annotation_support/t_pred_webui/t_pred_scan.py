"""未アノテーション眼を CNN で推論し、per-eye の min/max t_pred を DB に保存する。

VLM を使わないため vlm_quality_scan より大幅に高速。
eye_scale >= min_eye_scale の未アノテーション眼が対象。

使い方:
    python tools/annotation_support/t_pred_webui/t_pred_scan.py \\
        --model-dir experiments/EXP020_dual_branch/outputs/X0_none_baseline
    python tools/annotation_support/t_pred_webui/t_pred_scan.py \\
        --max-eyes 500 --dry-run
    python tools/annotation_support/t_pred_webui/t_pred_scan.py \\
        --force
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from tools.annotation_support.t_pred_webui.t_pred_db import (
    DEFAULT_DB,
    count_stats,
    get_processed_keys,
    init_db,
    upsert,
)
from tools.annotation_support.t_pred_webui.eye_crop_renderer import make_predictor

logger = logging.getLogger(__name__)

EXTREME_LO = 0.0
EXTREME_HI = 1.2


def _get_target_eyes(
    db_path: Path, min_eye_scale: float
) -> List[Tuple[str, str, int]]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT image_path, part_name, MIN(idx) AS repr_idx
            FROM samples
            WHERE eye_scale >= ?
            GROUP BY image_path, part_name
            HAVING SUM(CASE WHEN t IS NULL AND skip_flag = 0 THEN 1 ELSE 0 END) > 0
            ORDER BY MAX(eye_scale) DESC
            """,
            (min_eye_scale,),
        ).fetchall()
    return [(r["image_path"], r["part_name"], r["repr_idx"]) for r in rows]


def _get_eye_samples(
    db_path: Path, image_path: str, part_name: str, repr_idx: int
) -> List[Dict[str, Any]]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        ref = conn.execute(
            "SELECT px, py, qx, qy FROM samples WHERE idx = ?", (repr_idx,)
        ).fetchone()
        rows = conn.execute(
            """
            SELECT sub_key, point_index, px, py, qx, qy, eye_scale
            FROM samples
            WHERE image_path = ? AND part_name = ? AND skip_flag = 0 AND t IS NULL
            ORDER BY sub_key, point_index
            """,
            (image_path, part_name),
        ).fetchall()
    if ref is None:
        return [dict(r) for r in rows]
    ref_mx = (float(ref["px"]) + float(ref["qx"])) / 2
    ref_my = (float(ref["py"]) + float(ref["qy"])) / 2
    ref_pq = max(
        math.hypot(float(ref["qx"]) - float(ref["px"]),
                   float(ref["qy"]) - float(ref["py"])),
        1.0,
    )
    threshold = ref_pq * 10.0
    result = []
    for r in rows:
        mx = (float(r["px"]) + float(r["qx"])) / 2
        my = (float(r["py"]) + float(r["qy"])) / 2
        if math.hypot(mx - ref_mx, my - ref_my) <= threshold:
            result.append(dict(r))
    return result


def _process_one(
    image_path: str,
    part_name: str,
    repr_idx: int,
    db_path: Path,
    predictor,
) -> Optional[Tuple[float, float, int, int]]:
    """(min_t, max_t, n_extreme, n_points) を返す。失敗時は None。"""
    import cv2

    samples = _get_eye_samples(db_path, image_path, part_name, repr_idx)
    if not samples:
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
    except Exception:
        return None

    pxs = [float(s["px"]) for s in samples] + [float(s["qx"]) for s in samples]
    pys = [float(s["py"]) for s in samples] + [float(s["qy"]) for s in samples]
    span = max(max(pxs) - min(pxs), max(pys) - min(pys), 1.0)
    margin = span * 1.5
    h, w = img_rgb.shape[:2]
    cx1 = max(0, int(min(pxs) - margin))
    cy1 = max(0, int(min(pys) - margin))
    cx2 = min(w, int(max(pxs) + margin))
    cy2 = min(h, int(max(pys) + margin))
    crop = img_rgb[cy1:cy2, cx1:cx2]
    del img_rgb
    gc.collect()

    t_values = []
    for s in samples:
        px = float(s["px"]) - cx1
        py = float(s["py"]) - cy1
        qx = float(s["qx"]) - cx1
        qy = float(s["qy"]) - cy1
        is_upper = 1.0 if str(s.get("sub_key", "upper")) == "upper" else 0.0
        eye_scale = float(s["eye_scale"]) if s.get("eye_scale") is not None else None
        t_raw = predictor.predict_t_raw(crop, px, py, qx, qy, is_upper, eye_scale)
        if t_raw is not None:
            t_values.append(t_raw)

    if not t_values:
        return None

    min_t = min(t_values)
    max_t = max(t_values)
    n_extreme = sum(1 for t in t_values if t < EXTREME_LO or t > EXTREME_HI)
    return min_t, max_t, n_extreme, len(t_values)


def scan(
    db_path: Path,
    hints_db_path: Path,
    model_dir: Path,
    min_eye_scale: float,
    max_eyes: Optional[int],
    force: bool,
    dry_run: bool,
    device: str,
) -> None:
    if not db_path.exists():
        print(f"ERROR: samples.db が見つかりません: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CNN モデルをロード中: {model_dir}")
    predictor = make_predictor(model_dir, device=device)
    print(f"  → {predictor.loaded_count} fold モデルをロードしました")

    if not dry_run:
        init_db(hints_db_path)

    all_eyes = _get_target_eyes(db_path, min_eye_scale)
    print(f"eye_scale >= {min_eye_scale} の未アノテーション眼: {len(all_eyes)}")

    if not force:
        processed = get_processed_keys(hints_db_path)
        targets = [(p, n, i) for p, n, i in all_eyes if (p, n) not in processed]
        print(f"処理済みスキップ: {len(processed)} → 残り {len(targets)}")
    else:
        targets = all_eyes
        print("--force: 全件再スキャン")

    if max_eyes is not None:
        targets = targets[:max_eyes]
        print(f"--max-eyes {max_eyes}: {len(targets)} 眼を処理")

    if not targets:
        print("処理対象がありません。")
        return

    processed_count = error_count = extreme_count = 0
    bar = tqdm(total=len(targets), desc="Scanning", unit="eye") if _HAS_TQDM else None

    interrupted = False
    try:
        for image_path, part_name, repr_idx in targets:
            result = _process_one(image_path, part_name, repr_idx, db_path, predictor)

            if result is None:
                error_count += 1
            else:
                min_t, max_t, n_extreme, n_points = result
                is_extreme = n_extreme > 0

                if dry_run:
                    status = "EXTREME" if is_extreme else "ok"
                    print(f"[{status:<7}] min={min_t:.3f} max={max_t:.3f} "
                          f"n_ext={n_extreme}/{n_points} idx={repr_idx}")
                else:
                    upsert(
                        image_path=image_path,
                        part_name=part_name,
                        repr_idx=repr_idx,
                        min_t_pred=min_t,
                        max_t_pred=max_t,
                        n_extreme=n_extreme,
                        n_points=n_points,
                        scan_model=str(model_dir),
                        db_path=hints_db_path,
                    )

                processed_count += 1
                if is_extreme:
                    extreme_count += 1

            if bar:
                bar.update(1)

    except KeyboardInterrupt:
        interrupted = True

    if bar:
        bar.close()

    print(f"\n--- {'中断' if interrupted else '完了'} ---")
    print(f"処理: {processed_count} / 極端値あり: {extreme_count} / エラー: {error_count}")
    if interrupted:
        print("再実行すると続きから再開します（--force で全件再スキャン）")

    if not dry_run and not interrupted:
        stats = count_stats(hints_db_path)
        print(f"\n[DB 累計] スキャン済み: {stats['total']} / 極端値あり: {stats['extreme']}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN t_pred バッチスキャン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="data/samples.db")
    p.add_argument("--hints-db", default=str(DEFAULT_DB))
    p.add_argument(
        "--model-dir",
        default="experiments/EXP021_loss_aug_sweep/outputs/phase3_X1_t_l1_woflip",
    )
    p.add_argument("--min-eye-scale", type=float, default=60.0)
    p.add_argument("--max-eyes", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    scan(
        db_path=Path(args.db),
        hints_db_path=Path(args.hints_db),
        model_dir=Path(args.model_dir),
        min_eye_scale=args.min_eye_scale,
        max_eyes=args.max_eyes,
        force=args.force,
        dry_run=args.dry_run,
        device=args.device,
    )


if __name__ == "__main__":
    main()

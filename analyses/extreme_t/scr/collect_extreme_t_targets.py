"""
t < 0 または t > 1.2 のアノテーション候補 idx を CSV に出力する。

アプローチ B (geo):
  すでに t 極端値がラベル済みの image_path に属する t=NULL サンプルを SQL で抽出。
  モデル不要・高速。

アプローチ C (model):
  t=NULL サンプルを K-Fold アンサンブルモデルで推論し、予測値が極端なサンプルを抽出。
  regression-to-mean を考慮して閾値を緩め設定（default: pred_t_raw < 0.1 or > 1.1）。

出力 CSV 列: idx, source, image_path, part_name, sub_key, point_index, pred_t_raw, reason

使用方法:
    python analyses/extreme_t/scr/collect_extreme_t_targets.py --mode geo
    python analyses/extreme_t/scr/collect_extreme_t_targets.py --mode geo --input-csv analyses/extreme_t/manifests/unlabeled.csv
    python analyses/extreme_t/scr/collect_extreme_t_targets.py --mode model --model-dir experiments/EXP017_xxx/outputs/S12/
    python analyses/extreme_t/scr/collect_extreme_t_targets.py --mode model --model-dir experiments/EXP017_xxx/outputs/S12/ --input-csv analyses/extreme_t/manifests/unlabeled.csv
    python analyses/extreme_t/scr/collect_extreme_t_targets.py --mode both --model-dir experiments/EXP017_xxx/outputs/S12/
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eyeliner.data.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_patch,
)

logger = logging.getLogger(__name__)

DB_PATH = Path("data/samples.db")
FIELDNAMES = [
    "idx", "source", "image_path", "part_name", "sub_key",
    "point_index", "pred_t_raw", "reason",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Approach B: geometric screening (SQL only)
# ---------------------------------------------------------------------------

def _load_csv_rows(csv_path: Path, prob_threshold: float | None = None) -> list[dict]:
    """CSV ファイルから未ラベルサンプルを読み込む。

    prob_threshold が指定された場合、pred_prob 列が存在する行のうち
    pred_prob > prob_threshold の行を除外する（predictions_unlabeled.csv を直接渡す用途）。
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if prob_threshold is not None:
        before = len(rows)
        rows = [
            r for r in rows
            if "pred_prob" not in r or r["pred_prob"] == "" or float(r["pred_prob"]) <= prob_threshold
        ]
        logger.info("prob_threshold=%.2f フィルタ: %d → %d 行", prob_threshold, before, len(rows))
    return rows


def screen_geo(
    conn: sqlite3.Connection,
    t_low: float,
    t_high: float,
    csv_rows: list[dict] | None = None,
) -> list[dict]:
    """t 極端値ラベル済み image_path に属する t=NULL サンプルを返す。

    csv_rows が指定された場合は DB の未ラベルクエリの代わりに CSV を使用する。
    """
    extreme_rows = conn.execute(
        """
        SELECT image_path, GROUP_CONCAT(ROUND(t, 3), ',') AS extreme_ts
        FROM samples
        WHERE t IS NOT NULL AND (t < ? OR t > ?)
        GROUP BY image_path
        """,
        (t_low, t_high),
    ).fetchall()

    if not extreme_rows:
        logger.info("アプローチ B: 極端 t を持つ image_path が見つかりませんでした")
        return []

    extreme_images: dict[str, str] = {r["image_path"]: r["extreme_ts"] for r in extreme_rows}
    logger.info("アプローチ B: 極端 t を持つ画像 %d 枚", len(extreme_images))

    if csv_rows is not None:
        unannotated = [
            r for r in csv_rows
            if r["image_path"] in extreme_images
            and (not r.get("true_t_raw", "").strip())
            and str(r.get("skip_flag", "0")).strip() == "0"
        ]
        unannotated_sorted = sorted(
            unannotated,
            key=lambda r: (r["image_path"], r["part_name"], r["sub_key"], int(r["point_index"])),
        )
        results = [
            {
                "idx": row["idx"],
                "source": "geo",
                "image_path": row["image_path"],
                "part_name": row["part_name"],
                "sub_key": row["sub_key"],
                "point_index": row["point_index"],
                "pred_t_raw": "",
                "reason": f"same_image_extreme_t=[{extreme_images[row['image_path']]}]",
            }
            for row in unannotated_sorted
        ]
    else:
        placeholders = ",".join("?" * len(extreme_images))
        unannotated = conn.execute(
            f"""
            SELECT idx, image_path, part_name, sub_key, point_index
            FROM samples
            WHERE t IS NULL
              AND skip_flag = 0
              AND image_path IN ({placeholders})
            ORDER BY image_path, part_name, sub_key, point_index
            """,
            list(extreme_images.keys()),
        ).fetchall()

        results = [
            {
                "idx": row["idx"],
                "source": "geo",
                "image_path": row["image_path"],
                "part_name": row["part_name"],
                "sub_key": row["sub_key"],
                "point_index": row["point_index"],
                "pred_t_raw": "",
                "reason": f"same_image_extreme_t=[{extreme_images[row['image_path']]}]",
            }
            for row in unannotated
        ]

    logger.info("アプローチ B: 候補 %d 件", len(results))
    return results


# ---------------------------------------------------------------------------
# Approach C: model-based screening
# ---------------------------------------------------------------------------

def screen_model(
    conn: sqlite3.Connection,
    model_dir: Path,
    screen_low: float,
    screen_high: float,
    patch_t_min: float,
    patch_t_max: float,
    label_t_min: float,
    label_t_max: float,
    patch_perp_scale: float,
    patch_h: int,
    patch_w: int,
    log_interval: int = 500,
    csv_rows: list[dict] | None = None,
) -> list[dict]:
    """t=NULL サンプルをモデル推論し、pred_t_raw が極端なサンプルを返す。

    csv_rows が指定された場合は DB クエリの代わりに CSV を使用する。
    """
    import torch
    from eyeliner.models.ensemble import create_ensemble_from_dir

    ensemble, backbone = create_ensemble_from_dir(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble.eval()
    ensemble.to(device)
    logger.info("モデル読み込み完了: backbone=%s, device=%s", backbone, device)

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
    patch_size = (patch_h, patch_w)
    label_range = label_t_max - label_t_min

    if csv_rows is not None:
        unannotated = [
            r for r in csv_rows
            if not r.get("true_t_raw", "").strip()
            and str(r.get("skip_flag", "0")).strip() == "0"
        ]
        unannotated = sorted(unannotated, key=lambda r: r["image_path"])

        class _Row:
            def __init__(self, d: dict) -> None:
                self._d = d

            def __getitem__(self, key: str):
                return self._d[key]

        unannotated = [_Row(r) for r in unannotated]
    else:
        unannotated = conn.execute(
            """
            SELECT idx, image_path, part_name, sub_key, point_index,
                   px, py, qx, qy
            FROM samples
            WHERE t IS NULL AND skip_flag = 0
            ORDER BY image_path
            """
        ).fetchall()
    logger.info("アプローチ C: 未ラベルサンプル %d 件を推論します", len(unannotated))

    results: list[dict] = []
    cached_path: str | None = None
    cached_image: np.ndarray | None = None
    error_images: set[str] = set()
    error_count = 0

    for i, row in enumerate(unannotated):
        if i % log_interval == 0:
            logger.info("  推論中: %d / %d ...", i, len(unannotated))

        img_path = row["image_path"]

        if img_path in error_images:
            error_count += 1
            continue

        if img_path != cached_path:
            bgr = cv2.imread(img_path)
            if bgr is None:
                error_images.add(img_path)
                error_count += 1
                continue
            cached_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cached_path = img_path

        image_rgb = cached_image

        is_upper = 1.0 if row["sub_key"] == "upper" else 0.0

        try:
            patch = extract_patch(
                cached_image,
                float(row["px"]), float(row["py"]),
                float(row["qx"]), float(row["qy"]),
                t_min=patch_t_min, t_max=patch_t_max,
                perp_scale=patch_perp_scale,
                patch_size=patch_size,
            )
            x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            x = (x - mean_t) / std_t
            is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)

            with torch.no_grad():
                t_norm = float(ensemble(x, is_upper_t).item())

            pred_t_raw = t_norm * label_range + label_t_min

        except Exception as exc:
            logger.debug("推論エラー idx=%d: %s", row["idx"], exc)
            error_count += 1
            continue

        if pred_t_raw < screen_low:
            reason = f"pred_t_raw={pred_t_raw:.4f} < {screen_low}"
        elif pred_t_raw > screen_high:
            reason = f"pred_t_raw={pred_t_raw:.4f} > {screen_high}"
        else:
            continue

        results.append({
            "idx": row["idx"],
            "source": "model",
            "image_path": img_path,
            "part_name": row["part_name"],
            "sub_key": row["sub_key"],
            "point_index": row["point_index"],
            "pred_t_raw": f"{pred_t_raw:.4f}",
            "reason": reason,
        })

    logger.info("アプローチ C: 候補 %d 件 (エラー: %d)", len(results), error_count)
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("出力: %s (%d 行)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="t 極端値（t<0 or t>1.2）のアノテーション候補 idx を CSV に出力する"
    )
    parser.add_argument(
        "--mode", choices=["geo", "model", "both"], default="both",
        help="スクリーニングモード (default: both)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="K-Fold モデルが格納されたディレクトリ（model/both モードで必須）",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"DB パス (default: {DB_PATH})",
    )
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="未ラベルサンプルの CSV ファイル（指定時は DB の t=NULL クエリの代わりに使用）。"
             "predictions_unlabeled.csv を直接渡す場合は --prob-threshold と組み合わせる。",
    )
    parser.add_argument(
        "--prob-threshold", type=float, default=None,
        help="input-csv に pred_prob 列がある場合、この値以下の行のみ使用 (default: 制限なし)。"
             "例: 0.5 → xgboost が skip と予測しにくいサンプルのみ対象",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analyses/extreme_t/outputs"),
        help="CSV 出力ディレクトリ (default: analyses/extreme_t/outputs/)",
    )
    # t 極端値の定義
    parser.add_argument(
        "--t-low", type=float, default=0.0,
        help="極端 t の下限閾値（これ未満を対象）(default: 0.0)",
    )
    parser.add_argument(
        "--t-high", type=float, default=1.2,
        help="極端 t の上限閾値（これ超を対象）(default: 1.2)",
    )
    # モデルスクリーニング閾値（regression-to-mean 考慮で緩め設定）
    parser.add_argument(
        "--screen-low", type=float, default=0.1,
        help="モデルスクリーニング下限 pred_t_raw (default: 0.1)",
    )
    parser.add_argument(
        "--screen-high", type=float, default=1.1,
        help="モデルスクリーニング上限 pred_t_raw (default: 1.1)",
    )
    # パッチ・ラベル正規化パラメータ（EXP016/017 設定に合わせたデフォルト）
    parser.add_argument("--patch-t-min", type=float, default=-2.0)
    parser.add_argument("--patch-t-max", type=float, default=3.0)
    parser.add_argument("--label-t-min", type=float, default=-1.0,
                        help="ラベル正規化下限（EXP016/017 は -1.0）")
    parser.add_argument("--label-t-max", type=float, default=2.0,
                        help="ラベル正規化上限（EXP016/017 は 2.0）")
    parser.add_argument("--patch-perp-scale", type=float, default=2.0)
    parser.add_argument("--patch-h", type=int, default=64)
    parser.add_argument("--patch-w", type=int, default=64)

    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.mode in ("model", "both") and args.model_dir is None:
        parser.error("--mode model/both には --model-dir が必要です")

    conn = _connect(args.db)
    csv_rows: list[dict] | None = None
    if args.input_csv is not None:
        csv_rows = _load_csv_rows(args.input_csv, prob_threshold=args.prob_threshold)
        logger.info("入力 CSV: %s (%d 行)", args.input_csv, len(csv_rows))

    geo_rows: list[dict] = []
    model_rows: list[dict] = []

    if args.mode in ("geo", "both"):
        geo_rows = screen_geo(conn, args.t_low, args.t_high, csv_rows=csv_rows)
        write_csv(geo_rows, args.output_dir / "extreme_t_targets_geo.csv")

    if args.mode in ("model", "both"):
        model_rows = screen_model(
            conn,
            args.model_dir,
            screen_low=args.screen_low,
            screen_high=args.screen_high,
            patch_t_min=args.patch_t_min,
            patch_t_max=args.patch_t_max,
            label_t_min=args.label_t_min,
            label_t_max=args.label_t_max,
            patch_perp_scale=args.patch_perp_scale,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            csv_rows=csv_rows,
        )
        write_csv(model_rows, args.output_dir / "extreme_t_targets_model.csv")

    if args.mode == "both":
        geo_idx = {r["idx"] for r in geo_rows}
        combined = geo_rows + [r for r in model_rows if r["idx"] not in geo_idx]
        combined.sort(key=lambda r: int(r["idx"]))
        write_csv(combined, args.output_dir / "extreme_t_targets_combined.csv")
        logger.info("合計候補（重複除去）: %d 件", len(combined))

    conn.close()


if __name__ == "__main__":
    main()

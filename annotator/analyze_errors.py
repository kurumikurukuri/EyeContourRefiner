"""
アノテーション品質分析スクリプト
webuiからも呼び出している

infer_eyeliner_detection.py の出力 CSV を受け取り、
回帰誤差の大きいサンプルを特定してアノテーションミスの候補を報告します。

使い方:
    # 既存の predictions CSV を分析（誤差の大きい順に CSV 出力）
    python annotator/analyze_errors.py \\
        --predictions data/predictions.csv \\
        --top 50

    # パッチ画像付き HTML レポートも生成
    python annotator/analyze_errors.py \\
        --predictions data/predictions.csv \\
        --top 50 \\
        --report data/error_report.html

    # 閾値以上の誤差のみ対象にする
    python annotator/analyze_errors.py \\
        --predictions data/predictions.csv \\
        --threshold 0.15

出力:
    - <output>            : 誤差の大きい順にソートした CSV（デフォルト: data/error_analysis.csv）
    - <report>            : 上位 N 件のパッチ画像付き HTML レポート（--report 指定時）

infer_eyeliner_detection.py の出力 CSV に含まれる列:
    idx, image_path, part_name, sub_key, point_index,
    px, py, qx, qy, t_true, t_pred, rx_pred, ry_pred
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from eyeliner.data.patch_utils import (
    PATCH_SIZE,
    extract_patch,
    reconstruct_point,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> List[dict]:
    """predictions CSV を読み込み、abs_error を付加して返す。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t_true = float(row["t_true"])
                t_pred = float(row["t_pred"])
            except (KeyError, ValueError):
                continue
            row["t_true"] = t_true
            row["t_pred"] = t_pred
            row["abs_error"] = abs(t_true - t_pred)
            rows.append(row)

    logger.info("%d サンプルを読み込みました: %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_zone_mae(rows: List[dict]) -> None:
    """t_true のゾーン別 MAE を出力する（エッジケース分析用）。"""
    zones = {
        "Low  edge (t < 0.3)":       lambda t: t < 0.3,
        "Middle    (0.3 ≤ t < 0.7)": lambda t: 0.3 <= t < 0.7,
        "High edge (t ≥ 0.7)":       lambda t: t >= 0.7,
    }

    logger.info("-" * 50)
    logger.info("  ゾーン別 MAE（t_true 基準）")
    logger.info("-" * 50)
    for zone_name, condition in zones.items():
        zone_rows = [r for r in rows if condition(r["t_true"])]
        if zone_rows:
            zone_errors = np.array([r["abs_error"] for r in zone_rows])
            logger.info(
                "  %-35s n=%4d  MAE=%.4f",
                zone_name, len(zone_rows), zone_errors.mean(),
            )
        else:
            logger.info("  %-35s n=   0  MAE=N/A", zone_name)

    # Pearson 相関: |t - 0.5| vs abs_error
    t_dist = np.array([abs(r["t_true"] - 0.5) for r in rows])
    errors_arr = np.array([r["abs_error"] for r in rows])
    if len(t_dist) > 1:
        corr = float(np.corrcoef(t_dist, errors_arr)[0, 1])
        logger.info("  Pearson相関（|t-0.5| vs abs_error）: %.4f", corr)
    logger.info("-" * 50)


def print_summary(rows: List[dict], threshold: float) -> None:
    errors = [r["abs_error"] for r in rows]
    if not errors:
        logger.warning("サンプルがありません。")
        return

    errors_arr = np.array(errors)
    n_above = int((errors_arr >= threshold).sum())

    logger.info("=" * 50)
    logger.info("  サンプル数    : %d", len(errors))
    logger.info("  MAE           : %.4f", errors_arr.mean())
    logger.info("  中央値        : %.4f", float(np.median(errors_arr)))
    logger.info("  P90           : %.4f", float(np.percentile(errors_arr, 90)))
    logger.info("  P95           : %.4f", float(np.percentile(errors_arr, 95)))
    logger.info("  最大誤差      : %.4f", errors_arr.max())
    logger.info("  閾値 %.2f 以上 : %d 件 (%.1f%%)",
                threshold, n_above, 100.0 * n_above / len(errors))
    logger.info("=" * 50)
    print_zone_mae(rows)

    # 画像ごとの集計（誤差の大きいものが多い画像）
    image_errors: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        image_errors[r["image_path"]].append(r["abs_error"])

    image_stats = [
        (img, np.mean(errs), np.max(errs), len(errs))
        for img, errs in image_errors.items()
    ]
    image_stats.sort(key=lambda x: x[1], reverse=True)

    logger.info("\n--- 平均誤差の高い画像 Top 10 ---")
    for img, mean_e, max_e, n in image_stats[:10]:
        logger.info("  mean=%.3f  max=%.3f  n=%3d  %s", mean_e, max_e, n, img)


# ---------------------------------------------------------------------------
# CSV 出力
# ---------------------------------------------------------------------------

def write_sorted_csv(rows: List[dict], output_path: Path, threshold: float) -> None:
    """誤差の大きい順にソートして CSV 出力する。"""
    sorted_rows = sorted(rows, key=lambda r: r["abs_error"], reverse=True)
    filtered = [r for r in sorted_rows if r["abs_error"] >= threshold] or sorted_rows

    out_fields = [
        "rank", "abs_error", "t_true", "t_pred",
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(filtered, 1):
            writer.writerow({
                **r,
                "rank":      rank,
                "abs_error": f"{r['abs_error']:.6f}",
                "t_true":    f"{r['t_true']:.6f}",
                "t_pred":    f"{r['t_pred']:.6f}",
            })

    logger.info("誤差 CSV を出力しました (%d 件): %s", len(filtered), output_path)


# ---------------------------------------------------------------------------
# パッチ画像描画
# ---------------------------------------------------------------------------

def _draw_patch_with_markers(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    t_true: float,
    t_pred: float,
    scale: int = 4,
) -> np.ndarray:
    """パッチを抽出し、t_true と t_pred の位置をマーカー表示する。"""
    patch_h, patch_w = PATCH_SIZE
    patch = extract_patch(image_rgb, px, py, qx, qy)
    patch_u8 = (patch * 255).clip(0, 255).astype(np.uint8)

    disp_h = patch_h * scale
    disp_w = patch_w * scale
    disp = cv2.resize(patch_u8, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    def t_to_x(t: float) -> int:
        """t_norm ∈ [0,1] をパッチ上の x 座標に変換する。"""
        return int(round((1.0 - t) * (disp_w - 1)))

    # t_true: 緑の縦線
    x_true = t_to_x(t_true)
    cv2.line(disp, (x_true, 0), (x_true, disp_h - 1), (0, 220, 0), 2)

    # t_pred: 赤の縦線
    x_pred = t_to_x(t_pred)
    cv2.line(disp, (x_pred, 0), (x_pred, disp_h - 1), (220, 0, 0), 2)

    return disp


def _patch_to_base64(patch_rgb: np.ndarray) -> str:
    """RGB uint8 パッチを PNG base64 文字列にエンコードする。"""
    bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML レポート生成
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>アノテーション誤差レポート</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 16px; }}
  h1 {{ font-size: 18px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: middle; }}
  th {{ background: #f0f0f0; text-align: center; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .error-high {{ color: #c00; font-weight: bold; }}
  .error-mid  {{ color: #c80; }}
  img {{ display: block; }}
  .legend {{ margin: 8px 0; font-size: 12px; }}
  .leg-true {{ color: green; font-weight: bold; }}
  .leg-pred {{ color: red;   font-weight: bold; }}
</style>
</head>
<body>
<h1>アノテーション誤差レポート</h1>
<p class="legend">
  パッチ内の縦線:
  <span class="leg-true">緑 = t_true（アノテーション）</span> /
  <span class="leg-pred">赤 = t_pred（モデル予測）</span>
</p>
<table>
<thead>
<tr>
  <th>#</th><th>abs_error</th><th>t_true</th><th>t_pred</th>
  <th>part / sub_key / pt</th><th>image_path</th><th>パッチ</th>
</tr>
</thead>
<tbody>
"""

_HTML_ROW = """\
<tr>
  <td style="text-align:right">{rank}</td>
  <td class="{error_cls}" style="text-align:right">{abs_error:.4f}</td>
  <td style="text-align:right">{t_true:.4f}</td>
  <td style="text-align:right">{t_pred:.4f}</td>
  <td>{part_name} / {sub_key} / {point_index}</td>
  <td style="font-size:11px">{image_path}</td>
  <td><img src="data:image/png;base64,{b64}" width="{w}" height="{h}"></td>
</tr>
"""

_HTML_TAIL = """\
</tbody>
</table>
</body>
</html>
"""


def write_html_report(
    rows: List[dict],
    report_path: Path,
    top: int,
    scale: int = 4,
) -> None:
    """上位 top 件のパッチ画像付き HTML レポートを生成する。"""
    sorted_rows = sorted(rows, key=lambda r: r["abs_error"], reverse=True)
    target = sorted_rows[:top]

    patch_h, patch_w = PATCH_SIZE
    disp_w = patch_w * scale
    disp_h = patch_h * scale

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_HTML_HEAD)

        for rank, r in enumerate(target, 1):
            abs_error = float(r["abs_error"])
            t_true = float(r["t_true"])
            t_pred = float(r["t_pred"])
            image_path = r["image_path"]

            # パッチ描画
            b64 = ""
            bgr = cv2.imread(image_path)
            if bgr is not None:
                image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                patch = _draw_patch_with_markers(
                    image_rgb,
                    float(r["px"]), float(r["py"]),
                    float(r["qx"]), float(r["qy"]),
                    t_true, t_pred,
                    scale=scale,
                )
                b64 = _patch_to_base64(patch)

            # 閾値は t_norm 単位 [0,1]
            error_cls = (
                "error-high" if abs_error >= 0.15
                else "error-mid" if abs_error >= 0.08
                else ""
            )

            f.write(_HTML_ROW.format(
                rank=rank,
                abs_error=abs_error,
                t_true=t_true,
                t_pred=t_pred,
                part_name=html.escape(r.get("part_name", "")),
                sub_key=html.escape(r.get("sub_key", "")),
                point_index=html.escape(str(r.get("point_index", ""))),
                image_path=html.escape(image_path),
                b64=b64,
                w=disp_w,
                h=disp_h,
                error_cls=error_cls,
            ))

            if rank % 10 == 0:
                logger.info("  HTML: %d / %d 件処理済み...", rank, len(target))

        f.write(_HTML_TAIL)

    logger.info("HTML レポートを出力しました (%d 件): %s", len(target), report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="アノテーション誤差分析 — 回帰誤差の大きいサンプルを特定します"
    )
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="infer_eyeliner_detection.py が出力した predictions CSV",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/error_analysis.csv"),
        help="誤差ソート済み CSV の出力先 (default: data/error_analysis.csv)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="パッチ画像付き HTML レポートの出力先 (省略時は生成しない)",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="HTML レポートに含める件数 (default: 50)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="この値以上の abs_error のみ CSV に出力する (default: 0.0 = 全件)",
    )
    args = parser.parse_args()

    rows = load_predictions(args.predictions)
    if not rows:
        logger.error("有効なサンプルが見つかりません。")
        sys.exit(1)

    print_summary(rows, threshold=args.threshold if args.threshold > 0 else 0.1)
    write_sorted_csv(rows, args.output, threshold=args.threshold)

    if args.report:
        write_html_report(rows, args.report, top=args.top)


if __name__ == "__main__":
    main()

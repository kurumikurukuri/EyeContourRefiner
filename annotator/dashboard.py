"""データ分析ダッシュボード API (FastAPI APIRouter)"""
from __future__ import annotations

import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

MANIFEST_PATH    = Path("data/manifest.csv")
PREDICTIONS_PATH = Path("data/predictions.csv")
RUNS_DIR         = Path("runs")

# インメモリキャッシュ: key → {"mtime": float, "data": Any}
_cache: Dict[str, Any] = {}


def _cached(path: Path, compute):
    """ファイルの更新日時をチェックし、変更がなければキャッシュを返す。"""
    key   = str(path.resolve())
    mtime = path.stat().st_mtime if path.exists() else 0.0
    entry = _cache.get(key)
    if entry is not None and entry["mtime"] == mtime:
        return entry["data"]
    data = compute()
    _cache[key] = {"mtime": mtime, "data": data}
    return data


def _load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t_raw   = row.get("t", "").strip()
                t_float: Optional[float] = float(t_raw) if t_raw else None
                px = float(row.get("px", 0))
                py = float(row.get("py", 0))
                qx = float(row.get("qx", 0))
                qy = float(row.get("qy", 0))
                rows.append({
                    **row,
                    "t_float":     t_float,
                    "px": px, "py": py,
                    "qx": qx, "qy": qy,
                    "point_index": int(row.get("point_index", 0)),
                    "pq_dist":     math.hypot(qx - px, qy - py),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _histogram(values: List[float], n_bins: int = 20) -> Dict[str, Any]:
    if not values:
        return {"bins": [], "counts": []}
    arr           = np.array(values, dtype=float)
    counts, edges = np.histogram(arr, bins=n_bins)
    centers       = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return {
        "bins":   [round(float(b), 4) for b in centers],
        "counts": [int(c) for c in counts],
    }


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean":   round(float(arr.mean()),         4),
        "median": round(float(np.median(arr)),      4),
        "std":    round(float(arr.std()),            4),
        "min":    round(float(arr.min()),            4),
        "max":    round(float(arr.max()),            4),
        "count":  len(values),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/overview")
def get_overview() -> Dict[str, Any]:
    rows = _cached(MANIFEST_PATH, _load_manifest)
    if not rows:
        return {"error": "manifest.csv が見つかりません"}

    total          = len(rows)
    annotated      = sum(1 for r in rows if r["t_float"] is not None)
    unique_images  = len(set(r.get("image_path", "") for r in rows))

    by_part:         Dict[str, int] = {}
    by_sub_key:      Dict[str, int] = {}
    by_label_source: Dict[str, int] = {}
    by_point_index:  Dict[str, int] = {str(i): 0 for i in range(7)}

    for r in rows:
        part = r.get("part_name", "unknown")
        sub  = r.get("sub_key",   "unknown")
        src  = r.get("label_source") or (
            "unannotated" if r["t_float"] is None else "manual"
        )
        pt = str(r["point_index"])
        by_part[part]            = by_part.get(part, 0) + 1
        by_sub_key[sub]          = by_sub_key.get(sub, 0) + 1
        by_label_source[src]     = by_label_source.get(src, 0) + 1
        by_point_index[pt]       = by_point_index.get(pt, 0) + 1

    return {
        "total":          total,
        "annotated":      annotated,
        "unannotated":    total - annotated,
        "progress_pct":   round(100.0 * annotated / total, 1) if total > 0 else 0.0,
        "unique_images":  unique_images,
        "by_part_name":   by_part,
        "by_sub_key":     by_sub_key,
        "by_label_source": by_label_source,
        "by_point_index": by_point_index,
    }


@router.get("/t_distribution")
def get_t_distribution() -> Dict[str, Any]:
    rows      = _cached(MANIFEST_PATH, _load_manifest)
    annotated = [r for r in rows if r["t_float"] is not None]
    all_t     = [r["t_float"] for r in annotated]

    by_sub_key:   Dict[str, List[float]] = {}
    by_part_name: Dict[str, List[float]] = {}
    by_sub_pt:    Dict[str, Dict[str, List[float]]] = {}
    for r in annotated:
        by_sub_key.setdefault(r.get("sub_key",   "unknown"), []).append(r["t_float"])
        by_part_name.setdefault(r.get("part_name", "unknown"), []).append(r["t_float"])
        sub = r.get("sub_key", "unknown")
        pt  = str(r["point_index"])
        by_sub_pt.setdefault(sub, {}).setdefault(pt, []).append(r["t_float"])

    return {
        "overall":      {"histogram": _histogram(all_t), "stats": _stats(all_t)},
        "by_sub_key":   {k: {"histogram": _histogram(v), "stats": _stats(v)} for k, v in by_sub_key.items()},
        "by_part_name": {k: {"histogram": _histogram(v), "stats": _stats(v)} for k, v in by_part_name.items()},
        "by_sub_pt": {
            sub: {
                pt: {"histogram": _histogram(v, n_bins=15), "stats": _stats(v)}
                for pt, v in sorted(pts.items())
            }
            for sub, pts in sorted(by_sub_pt.items())
        },
    }


@router.get("/quality")
def get_quality() -> Dict[str, Any]:
    rows      = _cached(MANIFEST_PATH, _load_manifest)
    annotated = [r for r in rows if r["t_float"] is not None]
    pq_dists  = [r["pq_dist"] for r in rows if r["pq_dist"] > 0]

    # PQ 距離外れ値 (IQR法)
    pq_arr              = np.array(pq_dists, dtype=float)
    pq_q1, pq_q3       = float(np.percentile(pq_arr, 25)), float(np.percentile(pq_arr, 75))
    pq_iqr              = pq_q3 - pq_q1
    pq_lo, pq_hi        = pq_q1 - 1.5 * pq_iqr, pq_q3 + 1.5 * pq_iqr
    pq_outliers = sorted(
        [
            {"idx": r.get("idx"), "image_path": r.get("image_path"),
             "pq_dist": round(r["pq_dist"], 2), "part_name": r.get("part_name")}
            for r in rows if r["pq_dist"] < pq_lo or r["pq_dist"] > pq_hi
        ],
        key=lambda x: x["pq_dist"], reverse=True,
    )

    # t 値外れ値 (IQR法)
    t_vals    = np.array([r["t_float"] for r in annotated], dtype=float)
    t_outliers: List[Dict] = []
    iqr_bounds: Dict       = {}
    if len(t_vals) > 0:
        q1, q3    = float(np.percentile(t_vals, 25)), float(np.percentile(t_vals, 75))
        iqr       = q3 - q1
        lo, hi    = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_bounds = {"low": round(lo, 4), "high": round(hi, 4)}
        for r in annotated:
            t = r["t_float"]
            if t < lo or t > hi:
                t_outliers.append({
                    "idx":         r.get("idx"),
                    "image_path":  r.get("image_path"),
                    "part_name":   r.get("part_name"),
                    "sub_key":     r.get("sub_key"),
                    "point_index": r["point_index"],
                    "t":           round(t, 4),
                })
        t_outliers.sort(key=lambda x: abs(x["t"] - 0.5), reverse=True)

    # point_index 別統計 (7パターン)
    pt_groups: Dict[str, List[float]] = {}
    for r in annotated:
        pt_groups.setdefault(str(r["point_index"]), []).append(r["t_float"])
    by_point_index = {pt: _stats(v) for pt, v in sorted(pt_groups.items())}

    # sub_key × point_index 別統計 (14パターン)
    sub_pt_groups: Dict[str, Dict[str, List[float]]] = {}
    for r in annotated:
        sub = r.get("sub_key", "unknown")
        pt  = str(r["point_index"])
        sub_pt_groups.setdefault(sub, {}).setdefault(pt, []).append(r["t_float"])
    by_sub_pt = {
        sub: {pt: _stats(v) for pt, v in sorted(pts.items())}
        for sub, pts in sorted(sub_pt_groups.items())
    }

    return {
        "pq_distance": {
            "histogram": _histogram(pq_dists),
            "stats":     _stats(pq_dists),
            "outliers":  pq_outliers[:20],
        },
        "t_outliers": {
            "count":      len(t_outliers),
            "iqr_bounds": iqr_bounds,
            "samples":    t_outliers[:30],
        },
        "by_point_index": by_point_index,
        "by_sub_pt":      by_sub_pt,
    }


@router.get("/experiments")
def get_experiments() -> Dict[str, Any]:
    if not RUNS_DIR.exists():
        return {"experiments": [], "message": "runs/ ディレクトリが見つかりません"}

    experiments = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not list(run_dir.rglob("events.out.tfevents.*")):
            continue

        exp: Dict[str, Any] = {"name": run_dir.name, "metrics": {}}
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            scalar_tags = ea.Tags().get("scalars", [])

            val_maes: List[float] = []
            all_metrics: Dict[str, Any] = {}
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                if not events:
                    continue
                vals = [e.value for e in events]
                all_metrics[tag] = {
                    "final": round(float(vals[-1]), 5),
                    "best":  round(float(min(vals)),  5),
                }
                if "val_mae" in tag and "fold" in tag:
                    val_maes.append(min(vals))

            exp["metrics"] = all_metrics
            if val_maes:
                exp["best_val_mae"] = round(float(sum(val_maes) / len(val_maes)), 5)
                exp["folds"]        = len(val_maes)

        except Exception as e:
            exp["error"] = str(e)

        experiments.append(exp)

    return {"experiments": experiments}


@router.get("/errors")
def get_errors(top: int = 50, threshold: float = 0.0) -> Dict[str, Any]:
    if not PREDICTIONS_PATH.exists():
        return {
            "available": False,
            "message": (
                "data/predictions.csv が見つかりません。"
                "python scripts/infer_eyeliner_detection.py を実行してください。"
            ),
        }

    rows: List[Dict] = []
    with open(PREDICTIONS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t_true = float(row["t_true"])
                t_pred = float(row["t_pred"])
                rows.append({**row, "t_true": t_true, "t_pred": t_pred,
                              "abs_error": abs(t_true - t_pred)})
            except (KeyError, ValueError):
                continue

    if not rows:
        return {"available": False, "message": "有効なサンプルがありません"}

    errors  = np.array([r["abs_error"] for r in rows], dtype=float)
    summary = {
        "total":  len(rows),
        "mae":    round(float(errors.mean()),                   5),
        "median": round(float(np.median(errors)),               5),
        "p90":    round(float(np.percentile(errors, 90)),       5),
        "p95":    round(float(np.percentile(errors, 95)),       5),
        "max":    round(float(errors.max()),                    5),
    }

    by_part:        Dict[str, List[float]] = {}
    by_sub_key:     Dict[str, List[float]] = {}
    by_point_index: Dict[str, List[float]] = {}
    by_sub_pt_errs: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        by_part.setdefault(r.get("part_name", "?"),           []).append(r["abs_error"])
        by_sub_key.setdefault(r.get("sub_key", "?"),          []).append(r["abs_error"])
        by_point_index.setdefault(str(r.get("point_index", "?")), []).append(r["abs_error"])
        sub = r.get("sub_key", "?")
        pt  = str(r.get("point_index", "?"))
        by_sub_pt_errs.setdefault(sub, {}).setdefault(pt, []).append(r["abs_error"])

    breakdown = {
        "by_part_name":   {k: round(float(np.mean(v)), 5) for k, v in by_part.items()},
        "by_sub_key":     {k: round(float(np.mean(v)), 5) for k, v in by_sub_key.items()},
        "by_point_index": {k: round(float(np.mean(v)), 5) for k, v in sorted(by_point_index.items())},
        "by_sub_pt":      {
            sub: {pt: round(float(np.mean(v)), 5) for pt, v in sorted(pts.items())}
            for sub, pts in sorted(by_sub_pt_errs.items())
        },
    }

    sorted_rows = sorted(rows, key=lambda r: r["abs_error"], reverse=True)
    top_errors  = [
        {
            "rank":        i + 1,
            "idx":         r.get("idx"),
            "abs_error":   round(r["abs_error"], 5),
            "t_true":      round(r["t_true"],    5),
            "t_pred":      round(r["t_pred"],    5),
            "part_name":   r.get("part_name"),
            "sub_key":     r.get("sub_key"),
            "point_index": r.get("point_index"),
            "image_path":  r.get("image_path"),
        }
        for i, r in enumerate(r for r in sorted_rows if r["abs_error"] >= threshold)
        if i < top
    ]

    return {
        "available":       True,
        "summary":         summary,
        "breakdown":       breakdown,
        "error_histogram": _histogram(list(errors), n_bins=20),
        "top_errors":      top_errors,
    }


@router.get("/errors/{idx}/patch")
def get_error_patch(idx: str) -> Response:
    """誤差サンプルのパッチ画像 (base64 PNG) を返す。"""
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="predictions.csv が見つかりません")

    target: Optional[Dict] = None
    with open(PREDICTIONS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("idx")) == str(idx):
                target = row
                break

    if target is None:
        raise HTTPException(status_code=404, detail=f"idx {idx} が見つかりません")

    try:
        import cv2
        from scripts.analyze_errors import _draw_patch_with_markers, _patch_to_base64
        bgr = cv2.imread(target["image_path"])
        if bgr is None:
            raise HTTPException(status_code=404, detail="画像を読み込めません")
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patch = _draw_patch_with_markers(
            image_rgb,
            float(target["px"]),    float(target["py"]),
            float(target["qx"]),    float(target["qy"]),
            float(target["t_true"]), float(target["t_pred"]),
            scale=4,
        )
        b64 = _patch_to_base64(patch)
        return Response(content=b64.encode(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

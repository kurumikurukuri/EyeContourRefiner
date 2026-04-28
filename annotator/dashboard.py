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

from annotator import db as _db  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_PROJECT_ROOT    = Path(__file__).parent.parent.resolve()
DB_PATH          = Path("data/samples.db")
PREDICTIONS_PATH = Path("data/predictions.csv")
EXPERIMENTS_ROOT = _PROJECT_ROOT / "experiments"

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
    """samples.db からダッシュボード用にサンプルを読み込む。"""
    if not DB_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for s in _db.load_all_samples(DB_PATH):
        try:
            t_val      = s.get("t")
            t_float: Optional[float] = float(t_val) if t_val is not None else None
            px = float(s.get("px", 0))
            py = float(s.get("py", 0))
            qx = float(s.get("qx", 0))
            qy = float(s.get("qy", 0))
            rows.append({
                "idx":         s["idx"],
                "image_path":  s["image_path"].replace("\\", "/"),
                "face_index":  s.get("face_index", 0),
                "part_name":   s.get("part_name", ""),
                "sub_key":     s.get("sub_key", ""),
                "point_index": int(s.get("point_index", 0)),
                "px": px, "py": py,
                "qx": qx, "qy": qy,
                "t":           "" if t_val is None else str(t_val),
                "t_float":     t_float,
                "pq_dist":     math.hypot(qx - px, qy - py),
            })
        except (ValueError, KeyError, TypeError):
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
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "q1": 0, "q3": 0, "max": 0, "count": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean":   round(float(arr.mean()),                4),
        "median": round(float(np.median(arr)),             4),
        "std":    round(float(arr.std()),                  4),
        "min":    round(float(arr.min()),                  4),
        "q1":     round(float(np.percentile(arr, 25)),     4),
        "q3":     round(float(np.percentile(arr, 75)),     4),
        "max":    round(float(arr.max()),                  4),
        "count":  len(values),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/overview")
def get_overview() -> Dict[str, Any]:
    rows = _cached(DB_PATH, _load_manifest)
    if not rows:
        return {"error": "samples.db が見つかりません"}

    total          = len(rows)
    annotated      = sum(1 for r in rows if r["t_float"] is not None)
    unique_images  = len(set(r.get("image_path", "") for r in rows))

    by_part:        Dict[str, int] = {}
    by_sub_key:     Dict[str, int] = {}
    by_point_index: Dict[str, int] = {str(i): 0 for i in range(7)}

    for r in rows:
        part = r.get("part_name", "unknown")
        sub  = r.get("sub_key",   "unknown")
        pt = str(r["point_index"])
        by_part[part]      = by_part.get(part, 0) + 1
        by_sub_key[sub]    = by_sub_key.get(sub, 0) + 1
        by_point_index[pt] = by_point_index.get(pt, 0) + 1

    return {
        "total":         total,
        "annotated":     annotated,
        "unannotated":   total - annotated,
        "progress_pct":  round(100.0 * annotated / total, 1) if total > 0 else 0.0,
        "unique_images": unique_images,
        "by_part_name":  by_part,
        "by_sub_key":    by_sub_key,
        "by_point_index": by_point_index,
    }


@router.get("/t_distribution")
def get_t_distribution() -> Dict[str, Any]:
    rows      = _cached(DB_PATH, _load_manifest)
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
    rows      = _cached(DB_PATH, _load_manifest)
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
    """experiments/ 以下の全 experiment.json を読んで比較用データを返す。"""
    if not EXPERIMENTS_ROOT.exists():
        return {"experiments": [], "message": "experiments/ ディレクトリが見つかりません"}

    import json as _json
    experiments = []
    for json_path in sorted(EXPERIMENTS_ROOT.rglob("experiment.json")):
        try:
            data = _json.loads(json_path.read_text(encoding="utf-8"))
            exp_dir = json_path.parent
            rel = json_path.relative_to(EXPERIMENTS_ROOT)
            group = rel.parts[0]
            hp = data.get("hyperparameters", {})
            results = data.get("results", {})
            exp_dir_rel = str(exp_dir.relative_to(_PROJECT_ROOT))
            experiments.append({
                "group": group,
                "exp_name": data.get("exp_name", exp_dir.name),
                "exp_dir": exp_dir_rel,
                "timestamp": data.get("timestamp"),
                "backbone": hp.get("model_arch", "custom"),
                "lr": hp.get("lr"),
                "batch_size": hp.get("batch_size"),
                "weight_decay": hp.get("weight_decay"),
                "use_pq_ratio": hp.get("use_pq_ratio", False),
                "k_folds": hp.get("k_folds"),
                "cv_mae": results.get("cv_mae"),
                "holdout_mae": results.get("holdout_mae"),
                "n_train_samples": data.get("data", {}).get("n_train_samples"),
                "has_predictions": any(
                    (exp_dir / n).exists()
                    for n in ("prediction.csv", "train_predictions.csv", "test_predictions.csv")
                ),
                "has_error_report": (exp_dir / "error_report.html").exists(),
            })
        except Exception as e:
            logger.warning("experiment.json 読み込み失敗: %s: %s", json_path, e)

    return {"experiments": experiments}


@router.get("/errors")
def get_errors(
    top: int = 50,
    threshold: float = 0.0,
    exp_path: Optional[str] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """エラー分析データを返す。

    exp_path を指定した場合（例: "experiments/EXP001.../exp_name"）、
    そのディレクトリの predictions.csv を使用する。
    split: "train", "test", "all" でファイルを選択。
      - test → test_predictions.csv (なければ prediction.csv)
      - train → train_predictions.csv
      - all → train + test を結合
    """
    if split not in ("train", "test", "all"):
        raise HTTPException(status_code=400, detail="split は train/test/all のいずれかです")

    if exp_path:
        exp_dir = (_PROJECT_ROOT / exp_path).resolve()
        if not exp_dir.is_relative_to(EXPERIMENTS_ROOT):
            raise HTTPException(status_code=400, detail="不正なパスです")

        if split == "train":
            predictions_file = exp_dir / "train_predictions.csv"
        elif split == "test":
            predictions_file = exp_dir / "test_predictions.csv"
            if not predictions_file.exists():
                predictions_file = exp_dir / "prediction.csv"
        else:  # all
            predictions_file = None  # 両方結合
    else:
        predictions_file = PREDICTIONS_PATH

    def _read_predictions(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        result: List[Dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    true_t_raw  = float(row["true_t_raw"])
                    true_t_norm = float(row["true_t_norm"])
                    pred_t_raw  = float(row["pred_t_raw"])
                    pred_t_norm = float(row["pred_t_norm"])
                    px = float(row.get("px", 0))
                    py = float(row.get("py", 0))
                    qx = float(row.get("qx", 0))
                    qy = float(row.get("qy", 0))
                    pq_dist = math.hypot(qx - px, qy - py)
                    abs_err = abs(true_t_raw - pred_t_raw)
                    result.append({
                        **row,
                        "true_t_raw":   true_t_raw,
                        "true_t_norm":  true_t_norm,
                        "pred_t_raw":   pred_t_raw,
                        "pred_t_norm":  pred_t_norm,
                        "pq_dist":      pq_dist,
                        "abs_error":    abs_err,
                        "abs_error_px": abs_err * pq_dist,
                    })
                except (KeyError, ValueError):
                    continue
        return result

    if predictions_file is None:
        # split="all" — train + test を結合
        train_path = exp_dir / "train_predictions.csv"
        test_path = exp_dir / "test_predictions.csv"
        if not train_path.exists() and not test_path.exists():
            fallback = exp_dir / "prediction.csv"
            if not fallback.exists():
                return {
                    "available": False,
                    "message": (
                        f"train_predictions.csv / test_predictions.csv / prediction.csv"
                        f" が {exp_dir} に見つかりません。"
                    ),
                }
            rows = _read_predictions(fallback)
        else:
            rows = _read_predictions(train_path) + _read_predictions(test_path)
    else:
        if not predictions_file.exists():
            return {
                "available": False,
                "message": (
                    f"{predictions_file} が見つかりません。"
                    "python scripts/infer_eyeliner_detection.py を実行してください。"
                ),
            }
        rows = _read_predictions(predictions_file)

    if not rows:
        return {"available": False, "message": "有効なサンプルがありません"}

    errors      = np.array([r["abs_error"]    for r in rows], dtype=float)
    errors_px   = np.array([r["abs_error_px"] for r in rows], dtype=float)
    t_true_arr  = np.array([r["true_t_raw"]   for r in rows], dtype=float)
    t_pred_arr  = np.array([r["pred_t_raw"]   for r in rows], dtype=float)
    pq_dist_arr = np.array([r["pq_dist"]      for r in rows], dtype=float)
    residuals    = t_pred_arr - t_true_arr
    residuals_px = residuals * pq_dist_arr

    summary = {
        "total":          len(rows),
        "mae":            round(float(errors.mean()),                5),
        "median":         round(float(np.median(errors)),            5),
        "p90":            round(float(np.percentile(errors, 90)),    5),
        "p95":            round(float(np.percentile(errors, 95)),    5),
        "max":            round(float(errors.max()),                 5),
        "mae_px":         round(float(errors_px.mean()),             3),
        "median_px":      round(float(np.median(errors_px)),         3),
        "p90_px":         round(float(np.percentile(errors_px, 90)), 3),
        "p95_px":         round(float(np.percentile(errors_px, 95)), 3),
        "max_px":         round(float(errors_px.max()),              3),
        "corr_coef":      round(float(np.corrcoef(t_true_arr, t_pred_arr)[0, 1]) if len(rows) > 1 else 0.0, 4),
        "resid_mean":     round(float(residuals.mean()),    5),
        "resid_std":      round(float(residuals.std()),     5),
        "resid_mean_px":  round(float(residuals_px.mean()), 3),
        "resid_std_px":   round(float(residuals_px.std()),  3),
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
        "raw": {
            "by_part_name":   {k: [round(x, 5) for x in v] for k, v in by_part.items()},
            "by_sub_key":     {k: [round(x, 5) for x in v] for k, v in by_sub_key.items()},
            "by_point_index": {k: [round(x, 5) for x in v] for k, v in sorted(by_point_index.items())},
            "by_sub_pt": {
                sub: {pt: [round(x, 5) for x in v] for pt, v in sorted(pts.items())}
                for sub, pts in sorted(by_sub_pt_errs.items())
            },
        },
    }

    def _make_top_row(i: int, r: Dict) -> Dict:
        return {
            "rank":          i + 1,
            "idx":           r.get("idx"),
            "abs_error":     round(r["abs_error"],    5),
            "abs_error_px":  round(r["abs_error_px"], 3),
            "true_t_raw":    round(r["true_t_raw"],   5),
            "pred_t_raw":    round(r["pred_t_raw"],   5),
            "pq_dist":       round(r["pq_dist"],      2),
            "part_name":     r.get("part_name"),
            "sub_key":       r.get("sub_key"),
            "point_index":   r.get("point_index"),
            "image_path":    r.get("image_path"),
        }

    sorted_rows    = sorted(rows, key=lambda r: r["abs_error"],    reverse=True)
    sorted_rows_px = sorted(rows, key=lambda r: r["abs_error_px"], reverse=True)

    top_errors = [
        _make_top_row(i, r)
        for i, r in enumerate(r for r in sorted_rows if r["abs_error"] >= threshold)
        if i < top
    ]
    top_errors_px = [
        _make_top_row(i, r)
        for i, r in enumerate(sorted_rows_px)
        if i < top
    ]

    # 散布図用データ（最大 1000 点サンプリング）
    _MAX_SCATTER = 1000
    rng = np.random.default_rng(seed=42)
    indices = (
        rng.choice(len(rows), size=_MAX_SCATTER, replace=False).tolist()
        if len(rows) > _MAX_SCATTER
        else list(range(len(rows)))
    )
    scatter_upper = [
        {"x": round(rows[i]["true_t_raw"], 5), "y": round(rows[i]["pred_t_raw"], 5)}
        for i in indices if rows[i].get("sub_key") == "upper"
    ]
    scatter_lower = [
        {"x": round(rows[i]["true_t_raw"], 5), "y": round(rows[i]["pred_t_raw"], 5)}
        for i in indices if rows[i].get("sub_key") != "upper"
    ]
    resid_upper = [
        {"x": round(rows[i]["true_t_raw"], 5), "y": round(rows[i]["pred_t_raw"] - rows[i]["true_t_raw"], 5)}
        for i in indices if rows[i].get("sub_key") == "upper"
    ]
    resid_lower = [
        {"x": round(rows[i]["true_t_raw"], 5), "y": round(rows[i]["pred_t_raw"] - rows[i]["true_t_raw"], 5)}
        for i in indices if rows[i].get("sub_key") != "upper"
    ]
    resid_upper_px = [
        {"x": round(rows[i]["true_t_raw"], 5),
         "y": round((rows[i]["pred_t_raw"] - rows[i]["true_t_raw"]) * rows[i]["pq_dist"], 3)}
        for i in indices if rows[i].get("sub_key") == "upper"
    ]
    resid_lower_px = [
        {"x": round(rows[i]["true_t_raw"], 5),
         "y": round((rows[i]["pred_t_raw"] - rows[i]["true_t_raw"]) * rows[i]["pq_dist"], 3)}
        for i in indices if rows[i].get("sub_key") != "upper"
    ]

    return {
        "available":              True,
        "split":                  split,
        "summary":                summary,
        "breakdown":              breakdown,
        "error_histogram":        _histogram(list(errors),       n_bins=20),
        "error_histogram_px":     _histogram(list(errors_px),    n_bins=20),
        "residual_histogram":     _histogram(list(residuals),    n_bins=30),
        "residual_histogram_px":  _histogram(list(residuals_px), n_bins=30),
        "scatter_data": {
            "upper": scatter_upper,
            "lower": scatter_lower,
        },
        "resid_scatter": {
            "upper": resid_upper,
            "lower": resid_lower,
        },
        "resid_scatter_px": {
            "upper": resid_upper_px,
            "lower": resid_lower_px,
        },
        "top_errors":    top_errors,
        "top_errors_px": top_errors_px,
    }


@router.get("/errors/{idx}/patch")
def get_error_patch(idx: str, exp_path: Optional[str] = None, split: str = "test") -> Response:
    """誤差サンプルのパッチ画像 (base64 PNG) を返す。"""
    if exp_path:
        exp_dir = (_PROJECT_ROOT / exp_path).resolve()
        if not exp_dir.is_relative_to(EXPERIMENTS_ROOT):
            raise HTTPException(status_code=400, detail="不正なパスです")

        if split == "train":
            predictions_file = exp_dir / "train_predictions.csv"
        elif split == "all":
            # 両方から検索
            for name in ("test_predictions.csv", "train_predictions.csv", "prediction.csv"):
                candidate = exp_dir / name
                if candidate.exists():
                    predictions_file = candidate
                    break
            else:
                predictions_file = exp_dir / "prediction.csv"
        else:
            predictions_file = exp_dir / "test_predictions.csv"
            if not predictions_file.exists():
                predictions_file = exp_dir / "prediction.csv"
    else:
        predictions_file = PREDICTIONS_PATH

    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="prediction.csv が見つかりません")

    target: Optional[Dict] = None
    with open(predictions_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("idx")) == str(idx):
                target = row
                break

    if target is None:
        raise HTTPException(status_code=404, detail=f"idx {idx} が見つかりません")

    try:
        import cv2
        from tools.analyze_errors import _draw_patch_with_markers, _patch_to_base64
        bgr = cv2.imread(target["image_path"])
        if bgr is None:
            raise HTTPException(status_code=404, detail="画像を読み込めません")
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patch = _draw_patch_with_markers(
            image_rgb,
            float(target["px"]),           float(target["py"]),
            float(target["qx"]),           float(target["qy"]),
            float(target["true_t_raw"]),   float(target["pred_t_raw"]),
            scale=4,
        )
        b64 = _patch_to_base64(patch)
        return Response(content=b64.encode(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

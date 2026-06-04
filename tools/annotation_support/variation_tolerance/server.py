"""
Tolerance calibration server for eyeliner completion criteria.

Measures acceptable t-range in pixels across 30 random upper[3] samples,
computes pixel_range / eye_scale ratios, and aggregates statistics.

Usage:
    python -m uvicorn tolerance.server:app --port 8001
    → http://127.0.0.1:8001/
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# Add project root to sys.path for eyeliner imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from eyeliner.data.patch_utils import (
    PATCH_SIZE,
    T_MAX,
    T_MIN,
    extract_patch,
    reconstruct_point,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path("data/samples.db")
NUM_SAMPLES = 100
DISPLAY_SCALE = 6
FILTER_EYE_SCALE_MIN = 60

_HTML_PATH = Path(__file__).parent / "calibrate.html"
_STATE_PATH = Path(__file__).parent / "calibration_state.json"

# In-memory storage for submitted results (single-user tool)
_last_results: Optional[List[Dict[str, Any]]] = None
_last_summary: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# t conversion helpers
# ---------------------------------------------------------------------------


def _t_norm_to_raw(t_norm: float) -> float:
    return t_norm * (T_MAX - T_MIN) + T_MIN


def _t_raw_to_norm(t_raw: float) -> float:
    return (t_raw - T_MIN) / (T_MAX - T_MIN)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _row_to_sample(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["pq_dist"] = round(math.hypot(d["qx"] - d["px"], d["qy"] - d["py"]), 4)
    t_raw = d["t"]
    d["t_norm"] = round(_t_raw_to_norm(t_raw), 6)
    return d


def _get_samples_by_idxs(idxs: List[int]) -> List[Dict[str, Any]]:
    """Load specific samples by idx, preserving order."""
    with _connect() as conn:
        placeholders = ",".join("?" * len(idxs))
        rows = conn.execute(
            f"""SELECT idx, image_path, part_name, sub_key, point_index,
                       px, py, qx, qy, t, eye_scale
                FROM samples
                WHERE idx IN ({placeholders})
                  AND skip_flag = 0""",
            idxs,
        ).fetchall()

    row_map = {r["idx"]: _row_to_sample(r) for r in rows}
    # Preserve original order, skip any that are no longer valid
    return [row_map[i] for i in idxs if i in row_map]


def _get_random_samples(n: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Select n random samples matching the filter criteria."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT idx, image_path, part_name, sub_key, point_index,
                   px, py, qx, qy, t, eye_scale
            FROM samples
            WHERE eye_scale > ?
              AND sub_key = 'upper'
              AND point_index = 3
              AND t IS NOT NULL
              AND skip_flag = 0
            """,
            (FILTER_EYE_SCALE_MIN,),
        ).fetchall()

    if len(rows) < n:
        raise HTTPException(
            status_code=500,
            detail=f"Not enough matching samples: {len(rows)} < {n}",
        )

    return [_row_to_sample(r) for r in random.sample(rows, n)]


def _get_sample(idx: int) -> Dict[str, Any]:
    """Get a single sample with computed pq_dist and t_norm."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT idx, image_path, face_index, part_name, sub_key, point_index, "
            "px, py, qx, qy, t, eye_scale FROM samples WHERE idx = ?",
            (idx,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Sample {idx} not found")

    d = dict(row)
    d["pq_dist"] = round(math.hypot(d["qx"] - d["px"], d["qy"] - d["py"]), 4)
    t_raw = d["t"]
    d["t_norm"] = round(_t_raw_to_norm(t_raw), 6)
    return d


def _get_eye_group_points(
    image_path: str, face_index: int, part_name: str
) -> List[Dict[str, Any]]:
    """Get all P/Q points for an eye group (7 upper + 7 lower)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT idx, sub_key, point_index, px, py, qx, qy, t "
            "FROM samples "
            "WHERE image_path = ? AND face_index = ? AND part_name = ? "
            "  AND skip_flag = 0 "
            "ORDER BY sub_key, point_index",
            (image_path, face_index, part_name),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _load_image_rgb(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_base64_png(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _build_calibration_patch(
    image_rgb: np.ndarray,
    px: float,
    py: float,
    qx: float,
    qy: float,
    t_norm: float,
    t_lower_norm: Optional[float] = None,
    t_upper_norm: Optional[float] = None,
) -> np.ndarray:
    """
    Build a display patch with P, Q, current-t, and optional bound markers.

    Returns:
        uint8 RGB image (patch_h * scale, patch_w * scale, 3).
    """
    patch_h, patch_w = PATCH_SIZE
    patch = extract_patch(image_rgb, px, py, qx, qy)
    patch_u8 = (patch * 255).clip(0, 255).astype(np.uint8)

    disp_h = patch_h * DISPLAY_SCALE
    disp_w = patch_w * DISPLAY_SCALE
    disp_rgb = cv2.resize(patch_u8, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    # Helper: t_norm → display x coordinate
    # t_norm=0 → t_raw=T_MIN → right edge; t_norm=1 → t_raw=T_MAX → left edge
    def _t_to_x(t_n: float) -> int:
        return int(round((1.0 - t_n) * (disp_w - 1)))

    # P marker (blue) at t=1.0 (t_norm ≈ 0.5714 with T_MIN=-3, T_MAX=4)
    p_t_norm = (1.0 - T_MIN) / (T_MAX - T_MIN)
    p_x = _t_to_x(p_t_norm)
    cv2.line(disp_rgb, (p_x, 0), (p_x, disp_h - 1), (0, 0, 255), 2)

    # Q marker (red) at t=0.0 (t_norm ≈ 0.4286)
    q_t_norm = (0.0 - T_MIN) / (T_MAX - T_MIN)
    q_x = _t_to_x(q_t_norm)
    cv2.line(disp_rgb, (q_x, 0), (q_x, disp_h - 1), (255, 0, 0), 2)

    # Current t marker (green, dashed)
    t_x = _t_to_x(t_norm)
    _draw_dashed_vline(disp_rgb, t_x, (0, 255, 0), dash_len=8, gap_len=4)

    # Lower bound marker (yellow, thick dashed)
    if t_lower_norm is not None:
        lower_x = _t_to_x(t_lower_norm)
        _draw_dashed_vline(disp_rgb, lower_x, (0, 255, 255), dash_len=10, gap_len=4, thickness=4)
        # "L" label at top
        cv2.putText(disp_rgb, "L", (lower_x + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Upper bound marker (magenta, thick dashed)
    if t_upper_norm is not None:
        upper_x = _t_to_x(t_upper_norm)
        _draw_dashed_vline(disp_rgb, upper_x, (255, 0, 255), dash_len=10, gap_len=4, thickness=4)
        cv2.putText(disp_rgb, "U", (upper_x + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    return disp_rgb


def _draw_dashed_vline(
    img: np.ndarray, x: int, color: tuple, dash_len: int = 8, gap_len: int = 4,
    thickness: int = 2,
) -> None:
    """Draw a dashed vertical line on the image."""
    h = img.shape[0]
    y = 0
    while y < h:
        y_end = min(y + dash_len, h)
        cv2.line(img, (x, y), (x, y_end - 1), color, thickness)
        y = y_end + gap_len


def _render_eye_crop(
    image_rgb: np.ndarray,
    eye_points: List[Dict[str, Any]],
    sample: Dict[str, Any],
    t_norm: float,
    t_lower_norm: Optional[float] = None,
    t_upper_norm: Optional[float] = None,
) -> np.ndarray:
    """
    Render an eye-region crop with P, Q, R, L, U markers overlaid.

    The crop bounding box covers all P/Q points in the eye group + R points,
    with 15% padding. The image is scaled so the longer side is ~400px.

    Returns:
        uint8 RGB image.
    """
    px_s, py_s = sample["px"], sample["py"]
    qx_s, qy_s = sample["qx"], sample["qy"]

    # Collect all P/Q points for bounding box
    all_xs = []
    all_ys = []
    for ep in eye_points:
        all_xs.append(float(ep["px"]))
        all_xs.append(float(ep["qx"]))
        all_ys.append(float(ep["py"]))
        all_ys.append(float(ep["qy"]))

    # Also include R points for the current t and marks
    def _add_r(tn: float) -> None:
        tr = _t_norm_to_raw(tn)
        rx, ry = reconstruct_point(px_s, py_s, qx_s, qy_s, tr)
        all_xs.append(rx)
        all_ys.append(ry)

    _add_r(t_norm)
    if t_lower_norm is not None:
        _add_r(t_lower_norm)
    if t_upper_norm is not None:
        _add_r(t_upper_norm)

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    margin = span * 0.15

    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, int(min_x - margin))
    y1 = max(0, int(min_y - margin))
    x2 = min(w_img, int(max_x + margin))
    y2 = min(h_img, int(max_y + margin))

    crop = image_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    # Scale so longer side is ~800px
    crop_h, crop_w = crop.shape[:2]
    target_long = 800
    scale = max(1, target_long // max(crop_h, crop_w))
    if scale > 1:
        crop = cv2.resize(
            crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_LANCZOS4
        )

    def to_crop(x: float, y: float) -> tuple:
        return (int((x - x1) * scale), int((y - y1) * scale))

    # Draw current t position (R) as a small green dot that updates in real-time
    t_raw = _t_norm_to_raw(t_norm)
    rx, ry = reconstruct_point(px_s, py_s, qx_s, qy_s, t_raw)
    r_cur = to_crop(rx, ry)
    cv2.circle(crop, r_cur, 3, (0, 255, 0), -1)  # small green dot

    # Draw L and U marks
    if t_lower_norm is not None:
        t_raw_l = _t_norm_to_raw(t_lower_norm)
        rlx, rly = reconstruct_point(px_s, py_s, qx_s, qy_s, t_raw_l)
        rl_c = to_crop(rlx, rly)
        cv2.circle(crop, rl_c, 4, (0, 255, 255), -1)  # yellow
        cv2.putText(crop, "L", (rl_c[0] + 6, rl_c[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if t_upper_norm is not None:
        t_raw_u = _t_norm_to_raw(t_upper_norm)
        rux, ruy = reconstruct_point(px_s, py_s, qx_s, qy_s, t_raw_u)
        ru_c = to_crop(rux, ruy)
        cv2.circle(crop, ru_c, 4, (255, 0, 255), -1)  # magenta
        cv2.putText(crop, "U", (ru_c[0] + 6, ru_c[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return crop


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CalibrationMark(BaseModel):
    idx: int
    t_lower: float
    t_upper: float


class SubmitRequest(BaseModel):
    marks: List[CalibrationMark]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Eyeliner Tolerance Calibrator")


@app.get("/")
def index():
    """Serve the calibration UI."""
    if not _HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="calibrate.html not found")
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/samples")
def get_samples():
    """Return 30 samples. Restores from saved state if available, otherwise random."""
    with _connect() as conn:
        total = conn.execute(
            """
            SELECT COUNT(*) FROM samples
            WHERE eye_scale > ?
              AND sub_key = 'upper'
              AND point_index = 3
              AND t IS NOT NULL
              AND skip_flag = 0
            """,
            (FILTER_EYE_SCALE_MIN,),
        ).fetchone()[0]

    # Try to restore from saved state
    restored = False
    if _STATE_PATH.exists():
        try:
            state_data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            saved_idxs = state_data.get("sample_idxs", [])
            if len(saved_idxs) == NUM_SAMPLES:
                samples = _get_samples_by_idxs(saved_idxs)
                if len(samples) == NUM_SAMPLES:
                    restored = True
        except (json.JSONDecodeError, KeyError):
            pass

    if not restored:
        samples = _get_random_samples(NUM_SAMPLES)

    return {"samples": samples, "total_available": total, "restored": restored}


@app.get("/api/patch/{idx}")
def get_patch(
    idx: int,
    t: float = Query(..., ge=0.0, le=1.0, description="t_norm value [0, 1]"),
    t_lower: Optional[float] = Query(None, ge=0.0, le=1.0),
    t_upper: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Return a display patch image for the given sample at the specified t."""
    sample = _get_sample(idx)
    try:
        image_rgb = _load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Image not found: {sample['image_path']}"
        )

    disp = _build_calibration_patch(
        image_rgb,
        sample["px"],
        sample["py"],
        sample["qx"],
        sample["qy"],
        t_norm=t,
        t_lower_norm=t_lower if t_lower is not None and not math.isnan(t_lower) else None,
        t_upper_norm=t_upper if t_upper is not None and not math.isnan(t_upper) else None,
    )

    # Compute R position in image coordinates for display
    t_raw = _t_norm_to_raw(t)
    rx, ry = reconstruct_point(sample["px"], sample["py"], sample["qx"], sample["qy"], t_raw)

    return {
        "patch_b64": _to_base64_png(disp),
        "patch_w": disp.shape[1],
        "patch_h": disp.shape[0],
        "t": round(t, 6),
        "t_raw": round(t_raw, 6),
        "rx": round(rx, 2),
        "ry": round(ry, 2),
    }


@app.get("/api/eye_crop/{idx}")
def get_eye_crop(
    idx: int,
    t: float = Query(..., ge=0.0, le=1.0, description="t_norm value [0, 1]"),
    t_lower: Optional[float] = Query(None, ge=0.0, le=1.0),
    t_upper: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Return an eye-region crop image with P, Q, R, L, U markers."""
    sample = _get_sample(idx)
    try:
        image_rgb = _load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Image not found: {sample['image_path']}"
        )

    eye_points = _get_eye_group_points(
        sample["image_path"], sample["face_index"], sample["part_name"]
    )

    t_lower_val = t_lower if t_lower is not None and not math.isnan(t_lower) else None
    t_upper_val = t_upper if t_upper is not None and not math.isnan(t_upper) else None

    crop = _render_eye_crop(
        image_rgb,
        eye_points,
        sample,
        t_norm=t,
        t_lower_norm=t_lower_val,
        t_upper_norm=t_upper_val,
    )

    return {
        "eye_b64": _to_base64_png(crop),
        "eye_w": crop.shape[1],
        "eye_h": crop.shape[0],
    }


@app.post("/api/submit")
def submit(request: SubmitRequest):
    """Compute statistics from calibration marks."""
    global _last_results, _last_summary

    if len(request.marks) == 0:
        raise HTTPException(status_code=400, detail="No marks provided")

    results: List[Dict[str, Any]] = []
    ratios: List[float] = []

    for mark in request.marks:
        sample = _get_sample(mark.idx)

        if sample["eye_scale"] == 0:
            raise HTTPException(status_code=422, detail=f"eye_scale=0 for idx={mark.idx}")

        t_range_norm = abs(mark.t_upper - mark.t_lower)
        t_range_raw = t_range_norm * (T_MAX - T_MIN)
        pixel_range = t_range_raw * sample["pq_dist"]
        ratio = pixel_range / sample["eye_scale"]

        entry = {
            "idx": mark.idx,
            "image_path": sample["image_path"],
            "t_annotated": sample["t"],
            "t_lower": round(mark.t_lower, 6),
            "t_upper": round(mark.t_upper, 6),
            "t_range_norm": round(t_range_norm, 6),
            "pq_dist": sample["pq_dist"],
            "pixel_range": round(pixel_range, 4),
            "eye_scale": sample["eye_scale"],
            "ratio": round(ratio, 6),
        }
        results.append(entry)
        ratios.append(ratio)

    sorted_ratios = sorted(ratios)
    n = len(sorted_ratios)

    def _percentile(p: float) -> float:
        """Linear interpolation percentile."""
        if n == 0:
            return 0.0
        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = k - f
        if f + 1 >= n:
            return sorted_ratios[f]
        return sorted_ratios[f] + c * (sorted_ratios[f + 1] - sorted_ratios[f])

    summary = {
        "count": n,
        "mean_ratio": round(sum(ratios) / n, 6) if n > 0 else 0.0,
        "median_ratio": round(_percentile(50), 6),
        "std_ratio": round(float(np.std(ratios)), 6) if n > 1 else 0.0,
        "p75_ratio": round(_percentile(75), 6),
        "p90_ratio": round(_percentile(90), 6),
        "p95_ratio": round(_percentile(95), 6),
        "min_ratio": round(min(ratios), 6) if ratios else 0.0,
        "max_ratio": round(max(ratios), 6) if ratios else 0.0,
    }

    _last_results = results
    _last_summary = summary

    return {"results": results, "summary": summary}


class SampleMark(BaseModel):
    idx: int
    t_lower: Optional[float] = None
    t_upper: Optional[float] = None


class SaveMarksRequest(BaseModel):
    samples: List[SampleMark]


@app.post("/api/marks/save")
def save_marks(request: SaveMarksRequest):
    """Persist current marks to a JSON file (survives server restart)."""
    marks_map = {}
    for m in request.samples:
        marks_map[str(m.idx)] = {
            "t_lower": m.t_lower,
            "t_upper": m.t_upper,
        }
    data = {
        "sample_idxs": [m.idx for m in request.samples],
        "marks": marks_map,
    }
    _STATE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"status": "ok", "saved": len(marks_map)}


@app.get("/api/marks/load")
def load_marks():
    """Load previously saved marks from JSON file."""
    if not _STATE_PATH.exists():
        return {"marks": None, "sample_idxs": None}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return {
            "marks": data.get("marks", {}),
            "sample_idxs": data.get("sample_idxs", []),
        }
    except (json.JSONDecodeError, KeyError):
        return {"marks": None, "sample_idxs": None}


@app.get("/api/export_csv")
def export_csv():
    """Download calibration results as CSV."""
    global _last_results

    if not _last_results:
        raise HTTPException(status_code=404, detail="No results yet. Submit first.")

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "idx",
            "image_path",
            "t_annotated",
            "t_lower",
            "t_upper",
            "t_range_norm",
            "pq_dist",
            "pixel_range",
            "eye_scale",
            "ratio",
        ],
    )
    writer.writeheader()
    writer.writerows(_last_results)

    csv_bytes = output.getvalue().encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=calibration_results.csv"},
    )

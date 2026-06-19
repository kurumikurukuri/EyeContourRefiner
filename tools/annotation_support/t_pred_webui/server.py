"""t_pred_hints ブラウザ専用スタンドアロン FastAPI アプリ（読み取り専用）。

annotator (8000) / dashboard (8001) とは独立したポート 8002 で動作する。
スキャン実行機能は持たない（CLI の t_pred_scan.py を使う）。

起動方法:
    python tools/annotation_support/t_pred_webui/server.py
    → http://127.0.0.1:8002/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# プロジェクトルートを sys.path に追加（python server.py で直接起動した場合に必要）
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from tools.annotation_support.t_pred_webui import (
    eye_crop_renderer,
    predictor_cache,
    query,
)
from tools.annotation_support.t_pred_webui.t_pred_db import (
    count_filtered,
    count_stats,
    get_hint,
    list_hints,
)

app = FastAPI(title="t_pred hints browser")

_STATIC_DIR = Path(__file__).parent / "static"

# テストから monkeypatch 可能なモジュールレベル変数
HINTS_DB = Path(__file__).parent / "t_pred_hints.db"
SAMPLES_DB = Path("data/samples.db")

_VALID_SORT_BY = {"min_t_pred", "max_t_pred", "n_extreme", "n_points", "scanned_at"}
_ANNOTATOR_BASE_URL = "http://127.0.0.1:8000"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/stats")
def get_stats() -> Dict[str, int]:
    return count_stats(HINTS_DB)


@app.get("/api/hints")
def get_hints(
    min_t_max: Optional[float] = None,
    max_t_min: Optional[float] = None,
    min_n_extreme: Optional[int] = None,
    sort_by: str = "scanned_at",
    order: str = "desc",
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    if sort_by not in _VALID_SORT_BY:
        raise HTTPException(status_code=400, detail=f"不正な sort_by: {sort_by}")
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail=f"不正な order: {order}")

    rows = list_hints(
        HINTS_DB,
        min_t_max=min_t_max,
        max_t_min=max_t_min,
        min_n_extreme=min_n_extreme,
        sort_by=sort_by,
        descending=(order == "desc"),
        limit=limit,
        offset=offset,
    )
    total = count_filtered(
        HINTS_DB,
        min_t_max=min_t_max,
        max_t_min=max_t_min,
        min_n_extreme=min_n_extreme,
    )

    data = [
        {**row, "annotator_url": f"{_ANNOTATOR_BASE_URL}/?idx={row['repr_idx']}"}
        for row in rows
    ]

    return {
        "success": True,
        "data": data,
        "meta": {"total": total, "limit": limit, "offset": offset},
    }


@app.get("/api/hints/crop")
def get_hint_crop(image_path: str, part_name: str) -> Response:
    hint = get_hint(HINTS_DB, image_path, part_name)
    if hint is None:
        raise HTTPException(status_code=404, detail="hint が見つかりません")

    predictor = predictor_cache.get_predictor(hint["scan_model"])
    if predictor is None:
        raise HTTPException(status_code=404, detail="モデルをロードできません")

    eye_samples = query.fetch_eye_samples(
        SAMPLES_DB, image_path, part_name, hint["repr_idx"]
    )
    if not eye_samples:
        raise HTTPException(status_code=404, detail="眼サンプルが見つかりません")

    try:
        crop_rgb, ox, oy = eye_crop_renderer.load_eye_region_rgb(image_path, eye_samples)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="画像ファイルが見つかりません")
    b64 = eye_crop_renderer.render_eye_crop_b64(
        crop_rgb, eye_samples, predictor, image_offset=(ox, oy)
    )
    if b64 is None:
        raise HTTPException(status_code=404, detail="クロップ画像を生成できません")

    return Response(content=b64.encode(), media_type="text/plain")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)

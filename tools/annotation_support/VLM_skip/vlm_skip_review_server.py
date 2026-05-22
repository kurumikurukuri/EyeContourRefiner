"""vlm_skip_hints.db の可視化・レビュー・samples.db マージ補助 WebUI。

起動:
    python tools/annotation_support/VLM_skip/vlm_skip_review_server.py
    python tools/annotation_support/VLM_skip/vlm_skip_review_server.py --port 8001
    python tools/annotation_support/VLM_skip/vlm_skip_review_server.py --hints-db data/vlm_skip_hints.db --db data/samples.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from tools.annotation_support.VLM_skip.vlm_skip_hints_db import DEFAULT_HINTS_DB

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

_hints_db_path: Path = DEFAULT_HINTS_DB
_samples_db_path: Path = Path("data/samples.db")

app = FastAPI(title="VLM Skip Review")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect_hints() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_hints_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _connect_samples() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_samples_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_FILTER_WHERE: Dict[str, str] = {
    "all":        "WHERE merged_at IS NULL",
    "vlm_skip":   "WHERE vlm_skip = 1 AND merged_at IS NULL",
    "unreviewed": "WHERE vlm_skip = 1 AND human_reviewed_at IS NULL AND merged_at IS NULL",
    "human_skip": "WHERE human_skip = 1 AND merged_at IS NULL",
    "human_keep": "WHERE human_skip = 0 AND human_reviewed_at IS NOT NULL AND merged_at IS NULL",
    "merged":     "WHERE merged_at IS NOT NULL",
}

_SORT_ORDER: Dict[str, str] = {
    "confidence_desc": "ORDER BY COALESCE(vlm_confidence, 0) DESC",
    "confidence_asc":  "ORDER BY COALESCE(vlm_confidence, 1) ASC",
    "path_asc":        "ORDER BY image_path ASC",
    "newest":          "ORDER BY COALESCE(vlm_processed_at, '') DESC",
}

_CHUNK = 500


def _chunked(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# API: stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def api_stats() -> Dict[str, int]:
    if not _hints_db_path.exists():
        return {"total": 0, "vlm_skip": 0, "unreviewed": 0, "human_skip": 0, "merged": 0}
    with _connect_hints() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM image_hints WHERE merged_at IS NULL").fetchone()[0]
        vlm_skip   = conn.execute("SELECT COUNT(*) FROM image_hints WHERE vlm_skip = 1 AND merged_at IS NULL").fetchone()[0]
        unreviewed = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE vlm_skip = 1 AND human_reviewed_at IS NULL AND merged_at IS NULL"
        ).fetchone()[0]
        human_skip = conn.execute("SELECT COUNT(*) FROM image_hints WHERE human_skip = 1 AND merged_at IS NULL").fetchone()[0]
        merged     = conn.execute("SELECT COUNT(*) FROM image_hints WHERE merged_at IS NOT NULL").fetchone()[0]
    return {"total": total, "vlm_skip": vlm_skip, "unreviewed": unreviewed, "human_skip": human_skip, "merged": merged}


# ---------------------------------------------------------------------------
# API: image list
# ---------------------------------------------------------------------------

@app.get("/api/images")
def api_images(
    filter: str = Query("unreviewed"),
    sort: str = Query("confidence_desc"),
    page: int = Query(0, ge=0),
    per_page: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    if not _hints_db_path.exists():
        return {"items": [], "total": 0, "page": 0, "per_page": per_page}

    where = _FILTER_WHERE.get(filter, "WHERE merged_at IS NULL")
    order = _SORT_ORDER.get(sort, "ORDER BY image_path ASC")
    offset = page * per_page

    with _connect_hints() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM image_hints {where}").fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM image_hints {where} {order} LIMIT ? OFFSET ?",
            (per_page, offset),
        ).fetchall()

    items = [
        {
            "image_path":       r["image_path"],
            "vlm_skip":         r["vlm_skip"],
            "vlm_reason":       r["vlm_reason"],
            "vlm_confidence":   r["vlm_confidence"],
            "vlm_model":        r["vlm_model"],
            "human_skip":       r["human_skip"],
            "human_note":       r["human_note"],
            "human_reviewed_at": r["human_reviewed_at"],
        }
        for r in rows
    ]
    return {"items": items, "total": total, "page": page, "per_page": per_page}


# ---------------------------------------------------------------------------
# API: thumbnail
# ---------------------------------------------------------------------------

@app.get("/api/image")
def api_image(
    path: str = Query(...),
    size: int = Query(200, ge=32, le=800),
) -> Response:
    abs_path = Path(path).resolve()
    try:
        abs_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    bgr = cv2.imread(str(abs_path))
    if bgr is None:
        raise HTTPException(status_code=422, detail="Cannot decode image")

    h, w = bgr.shape[:2]
    if max(h, w) > size:
        scale = size / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# API: human judgment (batch)
# ---------------------------------------------------------------------------

class HumanJudgmentRequest(BaseModel):
    image_paths: List[str] = Field(..., max_length=500)
    human_skip: int = Field(..., ge=0, le=1)
    human_note: Optional[str] = None


@app.post("/api/human")
def api_human(req: HumanJudgmentRequest) -> Dict[str, int]:
    if not req.image_paths:
        return {"updated": 0}
    now = datetime.now(timezone.utc).isoformat()
    with _connect_hints() as conn:
        conn.executemany(
            """
            INSERT INTO image_hints (image_path, human_skip, human_note, human_reviewed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                human_skip        = excluded.human_skip,
                human_note        = excluded.human_note,
                human_reviewed_at = excluded.human_reviewed_at
            """,
            [(p, req.human_skip, req.human_note, now) for p in req.image_paths],
        )
    return {"updated": len(req.image_paths)}


# ---------------------------------------------------------------------------
# API: merge to samples.db
# ---------------------------------------------------------------------------

class MergeRequest(BaseModel):
    use: str = "vlm_only"
    min_confidence: float = 0.7
    dry_run: bool = False


@app.post("/api/merge")
def api_merge(req: MergeRequest) -> Dict[str, Any]:
    if not _hints_db_path.exists():
        raise HTTPException(status_code=404, detail="hints DB not found")
    if not _samples_db_path.exists():
        raise HTTPException(status_code=404, detail="samples DB not found")

    with _connect_hints() as conn:
        if req.use == "human_only":
            rows = conn.execute(
                "SELECT image_path FROM image_hints WHERE human_skip = 1"
            ).fetchall()
        else:  # vlm_only
            rows = conn.execute(
                "SELECT image_path FROM image_hints "
                "WHERE vlm_skip = 1 AND (vlm_confidence IS NULL OR vlm_confidence >= ?)",
                (req.min_confidence,),
            ).fetchall()

    target_paths = [r[0] for r in rows]

    if req.dry_run:
        return {"target_count": len(target_paths), "updated_rows": 0, "marked_hints": 0, "dry_run": True}

    now = datetime.now(timezone.utc).isoformat()
    total_rows = 0
    marked = 0

    # ATTACH DATABASE で両 DB を単一トランザクションにまとめ、
    # samples.db 更新と merged_at スタンプをアトミックに実行する
    conn = sqlite3.connect(str(_samples_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("ATTACH DATABASE ? AS hints", (str(_hints_db_path),))
    try:
        for chunk in _chunked(target_paths, _CHUNK):
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"UPDATE main.samples SET skip_flag = 1 WHERE image_path IN ({placeholders})",
                chunk,
            )
            total_rows += cur.rowcount
            cur = conn.execute(
                f"UPDATE hints.image_hints SET merged_at = ? WHERE image_path IN ({placeholders})",
                [now, *chunk],
            )
            marked += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("DETACH DATABASE hints")
        conn.close()

    return {
        "target_count": len(target_paths),
        "updated_rows": total_rows,
        "marked_hints": marked,
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM Skip Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#e0e0e0;display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* Header */
.header{padding:10px 16px;background:#242424;border-bottom:1px solid #333;display:flex;align-items:center;gap:10px;flex-shrink:0}
.header h1{font-size:15px;font-weight:600;margin-right:4px}
.chip{padding:3px 10px;border-radius:12px;font-size:12px;background:#333;white-space:nowrap}
.chip.c-vlm{background:#5a1a1a;color:#f88}
.chip.c-unrev{background:#3d3010;color:#fda}
.chip.c-hskip{background:#1a2a4a;color:#aaf}
.chip.c-merged{background:#1a3a1a;color:#8d8}

/* Toolbar */
.toolbar{padding:7px 16px;background:#1e1e1e;border-bottom:1px solid #333;display:flex;align-items:center;gap:6px;flex-wrap:wrap;flex-shrink:0}
.filter-tabs{display:flex;gap:3px}
.ftab{padding:4px 11px;border-radius:4px;border:1px solid #444;background:transparent;color:#aaa;cursor:pointer;font-size:12px}
.ftab:hover{border-color:#666;color:#ddd}
.ftab.active{background:#0078d4;border-color:#0078d4;color:#fff}
select{padding:4px 8px;background:#2a2a2a;border:1px solid #555;color:#e0e0e0;border-radius:4px;font-size:12px}
.spacer{flex:1}
.btn{padding:5px 13px;border-radius:4px;border:none;cursor:pointer;font-size:12px;font-weight:500}
.btn:disabled{opacity:.4;cursor:default}
.btn-skip{background:#b03020;color:#fff}
.btn-skip:not(:disabled):hover{background:#d04030}
.btn-keep{background:#207040;color:#fff}
.btn-keep:not(:disabled):hover{background:#30a060}
.btn-merge{background:#0060b0;color:#fff}
.btn-merge:hover{background:#0078d4}
.btn-neutral{background:#333;color:#bbb;border:1px solid #555}
.btn-neutral:hover{background:#444}
.sel-count{font-size:12px;color:#888;min-width:50px}

/* Scroll area */
.scroll-area{flex:1;overflow-y:auto;padding:10px 16px}

/* Grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.empty{padding:60px;color:#555;text-align:center;font-size:14px}

/* Card */
.card{position:relative;border-radius:6px;overflow:hidden;background:#2a2a2a;border:2px solid transparent;cursor:pointer;transition:border-color .1s}
.card:hover{border-color:#555}
.card.selected{border-color:#0078d4!important}
.card.hs-yes{border-color:#c0392b}
.card.hs-no{border-color:#27ae60}
.card img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#111}
.card-bar{padding:4px 6px;font-size:11px;background:#1a1a1a;min-height:22px;display:flex;gap:3px;flex-wrap:wrap;align-items:center}
.badge{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap}
.b-skip-hi{background:#8b0000;color:#faa}
.b-skip-lo{background:#7a4000;color:#fda}
.b-keep{background:#1a4a1a;color:#8f8}
.b-none{background:#333;color:#777}
.b-hskip{background:#6a0000;color:#fbb}
.b-hkeep{background:#004a20;color:#8fb}
.card-check{position:absolute;top:5px;left:5px;width:18px;height:18px;border-radius:3px;background:rgba(0,0,0,.65);border:2px solid rgba(255,255,255,.4);display:flex;align-items:center;justify-content:center}
.card.selected .card-check{background:#0078d4;border-color:#0078d4}
.card-check::after{content:'';display:none;width:5px;height:9px;border:2px solid #fff;border-top:none;border-left:none;transform:rotate(45deg) translate(-1px,-1px)}
.card.selected .card-check::after{display:block}

/* Pagination */
.pagination{padding:8px 16px;border-top:1px solid #333;display:flex;align-items:center;gap:8px;justify-content:center;flex-shrink:0;background:#1a1a1a}
.pgbtn{padding:4px 12px;border-radius:4px;border:1px solid #444;background:#2a2a2a;color:#ccc;cursor:pointer;font-size:12px}
.pgbtn:disabled{opacity:.4;cursor:default}
.pginfo{font-size:12px;color:#777;min-width:120px;text-align:center}

/* Toast */
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#2a2a2a;border:1px solid #444;color:#e0e0e0;padding:9px 20px;border-radius:6px;font-size:13px;opacity:0;transition:opacity .25s;pointer-events:none;z-index:200;white-space:nowrap}
.toast.show{opacity:1}

/* Modal */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);display:none;align-items:center;justify-content:center;z-index:100}
.overlay.open{display:flex}
.modal{background:#252525;border:1px solid #3a3a3a;border-radius:8px;padding:22px;min-width:340px;max-width:460px;width:90%}
.modal h3{font-size:15px;margin-bottom:10px}
.modal p{font-size:12px;color:#999;margin-bottom:10px;line-height:1.5}
.modal label{font-size:12px;color:#aaa;display:block;margin-bottom:3px;margin-top:8px}
.modal select{width:100%}
.modal-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
#merge-result{margin-top:10px;font-size:12px;color:#7df;display:none;line-height:1.5}
</style>
</head>
<body>

<div class="header">
  <h1>VLM Skip Review</h1>
  <span class="chip" id="s-total">total: —</span>
  <span class="chip c-vlm" id="s-vlm">VLM skip: —</span>
  <span class="chip c-unrev" id="s-unrev">未レビュー: —</span>
  <span class="chip c-hskip" id="s-hskip">human skip: —</span>
  <span class="chip c-merged" id="s-merged">マージ済み: —</span>
</div>

<div class="toolbar">
  <div class="filter-tabs" id="filter-tabs">
    <button class="ftab" data-f="all">全件</button>
    <button class="ftab" data-f="vlm_skip">VLM Skip</button>
    <button class="ftab active" data-f="unreviewed">未レビュー</button>
    <button class="ftab" data-f="human_skip">Human Skip</button>
    <button class="ftab" data-f="human_keep">Human Keep</button>
    <button class="ftab" data-f="merged">マージ済み</button>
  </div>
  <select id="sort-sel">
    <option value="confidence_desc">confidence ↓</option>
    <option value="confidence_asc">confidence ↑</option>
    <option value="path_asc">ファイル名順</option>
    <option value="newest">新着順</option>
  </select>
  <div class="spacer"></div>
  <button class="btn btn-neutral" id="btn-selall">全選択</button>
  <span class="sel-count" id="sel-count">0件選択</span>
  <button class="btn btn-skip" id="btn-skip" disabled>✕ SKIP</button>
  <button class="btn btn-keep" id="btn-keep" disabled>✓ KEEP</button>
  <button class="btn btn-merge" id="btn-merge">→ samples.db マージ</button>
</div>

<div class="scroll-area">
  <div class="grid" id="grid"></div>
</div>

<div class="pagination" id="pagination"></div>
<div class="toast" id="toast"></div>

<!-- Merge modal -->
<div class="overlay" id="merge-overlay">
  <div class="modal">
    <h3>samples.db にマージ</h3>
    <p>vlm_skip_hints.db の判定結果を samples.db の skip_flag=1 に反映します。<br>アノテーション済み行も更新されます（注意）。</p>
    <label>適用ロジック</label>
    <select id="m-use">
      <option value="human_only">human_only（人間確認済みのみ）</option>
      <option value="vlm_only">vlm_only（VLM 判定のみ）</option>
    </select>
    <label>VLM 信頼度下限</label>
    <select id="m-conf">
      <option value="0.5">0.5</option>
      <option value="0.7" selected>0.7</option>
      <option value="0.8">0.8</option>
      <option value="0.9">0.9</option>
    </select>
    <div id="merge-result"></div>
    <div class="modal-actions">
      <button class="btn btn-neutral" id="m-dryrun">dry-run</button>
      <button class="btn btn-neutral" id="m-cancel">キャンセル</button>
      <button class="btn btn-merge" id="m-exec">実行</button>
    </div>
  </div>
</div>

<script>
'use strict';
let state = { filter: 'unreviewed', sort: 'confidence_desc', page: 0, perPage: 50, total: 0 };
let selected = new Set();

// ---- Stats ----
async function loadStats() {
  const d = await fetch('/api/stats').then(r => r.json());
  document.getElementById('s-total').textContent = 'total: ' + d.total;
  document.getElementById('s-vlm').textContent = 'VLM skip: ' + d.vlm_skip;
  document.getElementById('s-unrev').textContent = '未レビュー: ' + d.unreviewed;
  document.getElementById('s-hskip').textContent = 'human skip: ' + d.human_skip;
  document.getElementById('s-merged').textContent = 'マージ済み: ' + (d.merged || 0);
}

// ---- Images ----
async function loadImages() {
  const { filter, sort, page, perPage } = state;
  const url = `/api/images?filter=${filter}&sort=${sort}&page=${page}&per_page=${perPage}`;
  const d = await fetch(url).then(r => r.json());
  state.total = d.total;
  renderGrid(d.items);
  renderPagination();
  updateSelCount();
}

function vlmBadge(item) {
  if (item.vlm_skip === null || item.vlm_skip === undefined)
    return '<span class="badge b-none">未処理</span>';
  if (item.vlm_skip === 0)
    return '<span class="badge b-keep">keep</span>';
  const c = item.vlm_confidence;
  const cls = (c !== null && c >= 0.8) ? 'b-skip-hi' : 'b-skip-lo';
  const cs = c !== null ? ' ' + Math.round(c * 100) + '%' : '';
  return `<span class="badge ${cls}">SKIP${cs}</span>`;
}

function humanBadge(item) {
  if (item.human_skip === 1)  return '<span class="badge b-hskip">HUMANSKIP</span>';
  if (item.human_skip === 0 && item.human_reviewed_at) return '<span class="badge b-hkeep">HUMANKEEP</span>';
  return '';
}

function renderGrid(items) {
  const grid = document.getElementById('grid');
  if (!items.length) {
    grid.innerHTML = '<div class="empty">表示対象がありません</div>';
    return;
  }
  grid.innerHTML = items.map(item => {
    const enc = encodeURIComponent(item.image_path);
    const sel = selected.has(item.image_path) ? 'selected' : '';
    const border = item.human_skip === 1 ? 'hs-yes' : (item.human_skip === 0 && item.human_reviewed_at ? 'hs-no' : '');
    const reason = (item.vlm_reason || '').substring(0, 80);
    const fname = item.image_path.replace(/.*[\\/]/, '');
    return `<div class="card ${sel} ${border}" data-path="${enc}" title="${fname}\\n${reason}">
      <div class="card-check"></div>
      <img src="/api/image?path=${enc}&size=200" loading="lazy" alt="${fname}">
      <div class="card-bar">${vlmBadge(item)}${humanBadge(item)}</div>
    </div>`;
  }).join('');

  grid.querySelectorAll('.card').forEach(c =>
    c.addEventListener('click', () => toggleCard(c))
  );
}

function toggleCard(card) {
  const path = decodeURIComponent(card.dataset.path);
  if (selected.has(path)) { selected.delete(path); card.classList.remove('selected'); }
  else                     { selected.add(path);    card.classList.add('selected'); }
  updateSelCount();
}

function updateSelCount() {
  const n = selected.size;
  document.getElementById('sel-count').textContent = n + '件選択';
  document.getElementById('btn-skip').disabled = n === 0;
  document.getElementById('btn-keep').disabled = n === 0;
}

// ---- Pagination ----
function renderPagination() {
  const pages = Math.max(1, Math.ceil(state.total / state.perPage));
  const pg = document.getElementById('pagination');
  pg.innerHTML = `
    <button class="pgbtn" id="pg-prev" ${state.page === 0 ? 'disabled' : ''}>◀ 前</button>
    <span class="pginfo">${state.page + 1} / ${pages} ページ（${state.total} 件）</span>
    <button class="pgbtn" id="pg-next" ${state.page >= pages - 1 ? 'disabled' : ''}>次 ▶</button>`;
  pg.querySelector('#pg-prev').addEventListener('click', () => { state.page--; loadImages(); });
  pg.querySelector('#pg-next').addEventListener('click', () => { state.page++; loadImages(); });
}

// ---- Filter / Sort ----
document.getElementById('filter-tabs').addEventListener('click', e => {
  const tab = e.target.closest('.ftab');
  if (!tab) return;
  document.querySelectorAll('.ftab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  state.filter = tab.dataset.f;
  state.page = 0;
  selected.clear();
  loadImages();
});

document.getElementById('sort-sel').addEventListener('change', e => {
  state.sort = e.target.value;
  state.page = 0;
  loadImages();
});

// ---- Select All (current page) ----
document.getElementById('btn-selall').addEventListener('click', () => {
  const cards = document.querySelectorAll('.card');
  const allSel = [...cards].every(c => c.classList.contains('selected'));
  cards.forEach(c => {
    const path = decodeURIComponent(c.dataset.path);
    if (allSel) { selected.delete(path); c.classList.remove('selected'); }
    else        { selected.add(path);    c.classList.add('selected'); }
  });
  updateSelCount();
});

// ---- SKIP / KEEP ----
async function applyHuman(skipVal) {
  const paths = [...selected];
  if (!paths.length) return;
  const d = await fetch('/api/human', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ image_paths: paths, human_skip: skipVal }),
  }).then(r => r.json());
  showToast(`${d.updated} 件を ${skipVal ? 'SKIP' : 'KEEP'} に設定しました`);
  selected.clear();
  await loadStats();
  await loadImages();
}

document.getElementById('btn-skip').addEventListener('click', () => applyHuman(1));
document.getElementById('btn-keep').addEventListener('click', () => applyHuman(0));

// ---- Merge modal ----
document.getElementById('btn-merge').addEventListener('click', () => {
  document.getElementById('merge-result').style.display = 'none';
  document.getElementById('merge-overlay').classList.add('open');
});
document.getElementById('m-cancel').addEventListener('click', () =>
  document.getElementById('merge-overlay').classList.remove('open')
);
document.getElementById('merge-overlay').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.classList.remove('open');
});

async function doMerge(dryRun) {
  const use  = document.getElementById('m-use').value;
  const conf = parseFloat(document.getElementById('m-conf').value);
  const d = await fetch('/api/merge', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ use, min_confidence: conf, dry_run: dryRun }),
  }).then(r => r.json());
  const el = document.getElementById('merge-result');
  el.style.display = 'block';
  if (dryRun) {
    el.textContent = `[dry-run] 対象: ${d.target_count} 件`;
  } else {
    el.textContent = `完了: ${d.target_count} 件 → ${d.updated_rows} サンプル行を更新、${d.marked_hints} 件をマージ済みに変更`;
    loadStats();
    loadImages();
  }
}
document.getElementById('m-dryrun').addEventListener('click', () => doMerge(true));
document.getElementById('m-exec').addEventListener('click', () => doMerge(false));

// ---- Toast ----
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.remove('show'), 2800);
}

// ---- Init ----
loadStats();
loadImages();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vlm_skip_hints.db レビュー WebUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hints-db", default=str(DEFAULT_HINTS_DB), help="vlm_skip_hints.db パス")
    parser.add_argument("--db", default="data/samples.db", help="samples.db パス")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = _parse_args()
    global _hints_db_path, _samples_db_path
    _hints_db_path = Path(args.hints_db)
    _samples_db_path = Path(args.db)
    print(f"hints DB : {_hints_db_path}")
    print(f"samples DB: {_samples_db_path}")
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

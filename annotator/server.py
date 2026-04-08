"""
アイラインアノテーションサーバー (FastAPI)。

起動方法:
    python annotator/server.py
    # または
    uvicorn annotator.server:app --reload

ブラウザで http://localhost:8000 を開いてアノテーションを行う。
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルートを sys.path に追加（python annotator/server.py で起動した場合に必要）
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
    T_MIN,
    T_MAX,
    build_display_patch,
    extract_patch,
    overlay_landmarks,
    reconstruct_point,
)

logger = logging.getLogger(__name__)

SAMPLES_PATH = Path("data/samples.json")
DISPLAY_SCALE = 6  # ROIパッチの表示倍率
STATIC_DIR = Path(__file__).parent / "static"
MODEL_PATH = Path("models/eyeliner_detection.pt")
KFOLD_MODEL_DIR = Path("models_kfold")

# パストラバーサル防止: 許可するモデルディレクトリ（プロジェクトルートからの絶対パス）
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_ALLOWED_MODEL_DIRS = [
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "models_kfold",
    _PROJECT_ROOT / "experiments",
]

# アノテーション書き込みの競合状態防止用ロック
_samples_lock = threading.Lock()

# モデル状態（_infer_models 等）の競合状態防止用ロック
_models_lock = threading.Lock()

# サムネイルキャッシュの競合状態防止用ロック
_thumbnail_cache_lock = threading.Lock()

# サムネイルキャッシュ上限（無制限メモリ使用防止）
_MAX_THUMBNAIL_CACHE = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動時にモデルを読み込む。"""
    _try_load_infer_model()
    yield


app = FastAPI(title="Eyeliner Annotator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from annotator.dashboard import router as dashboard_router  # noqa: E402
app.include_router(dashboard_router)


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))



# ---------------------------------------------------------------------------
# Inference model (optional — loaded if K-Fold or single model exists)
# ---------------------------------------------------------------------------

_infer_models: List = []  # K-Fold アンサンブル用（複数モデルを保持）
_infer_device = None
_active_model_paths: List[Path] = []

# ImageNet 正規化テンソル（デバイスに配置後に設定）
_imagenet_mean_t = None
_imagenet_std_t = None

# EXP固有predictor（create_predictor()が提供する callable）
_active_predictor = None
_active_infer_mode = "default"  # "default" | "exp_native"


def _find_model_paths() -> List[Path]:
    """利用可能なモデルファイルを優先順位順に返す。"""
    # 1. K-Fold アンサンブル（models_kfold/eyeliner_detection_fold*.pt）
    kfold_paths = sorted(KFOLD_MODEL_DIR.glob("eyeliner_detection_fold*.pt"))
    if kfold_paths:
        return kfold_paths
    # 2. models/ 内の fold モデル
    fold_paths = sorted(Path("models").glob("eyeliner_detection_fold*.pt"))
    if fold_paths:
        return fold_paths
    # 3. 単体モデル（後方互換）
    if MODEL_PATH.exists():
        return [MODEL_PATH]
    return []


def _discover_all_model_paths() -> List[Path]:
    """検索ディレクトリ内の全 .pt ファイルを返す。"""
    paths: List[Path] = []
    for d in [_PROJECT_ROOT / "models", _PROJECT_ROOT / "models_kfold"]:
        if d.exists():
            paths.extend(sorted(d.glob("*.pt")))
    exp_dir = _PROJECT_ROOT / "experiments"
    if exp_dir.exists():
        paths.extend(sorted(exp_dir.rglob("*.pt")))
    return paths


def _find_infer_script(directory: Path) -> Optional[Path]:
    """ディレクトリの祖先をたどって scr/infer_exp*.py を探す（最大5階層）。"""
    candidate = directory
    for _ in range(5):
        if candidate == _PROJECT_ROOT or candidate.parent == candidate:
            break
        scr_dir = candidate / "scr"
        if scr_dir.exists():
            scripts = sorted(scr_dir.glob("infer_exp*.py"))
            if scripts:
                return scripts[0]
        candidate = candidate.parent
    return None


def _load_exp_predictor(infer_script: Path, models_dir: Path):
    """EXPの infer_exp*.py から create_predictor を動的インポートして呼び出す。"""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_exp_infer_module", str(infer_script))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "create_predictor"):
            logger.info("create_predictor が見つかりません: %s", infer_script)
            return None
        predictor = module.create_predictor(models_dir)
        logger.info("EXP固有predictor を読み込みました: %s", infer_script.name)
        return predictor
    except Exception as exc:
        logger.warning("EXP固有predictor の読み込みに失敗しました: %s", exc)
        return None


def _reload_models(model_paths: List[Path], predictor=None, infer_mode: str = "default") -> None:
    """モデルをロードしてグローバル状態を一括で更新する。

    predictor と infer_mode も同一ロック内で設定することで
    _active_predictor / _active_infer_mode の TOCTOU を防止する。
    """
    global _infer_models, _infer_device, _active_model_paths, _imagenet_mean_t, _imagenet_std_t, _active_predictor, _active_infer_mode
    if not model_paths:
        with _models_lock:
            _infer_models = []
            _active_model_paths = []
            _active_predictor = predictor
            _active_infer_mode = infer_mode
        logger.info("モデルをクリアしました。推論補助は無効")
        return
    try:
        import torch
        from eyeliner.models.ensemble import load_model as _load_model_from_ckpt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = []
        successfully_loaded_paths = []
        for path in model_paths:
            try:
                model = _load_model_from_ckpt(path)
                model.eval()
                model.to(device)
                loaded.append(model)
                successfully_loaded_paths.append(path)
                logger.info("推論モデルを読み込みました (%s): %s", device, path)
            except Exception as exc:
                logger.warning("モデルの読み込みをスキップしました %s: %s", path, exc)
        if not loaded:
            logger.warning("有効な推論モデルが見つかりませんでした")
            with _models_lock:
                _infer_models = []
                _active_model_paths = []
                _active_predictor = None
                _active_infer_mode = "default"
            return
        # モデルロード完了後に一括でグローバル状態を更新（部分更新による不整合を防止）
        mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
        std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
        with _models_lock:
            _infer_models = loaded
            _infer_device = device
            _active_model_paths = successfully_loaded_paths
            _imagenet_mean_t = mean_t
            _imagenet_std_t  = std_t
            _active_predictor = predictor
            _active_infer_mode = infer_mode
        logger.info("%d 個のモデルをアンサンブルとして使用します", len(loaded))
    except Exception as exc:
        logger.warning("推論モデルの読み込みに失敗しました: %s", exc)


def _try_load_infer_model() -> None:
    _reload_models(_find_model_paths())


def _predict_t(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    is_upper: float,
) -> Optional[float]:
    # グローバル状態のスナップショットをロック下で取得してからロックを解放し推論を実行
    with _models_lock:
        predictor = _active_predictor
        if predictor is None and not _infer_models:
            return None
        models = list(_infer_models) if predictor is None else []
        device = _infer_device
        mean_t = _imagenet_mean_t
        std_t  = _imagenet_std_t

    # EXP固有predictorを優先使用
    if predictor is not None:
        try:
            return predictor(image_rgb, px, py, qx, qy, is_upper)
        except Exception as exc:
            logger.warning("EXP固有predictor の推論に失敗しました: %s", exc)
            return None

    try:
        import torch
        patch = extract_patch(image_rgb, px, py, qx, qy)
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        x = (x - mean_t) / std_t
        is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = [model(x, is_upper_t).item() for model in models]
        return float(sum(preds) / len(preds))
    except TypeError as exc:
        logger.error("モデル呼び出しの引数エラー（バグ）: %s", exc)
        raise
    except Exception as exc:
        logger.warning("推論に失敗しました: %s", exc)
        return None


# サムネイルキャッシュ（image_path → base64文字列）— 上限付き OrderedDict で LRU 管理
_thumbnail_cache: OrderedDict[str, str] = OrderedDict()


# ---------------------------------------------------------------------------
# Samples I/O
# ---------------------------------------------------------------------------

def _load_samples() -> List[Dict[str, Any]]:
    if not SAMPLES_PATH.exists():
        return []
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def _save_samples(samples: List[Dict[str, Any]]) -> None:
    SAMPLES_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image_rgb(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_base64_jpeg(image_rgb: np.ndarray, max_side: int = 640) -> str:
    h, w = image_rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_rgb = cv2.resize(
            image_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _to_base64_png(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()




# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class AnnotateRequest(BaseModel):
    t: float
    # label_source を省略した場合はサーバー側でルール決定:
    #   既存が auto_vN → "auto_reviewed"、それ以外 → "manual"
    label_source: Optional[str] = None


class ModelSelectRequest(BaseModel):
    paths: Optional[List[str]] = None   # 個別ファイル指定（後方互換）
    exp_dir: Optional[str] = None       # 実験フォルダ指定（fold*.pt を自動収集）


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
def get_models():
    """利用可能なモデルファイルの一覧と現在アクティブなモデルを返す。"""
    available = [str(p) for p in _discover_all_model_paths()]
    active = [str(p) for p in _active_model_paths]
    return {
        "available": available,
        "active": active,
        "loaded_count": len(_infer_models),
        "infer_mode": _active_infer_mode,
    }


@app.get("/api/models/folders")
def get_model_folders():
    """モデル(.pt)が含まれるフォルダ一覧と現在アクティブなフォルダを返す。"""
    folders = []

    # models/, models_kfold/ (ルートレベルのフラットなディレクトリ)
    for d in [_PROJECT_ROOT / "models", _PROJECT_ROOT / "models_kfold"]:
        if d.exists():
            pts = sorted(d.glob("*.pt"))
            if pts:
                rel = str(d.relative_to(_PROJECT_ROOT))
                folders.append({
                    "path": rel,
                    "name": d.name,
                    "fold_count": len([p for p in pts if "fold" in p.name]),
                    "has_ensemble": any("ensemble" in p.name for p in pts),
                    "file_count": len(pts),
                    "has_infer_script": False,
                })

    # experiments/ 配下の各サブディレクトリ
    exp_root = _PROJECT_ROOT / "experiments"
    if exp_root.exists():
        seen_dirs: set = set()
        for pt_file in sorted(exp_root.rglob("*.pt")):
            pt_dir = pt_file.parent
            if pt_dir in seen_dirs:
                continue
            seen_dirs.add(pt_dir)
            pts = sorted(pt_dir.glob("*.pt"))
            rel = str(pt_dir.relative_to(_PROJECT_ROOT))
            infer_script = _find_infer_script(pt_dir)
            folders.append({
                "path": rel,
                "name": pt_dir.name,
                "fold_count": len([p for p in pts if "fold" in p.name]),
                "has_ensemble": any("ensemble" in p.name for p in pts),
                "file_count": len(pts),
                "has_infer_script": infer_script is not None,
            })

    # 現在アクティブなフォルダ (アクティブモデルの親ディレクトリ)
    active_dir = None
    if _active_model_paths:
        parent = _active_model_paths[0].parent
        try:
            active_dir = str(parent.relative_to(_PROJECT_ROOT))
        except ValueError:
            active_dir = str(parent)

    return {
        "folders": folders,
        "active_dir": active_dir,
        "loaded_count": len(_infer_models),
        "infer_mode": _active_infer_mode,
    }


def _is_allowed_model_path(path: Path) -> bool:
    """パスが許可ディレクトリ内にあるか検証する（パストラバーサル防止）。"""
    try:
        # _PROJECT_ROOT を基準に解決することで、実行時のカレントディレクトリに依存しない
        resolved = (_PROJECT_ROOT / path).resolve()
        return any(resolved.is_relative_to(allowed) for allowed in _ALLOWED_MODEL_DIRS)
    except Exception:
        return False


@app.post("/api/models/select")
def select_models(body: ModelSelectRequest):
    """指定されたモデルファイルを読み込んでアクティブにする。

    exp_dir を指定した場合、そのフォルダから fold*.pt を自動収集する。
    EXP の scr/infer_exp*.py に create_predictor() があればそれを使う。
    fold*.pt がなければ ensemble.pt を使用する。
    paths を指定した場合は個別ファイルを直接ロード（後方互換）。
    """
    global _active_predictor, _active_infer_mode

    exp_native_predictor = None
    resolved_dir = None

    if body.exp_dir:
        exp_path = Path(body.exp_dir)
        if not _is_allowed_model_path(exp_path):
            raise HTTPException(
                status_code=400,
                detail=f"許可されていないパスです: {body.exp_dir}",
            )
        resolved_dir = (_PROJECT_ROOT / exp_path).resolve()
        fold_paths = sorted(resolved_dir.glob("eyeliner_detection_fold*.pt"))
        if fold_paths:
            paths = fold_paths
        else:
            ensemble = resolved_dir / "eyeliner_detection_ensemble.pt"
            if ensemble.exists():
                paths = [ensemble]
            else:
                # fallback: ディレクトリ内の全 .pt ファイル (models/ など)
                any_pts = sorted(resolved_dir.glob("*.pt"))
                if any_pts:
                    paths = any_pts
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"モデルファイルが見つかりません: {body.exp_dir}",
                    )

        # EXP固有のinfer_exp*.pyを探してcreate_predictorを読み込む
        infer_script = _find_infer_script(resolved_dir)
        if infer_script:
            exp_native_predictor = _load_exp_predictor(infer_script, resolved_dir)

    elif body.paths:
        paths = [Path(p) for p in body.paths]
        invalid = [str(p) for p in paths if not _is_allowed_model_path(p)]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"許可されていないパスが含まれています: {invalid}",
            )
    else:
        raise HTTPException(status_code=400, detail="exp_dir または paths を指定してください")

    # predictor と infer_mode を一括で渡して TOCTOU を防止
    active_predictor = exp_native_predictor
    active_mode = "exp_native" if exp_native_predictor is not None else "default"
    _reload_models(paths, predictor=active_predictor, infer_mode=active_mode)
    if exp_native_predictor is not None:
        logger.info("EXP固有predictor をアクティブにしました")

    return {
        "ok": True,
        "loaded_count": len(_infer_models),
        "active": [str(p) for p in _active_model_paths],
        "infer_mode": _active_infer_mode,
    }


@app.get("/api/samples")
def get_samples_summary():
    samples = _load_samples()
    annotated = sum(1 for s in samples if s.get("t") is not None)

    # label_source 別集計
    source_counts: Dict[str, int] = {}
    for s in samples:
        src = s.get("label_source") or ("unannotated" if s.get("t") is None else "manual")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "total": len(samples),
        "annotated": annotated,
        "label_source_counts": source_counts,
        "samples": [
            {
                "idx": s["idx"],
                "image_path": s["image_path"],
                "part_name": s["part_name"],
                "sub_key": s["sub_key"],
                "point_index": s["point_index"],
                "t": s.get("t"),
                "label_source": s.get("label_source"),
            }
            for s in samples
        ],
    }


@app.get("/api/resolve_idx/{idx_field}")
def resolve_idx(idx_field: int):
    """idx フィールド値から配列位置を解決する。
    samples.json 再生成後も idx フィールドで正しいサンプルを特定できる。
    """
    samples = _load_samples()
    for pos, s in enumerate(samples):
        if s.get("idx") == idx_field:
            return {"array_position": pos}
    raise HTTPException(status_code=404, detail=f"idx={idx_field} のサンプルが見つかりません")


@app.get("/api/sample/{idx}")
def get_sample(idx: int):
    samples = _load_samples()
    if idx < 0 or idx >= len(samples):
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    sample = samples[idx]
    px, py = sample["px"], sample["py"]
    qx, qy = sample["qx"], sample["qy"]
    # samples.json の t は生の物理 t 値（PQ 座標系）
    existing_t_raw: Optional[float] = float(sample["t"]) if sample.get("t") is not None else None
    # フロントエンドは t_norm [0,1] で動作するため、表示用に変換する
    existing_t: Optional[float] = (
        (existing_t_raw - T_MIN) / (T_MAX - T_MIN) if existing_t_raw is not None else None
    )

    try:
        image_rgb = _load_image_rgb(sample["image_path"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


    # R の現在位置（アノテーション済みなら表示）
    rx: Optional[float] = None
    ry: Optional[float] = None
    if existing_t_raw is not None:
        rx, ry = reconstruct_point(px, py, qx, qy, existing_t_raw)

    # 元画像オーバーレイ
    overlaid = overlay_landmarks(image_rgb, px, py, qx, qy, rx, ry)
    image_b64 = _to_base64_jpeg(overlaid)

    # 顔クロップ画像（P, Q 周辺の顔領域）
    try:
        cx = (px + qx) / 2
        cy = (py + qy) / 2
        pq_dist = float(np.hypot(qx - px, qy - py))
        face_margin = max(int(pq_dist * 1.5), 100)
        h_img, w_img = image_rgb.shape[:2]
        x1 = max(0, int(cx - face_margin))
        y1 = max(0, int(cy - face_margin))
        x2 = min(w_img, int(cx + face_margin))
        y2 = min(h_img, int(cy + face_margin * 1.5))
        face_crop_rgb = overlaid[y1:y2, x1:x2]
        if face_crop_rgb.size == 0:
            raise ValueError("face crop is empty")
        crop_h, crop_w = face_crop_rgb.shape[:2]
        face_crop_rgb = cv2.resize(
            face_crop_rgb,
            (crop_w * 3, crop_h * 3),
            interpolation=cv2.INTER_LINEAR,
        )
        face_b64 = _to_base64_jpeg(face_crop_rgb, max_side=1200)
    except Exception as exc:
        logger.warning("顔クロップ生成に失敗しました: %s", exc)
        face_b64 = None

    # ROIパッチ (display_scale 倍)
    patch_h, patch_w = PATCH_SIZE
    disp_rgb = build_display_patch(image_rgb, px, py, qx, qy, display_scale=DISPLAY_SCALE)

    # 既存の t がある場合は R 位置を緑線でオーバーレイ
    if existing_t is not None:
        x_in_disp = int((1.0 - existing_t) * (patch_w * DISPLAY_SCALE - 1))
        cv2.line(
            disp_rgb,
            (x_in_disp, 0),
            (x_in_disp, patch_h * DISPLAY_SCALE - 1),
            (0, 255, 0), 2,
        )

    patch_b64 = _to_base64_png(disp_rgb)

    # 推論補助
    is_upper = 1.0 if sample.get("sub_key", "") == "upper" else 0.0
    t_pred = _predict_t(image_rgb, px, py, qx, qy, is_upper)

    annotated = sum(1 for s in samples if s.get("t") is not None)

    return {
        "idx": idx,
        "total": len(samples),
        "annotated": annotated,
        "image_b64": image_b64,
        "face_b64": face_b64,
        "patch_b64": patch_b64,
        "patch_display_w": patch_w * DISPLAY_SCALE,
        "patch_display_h": patch_h * DISPLAY_SCALE,
        "px": px, "py": py,
        "qx": qx, "qy": qy,
        "existing_t": existing_t,
        "t_pred": t_pred,
        "part_name": sample["part_name"],
        "sub_key": sample["sub_key"],
        "point_index": sample["point_index"],
        "image_path": sample["image_path"],
        "label_source": sample.get("label_source"),
        "confidence": sample.get("confidence"),
    }


def _resolve_label_source(existing: Optional[str], requested: Optional[str]) -> str:
    """保存時の label_source を決定する。

    - クライアントが明示した場合はそれを使用（ただし "auto_vN" は上書き不可）
    - 既存が auto_vN で人手修正 → "auto_reviewed"
    - それ以外 → "manual"
    """
    if requested and not requested.startswith("auto_v"):
        return requested
    if existing and existing.startswith("auto_v"):
        return "auto_reviewed"
    return "manual"


@app.post("/api/annotate/{idx}")
def annotate(idx: int, body: AnnotateRequest):
    if not (0.0 <= body.t <= 1.0):
        raise HTTPException(status_code=422, detail="t は [0, 1] の範囲で指定してください")

    # フロントエンドから受け取る body.t は t_norm [0,1]。
    # samples.json には生の物理 t 値で保存する。
    t_raw = body.t * (T_MAX - T_MIN) + T_MIN

    with _samples_lock:
        samples = _load_samples()
        if idx < 0 or idx >= len(samples):
            raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

        def _update(i: int, s: dict) -> dict:
            if i != idx:
                return s
            existing_source = s.get("label_source")
            label_source = _resolve_label_source(existing_source, body.label_source)
            return {**s, "t": round(t_raw, 6), "label_source": label_source}

        samples = [_update(i, s) for i, s in enumerate(samples)]
        _save_samples(samples)

    saved = samples[idx]
    return {"ok": True, "idx": idx, "t": body.t, "label_source": saved.get("label_source")}


@app.post("/api/apply_pred_eye/{idx}")
def apply_pred_eye(idx: int):
    """現在サンプルと同一目（image_path + face_index + part_name）の未ラベルサンプルに予測を一括適用する。"""
    with _models_lock:
        has_model = len(_infer_models) > 0 or _active_predictor is not None
    if not has_model:
        raise HTTPException(status_code=400, detail="推論モデルが読み込まれていません")

    samples = _load_samples()
    if idx < 0 or idx >= len(samples):
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    current = samples[idx]
    image_path = current["image_path"]
    face_index = current.get("face_index")
    part_name = current.get("part_name", "")

    targets = [
        s for s in samples
        if s["image_path"] == image_path
        and s.get("face_index") == face_index
        and s.get("part_name") == part_name
        and s.get("t") is None
    ]

    if not targets:
        return {"ok": True, "updated": 0, "message": "未ラベルのサンプルはありません"}

    try:
        image_rgb = _load_image_rgb(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    predictions: Dict[int, float] = {}
    for s in targets:
        is_upper = 1.0 if s.get("sub_key", "") == "upper" else 0.0
        t_norm = _predict_t(image_rgb, float(s["px"]), float(s["py"]), float(s["qx"]), float(s["qy"]), is_upper)
        if t_norm is not None:
            predictions[s["idx"]] = t_norm * (T_MAX - T_MIN) + T_MIN

    if not predictions:
        raise HTTPException(status_code=500, detail="全サンプルの予測に失敗しました")

    with _samples_lock:
        samples = _load_samples()
        updated = [{**s, "t": predictions[s["idx"]], "label_source": "auto_v1"} if s["idx"] in predictions else s
                   for s in samples]
        _save_samples(updated)

    return {"ok": True, "updated": len(predictions)}


@app.get("/api/gallery")
def get_gallery():
    """各画像のサムネイルとアノテーション進捗を返す。"""
    samples = _load_samples()

    # image_path でグループ化（挿入順を保持）。first_pos は配列位置を記録する。
    groups: Dict[str, Dict] = {}
    for pos, s in enumerate(samples):
        key = s["image_path"]
        if key not in groups:
            groups[key] = {"samples": [], "first_pos": pos}
        groups[key]["samples"].append(s)

    result = []
    for image_path, group in groups.items():
        group_samples = group["samples"]
        total = len(group_samples)
        annotated = sum(1 for s in group_samples if s.get("t") is not None)
        first_idx = group["first_pos"]  # 配列位置（/api/sample/{idx} で使用）

        # キャッシュにあればそのまま使用（LRU: 末尾に移動）
        with _thumbnail_cache_lock:
            thumbnail_b64: Optional[str] = _thumbnail_cache.get(image_path)
            if thumbnail_b64 is not None:
                _thumbnail_cache.move_to_end(image_path)  # LRU 順序維持
        if thumbnail_b64 is None:
            try:
                image_rgb = _load_image_rgb(image_path)
                new_thumbnail = _to_base64_jpeg(image_rgb, max_side=200)
                with _thumbnail_cache_lock:
                    if len(_thumbnail_cache) >= _MAX_THUMBNAIL_CACHE:
                        _thumbnail_cache.popitem(last=False)  # 最も古いエントリを削除
                    _thumbnail_cache[image_path] = new_thumbnail
                thumbnail_b64 = new_thumbnail
            except Exception as exc:
                logger.warning("サムネイル生成に失敗しました %s: %s", image_path, exc)

        result.append({
            "image_key": Path(image_path).name,
            "thumbnail_b64": thumbnail_b64,
            "total": total,
            "annotated": annotated,
            "first_idx": first_idx,
        })

    return result


# ---------------------------------------------------------------------------
# Eyeliner preview rendering
# ---------------------------------------------------------------------------

def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 1,
    dash_len: int = 3,
) -> None:
    """OpenCV にはダッシュ線機能がないため手動実装。"""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = max(1, int(np.hypot(dx, dy)))
    step = dash_len * 2
    for i in range(0, dist, step):
        t1 = i / dist
        t2 = min((i + dash_len) / dist, 1.0)
        x1 = int(pt1[0] + dx * t1)
        y1 = int(pt1[1] + dy * t1)
        x2 = int(pt1[0] + dx * t2)
        y2 = int(pt1[1] + dy * t2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _render_eye_crop(
    image_rgb: np.ndarray,
    eye_samples: List[Dict[str, Any]],
    use_pred: bool,
) -> Optional[str]:
    """
    眼领域のクロップにアイラインを描画して base64 JPEG を返す。

    use_pred=False: sample["t"]（アノテーション）を使用。
    use_pred=True : _predict_t() でモデル予測を使用。

    R点が1つも得られない場合は None を返す。
    """
    if not eye_samples:
        return None

    # sub_key ごとに point_index 順に並べた R 点を収集
    points_by_sub: Dict[str, List[tuple]] = {"upper": [], "lower": []}

    for s in sorted(eye_samples, key=lambda x: x["point_index"]):
        px, py = float(s["px"]), float(s["py"])
        qx, qy = float(s["qx"]), float(s["qy"])
        sub = s.get("sub_key", "upper")

        if use_pred:
            is_upper_val = 1.0 if sub == "upper" else 0.0
            t_norm = _predict_t(image_rgb, px, py, qx, qy, is_upper_val)
            if t_norm is None:
                continue
            t_raw = t_norm * (T_MAX - T_MIN) + T_MIN
        else:
            # samples.json の t は生の物理 t 値
            t_raw_str = s.get("t")
            if t_raw_str is None:
                continue
            t_raw = float(t_raw_str)

        rx, ry = reconstruct_point(px, py, qx, qy, t_raw)

        key = sub if sub in points_by_sub else "upper"
        points_by_sub[key].append((rx, ry))

    all_r = points_by_sub["upper"] + points_by_sub["lower"]
    if not all_r:
        return None

    # バウンディングボックスを P/Q/R 全点から計算
    all_for_bbox = all_r + [
        (float(s["px"]), float(s["py"])) for s in eye_samples
    ] + [
        (float(s["qx"]), float(s["qy"])) for s in eye_samples
    ]
    xs = [p[0] for p in all_for_bbox]
    ys = [p[1] for p in all_for_bbox]

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = max(span * 0.1, 10)

    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, int(min(xs) - margin))
    y1 = max(0, int(min(ys) - margin))
    x2 = min(w_img, int(max(xs) + margin))
    y2 = min(h_img, int(max(ys) + margin))

    crop = image_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    # 色設定: t_true=緑系, t_pred=橙系
    if use_pred:
        color_upper = (0, 0,   0)
        color_lower = (0, 0,   0)
    else:
        color_upper = (  0, 0, 0)
        color_lower = (  0, 0,  0)

    # 先に拡大してから描画（拡大後の座標で thickness=1 にすることで線を細く保つ）
    crop_h, crop_w = crop.shape[:2]
    scale = max(5, min(8, 1200 // max(crop_h, crop_w, 5)))
    if scale > 1:
        crop = cv2.resize(
            crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_LANCZOS4
        )

    def to_crop(rx: float, ry: float) -> tuple:
        return (int((rx - x1) * scale), int((ry - y1) * scale))

    # upper: 実線
    upper_pts = [to_crop(rx, ry) for rx, ry in points_by_sub["upper"]]
    for i in range(len(upper_pts) - 1):
        cv2.line(crop, upper_pts[i], upper_pts[i + 1], color_upper, 1, cv2.LINE_AA)
    for pt in upper_pts:
        cv2.circle(crop, pt, 5, color_upper, -1)

    # lower: 破線
    lower_pts = [to_crop(rx, ry) for rx, ry in points_by_sub["lower"]]
    for i in range(len(lower_pts) - 1):
        _draw_dashed_line(crop, lower_pts[i], lower_pts[i + 1], color_lower, 1)
    for pt in lower_pts:
        cv2.circle(crop, pt, 5, color_lower, -1)

    return _to_base64_png(crop)


def _filter_same_face(reference: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    reference サンプルの PQ 中点から 2×PQ長 以内のサンプルを同一顔とみなして返す。

    判定根拠:
      - 同一眼の各点: 距離 ≈ 0
      - 同一顔の反対眼: 距離 ≈ 眼間距離 ≈ 1×PQ長
      - 別人の眼: 距離 >> 3×PQ長（通常のグループ撮影時）
    """
    ref_mx = (float(reference["px"]) + float(reference["qx"])) / 2
    ref_my = (float(reference["py"]) + float(reference["qy"])) / 2
    ref_pq = max(
        float(np.hypot(float(reference["qx"]) - float(reference["px"]),
                       float(reference["qy"]) - float(reference["py"]))),
        1.0,
    )
    threshold = ref_pq * 10.0
    return [
        s for s in candidates
        if float(np.hypot(
            (float(s["px"]) + float(s["qx"])) / 2 - ref_mx,
            (float(s["py"]) + float(s["qy"])) / 2 - ref_my,
        )) <= threshold
    ]


@app.get("/api/eyeliner_preview/{idx}")
def get_eyeliner_preview(idx: int):
    """
    現在サンプルと同一顔・同一側の眼のみを使いアイラインを描画した2枚の画像を返す。

    複数顔が存在する画像でも、現在サンプルの PQ 中点から 2×PQ長 以内かつ
    同一 part_name のサンプルのみを使用する。

    Returns:
        eye_true_b64: アノテーション(t_true)による現在側の目
        eye_pred_b64: モデル予測(t_pred)による現在側の目
        side: "left" | "right" | "unknown"
        has_model: モデルが読み込まれているか
    """
    samples = _load_samples()
    if idx < 0 or idx >= len(samples):
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    current = samples[idx]
    image_path = current["image_path"]
    part_name = current.get("part_name", "")
    image_samples = [s for s in samples if s["image_path"] == image_path]

    try:
        image_rgb = _load_image_rgb(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # 現在サンプルと同じ part_name（同一側の目）の中から同一顔のみ抽出
    same_side_all = [s for s in image_samples if s.get("part_name") == part_name]
    eye_samples = _filter_same_face(current, same_side_all)

    if part_name == "eyeliner_left":
        side = "left"
    elif part_name == "eyeliner_right":
        side = "right"
    else:
        side = "unknown"

    with _models_lock:
        has_model = len(_infer_models) > 0 or _active_predictor is not None

    return {
        "eye_true_b64": _render_eye_crop(image_rgb, eye_samples, use_pred=False),
        "eye_pred_b64": _render_eye_crop(image_rgb, eye_samples, use_pred=True) if has_model else None,
        "side": side,
        "has_model": has_model,
    }


_EXPORT_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "label_source", "confidence",
]


@app.get("/api/export")
def export_manifest():
    """アノテーション済みサンプルを manifest.csv としてダウンロード。"""
    samples = _load_samples()
    annotated = [s for s in samples if s.get("t") is not None]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_EXPORT_FIELDNAMES)
    writer.writeheader()
    for s in annotated:
        writer.writerow({
            "idx": s["idx"],
            "image_path": s["image_path"],
            "part_name": s["part_name"],
            "sub_key": s["sub_key"],
            "point_index": s["point_index"],
            "px": s["px"],
            "py": s["py"],
            "qx": s["qx"],
            "qy": s["qy"],
            "t": s["t"],
            "label_source": s.get("label_source") or "manual",
            "confidence": s.get("confidence") or "",
        })

    content = buf.getvalue().encode("utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=manifest.csv"},
    )


@app.post("/api/open_image/{idx}")
async def open_image(idx: int) -> Dict[str, Any]:
    """Windows の写真アプリで元画像を開く。"""
    samples = _load_samples()
    if idx < 0 or idx >= len(samples):
        raise HTTPException(status_code=404, detail="インデックスが範囲外です")
    image_path = Path(samples[idx]["image_path"]).resolve()
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"画像ファイルが見つかりません: {image_path}")
    os.startfile(str(image_path))
    return {"status": "ok", "path": str(image_path)}



if __name__ == "__main__":
    import uvicorn
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)

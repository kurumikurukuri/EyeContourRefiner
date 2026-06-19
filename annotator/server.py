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
import logging
import logging.handlers
import os
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルートを sys.path に追加（python annotator/server.py で起動した場合に必要）
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from eyeliner.data.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
    T_MIN,
    T_MAX,
    build_display_patch,
    extract_patch,
    get_pq_prime_dist,
    overlay_landmarks,
    reconstruct_point,
)


logger = logging.getLogger(__name__)

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

# 画像グルーピングインデックス（ギャラリー用）
_image_index: Optional[Dict[str, Dict[str, Any]]] = None  # image_path → {first_pos, total, annotated, skipped}
_image_order: Optional[List[str]] = None  # 挿入順の画像パスリスト
# DB 由来のギャラリーソートキャッシュ（遅延ロード）: col名 → {image_path → value}
_gallery_sort_cache: Dict[str, Optional[Dict[str, float]]] = {
    "eye_scale": None,
    "norm_eye_scale": None,
    "lap_var": None,
}

# モデル状態（_infer_models 等）の競合状態防止用ロック
_models_lock = threading.Lock()

# サムネイルキャッシュの競合状態防止用ロック
_thumbnail_cache_lock = threading.Lock()

# サムネイルキャッシュ上限（無制限メモリ使用防止）
_MAX_THUMBNAIL_CACHE = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動時にモデルとサンプルを読み込む。"""
    _setup_file_logging()
    _try_load_infer_model()
    _db.init_db()
    _build_image_index()
    total, annotated = _db.count_samples()
    logger.info("DB initialized: %d samples (%d annotated)", total, annotated)
    logger.info("Image index built: %d images", len(_image_order) if _image_order else 0)
    yield


app = FastAPI(title="Eyeliner Annotator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from annotator import db as _db  # noqa: E402
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
_active_predictor_dir: Optional[Path] = None  # EXP native predictor 使用時のアクティブディレクトリ

# ImageNet 正規化テンソル（デバイスに配置後に設定）
_imagenet_mean_t = None
_imagenet_std_t = None

# EXP固有predictor（create_predictor()が提供する callable）
_active_predictor = None
_active_infer_mode = "default"  # "default" | "exp_native"

# 推論時のパッチ抽出・ラベル正規化パラメータ（experiment.json から読み込む）
# _predict_t はこれらを使い、モデル出力を WebUI の t_norm [T_MIN, T_MAX] に変換する
_infer_patch_size: tuple = PATCH_SIZE
_infer_patch_t_min: float = T_MIN
_infer_patch_t_max: float = T_MAX
_infer_label_t_min: float = T_MIN
_infer_label_t_max: float = T_MAX
_infer_pq_rescale_mode: str = "none"
_infer_pq_rescale_factor: float = 1.0
_infer_patch_perp_scale: float = 1.0


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


def _load_generic_dual_predictor(models_dir: Path):
    """experiment.json の fusion_mode を見て共有の汎用 dual predictor を構築する。

    EXP固有の infer_exp*.py が存在しない、または create_predictor を実装していない
    dual 実験（fusion_mode を持つ experiment.json）向けのフォールバック。
    単一ブランチ実験（fusion_mode なし）では None を返し、通常の .pt 読み込みに委ねる。
    """
    try:
        from eyeliner.models.dual_predictor import build_dual_predictor
        predictor = build_dual_predictor(models_dir)
        logger.info("共有 dual predictor を読み込みました: %s", models_dir)
        return predictor
    except (FileNotFoundError, ValueError):
        return None
    except Exception as exc:
        logger.warning("共有 dual predictor の読み込みに失敗しました: %s", exc)
        return None


def _load_exp_params(models_dir: Path) -> dict:
    """experiment.json からパッチ抽出・ラベル正規化パラメータを読み込む。

    Returns: patch_size, patch_t_min, patch_t_max, label_t_min, label_t_max を含む dict。
             読み込み失敗時は空 dict（デフォルト値が使われる）。
    """
    import json as _json
    exp_json = models_dir / "experiment.json"
    if not exp_json.exists():
        return {}
    try:
        with open(exp_json, encoding="utf-8") as f:
            data = _json.load(f)
        hp = data.get("hyperparameters", {})
        params: dict = {}
        if "patch_h" in hp and "patch_w" in hp:
            params["patch_size"] = (int(hp["patch_h"]), int(hp["patch_w"]))
        if "patch_t_min" in hp:
            params["patch_t_min"] = float(hp["patch_t_min"])
        if "patch_t_max" in hp:
            params["patch_t_max"] = float(hp["patch_t_max"])
        # label_t_min/max が null なら patch 範囲と同じ（train.py の挙動に合わせる）
        lt_min = hp.get("label_t_min")
        lt_max = hp.get("label_t_max")
        params["label_t_min"] = float(lt_min) if lt_min is not None else params.get("patch_t_min", T_MIN)
        params["label_t_max"] = float(lt_max) if lt_max is not None else params.get("patch_t_max", T_MAX)
        # デュアルブランチ形式の場合 branch_b の設定を使う（pq_prime 計算に必要）
        branch_b_hp = hp.get("branch_b") or {}
        raw_mode   = hp.get("pq_rescale_mode")   or branch_b_hp.get("pq_rescale_mode",   "none")
        raw_factor = hp.get("pq_rescale_factor")
        if raw_factor is None:
            raw_factor = branch_b_hp.get("pq_rescale_factor", 1.0)
        params["pq_rescale_mode"]   = str(raw_mode)
        params["pq_rescale_factor"] = float(raw_factor)
        raw_perp = hp.get("patch_perp_scale")
        if raw_perp is None:
            raw_perp = (hp.get("branch_a") or {}).get("patch_perp_scale")
        params["patch_perp_scale"] = float(raw_perp) if raw_perp is not None else 1.0
        logger.info(
            "experiment.json 読み込み: patch_size=%s patch_t=[%.1f,%.1f] label_t=[%.1f,%.1f]"
            " pq_rescale=%s×%.2f perp_scale=%.1f",
            params.get("patch_size", PATCH_SIZE),
            params.get("patch_t_min", T_MIN), params.get("patch_t_max", T_MAX),
            params["label_t_min"], params["label_t_max"],
            params["pq_rescale_mode"], params["pq_rescale_factor"],
            params["patch_perp_scale"],
        )
        return params
    except Exception as exc:
        logger.warning("experiment.json の読み込みに失敗しました: %s", exc)
        return {}


def _reload_models(
    model_paths: List[Path],
    predictor=None,
    infer_mode: str = "default",
    patch_params: Optional[dict] = None,
    predictor_dir: Optional[Path] = None,
) -> None:
    """モデルをロードしてグローバル状態を一括で更新する。

    predictor と infer_mode も同一ロック内で設定することで
    _active_predictor / _active_infer_mode の TOCTOU を防止する。
    patch_params: experiment.json から読み込んだパッチ・ラベルパラメータ。
    predictor_dir: EXP native predictor 使用時のモデルディレクトリ（UI の active_dir 表示用）。
    """
    global _infer_models, _infer_device, _active_model_paths, _imagenet_mean_t, _imagenet_std_t, _active_predictor, _active_infer_mode
    global _infer_patch_size, _infer_patch_t_min, _infer_patch_t_max, _infer_label_t_min, _infer_label_t_max
    global _infer_pq_rescale_mode, _infer_pq_rescale_factor, _infer_patch_perp_scale, _active_predictor_dir
    # パッチパラメータを解決（patch_params が指定されていれば優先、なければデフォルト）
    new_patch_size  = tuple(patch_params["patch_size"])  if patch_params and "patch_size"  in patch_params else PATCH_SIZE
    new_pt_min      = patch_params["patch_t_min"]        if patch_params and "patch_t_min"  in patch_params else T_MIN
    new_pt_max      = patch_params["patch_t_max"]        if patch_params and "patch_t_max"  in patch_params else T_MAX
    new_lt_min      = patch_params["label_t_min"]        if patch_params and "label_t_min"  in patch_params else T_MIN
    new_lt_max      = patch_params["label_t_max"]        if patch_params and "label_t_max"  in patch_params else T_MAX
    new_pq_mode     = patch_params["pq_rescale_mode"]    if patch_params and "pq_rescale_mode"   in patch_params else "none"
    new_pq_factor   = patch_params["pq_rescale_factor"]  if patch_params and "pq_rescale_factor" in patch_params else 1.0
    new_perp_scale  = patch_params["patch_perp_scale"]   if patch_params and "patch_perp_scale"  in patch_params else 1.0

    # infer_mode == "exp_native" のとき、model_paths は EXP native predictor が
    # 既に .pt を内部で管理しているため二重読みしない。
    # paths が渡された場合も EyelinerCNN として読もうとして失敗し predictor が
    # 破棄されるバグを防ぐため、exp_native モードでは paths を無視する。
    if not model_paths or infer_mode == "exp_native":
        with _models_lock:
            _infer_models = []
            _active_model_paths = []
            _active_predictor = predictor
            _active_infer_mode = infer_mode
            _active_predictor_dir = predictor_dir
            _infer_patch_size  = new_patch_size
            _infer_patch_t_min = new_pt_min
            _infer_patch_t_max = new_pt_max
            _infer_label_t_min = new_lt_min
            _infer_label_t_max = new_lt_max
            _infer_pq_rescale_mode   = new_pq_mode
            _infer_pq_rescale_factor = new_pq_factor
            _infer_patch_perp_scale  = new_perp_scale
        if not model_paths:
            logger.info("モデルをクリアしました。推論補助は無効")
        else:
            logger.info("exp_native モード: .pt の二重読みをスキップし predictor を保持します")
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
                _active_predictor_dir = None
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
            _infer_patch_size  = new_patch_size
            _infer_patch_t_min = new_pt_min
            _infer_patch_t_max = new_pt_max
            _infer_label_t_min = new_lt_min
            _infer_label_t_max = new_lt_max
            _infer_pq_rescale_mode   = new_pq_mode
            _infer_pq_rescale_factor = new_pq_factor
            _infer_patch_perp_scale  = new_perp_scale
        logger.info(
            "%d 個のモデルをアンサンブルとして使用します (patch=%s t_patch=[%.1f,%.1f]"
            " t_label=[%.1f,%.1f] pq_rescale=%s×%.2f perp=%.1f)",
            len(loaded), new_patch_size, new_pt_min, new_pt_max, new_lt_min, new_lt_max,
            new_pq_mode, new_pq_factor, new_perp_scale,
        )
    except Exception as exc:
        logger.warning("推論モデルの読み込みに失敗しました: %s", exc)


def _try_load_infer_model() -> None:
    _reload_models(_find_model_paths())


def _get_pq_prime(
    sample: dict,
    orig_pq_dist: float,
    pq_rescale_mode: str,
    pq_rescale_factor: float,
    eye_pq_dists: Optional[List[float]] = None,
) -> float:
    """pq_rescale モードに応じた pq_prime を返す。逆変換不可時は orig_pq_dist を返す。"""
    if pq_rescale_mode == "none":
        return orig_pq_dist
    if pq_rescale_mode == "eye_scale":
        return get_pq_prime_dist(sample, orig_pq_dist, pq_rescale_mode, pq_rescale_factor)
    if pq_rescale_mode == "mean_pq_dist":
        if eye_pq_dists:
            mean_pq = sum(eye_pq_dists) / len(eye_pq_dists)
        else:
            mean_pq = _db.get_mean_pq_dist_for_eye(
                sample["image_path"],
                sample.get("face_index", 0),
                sample["part_name"],
            )
        if mean_pq is None or mean_pq <= 0:
            return orig_pq_dist
        return max(mean_pq * pq_rescale_factor, 1.0)
    # eye_height: DB から全点 y 座標を集めて算出する
    eye_height = _db.get_eye_height_for_eye(
        sample["image_path"],
        sample.get("face_index", 0),
        sample["part_name"],
    )
    if eye_height is None or eye_height <= 0:
        return orig_pq_dist
    return max(eye_height * pq_rescale_factor, 1.0)


def _predict_t(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    is_upper: float,
    *,
    pq_prime: Optional[float] = None,
) -> Optional[float]:
    """モデルで t を推論し、WebUI の t_norm [T_MIN, T_MAX] で返す。

    モデルの出力は [label_t_min, label_t_max] で正規化された t_prime_norm（pq_rescale 時は t' 空間）。
    pq_prime が指定された場合、t' → t_raw の逆変換を適用してから WebUI 正規化する。
    EXP固有predictor は自前で変換を行うため、この変換をスキップする。
    """
    # グローバル状態のスナップショットをロック下で取得してからロックを解放し推論を実行
    with _models_lock:
        predictor   = _active_predictor
        if predictor is None and not _infer_models:
            return None
        models      = list(_infer_models) if predictor is None else []
        device      = _infer_device
        mean_t      = _imagenet_mean_t
        std_t       = _imagenet_std_t
        ps          = _infer_patch_size
        pt_min      = _infer_patch_t_min
        pt_max      = _infer_patch_t_max
        lt_min      = _infer_label_t_min
        lt_max      = _infer_label_t_max
        perp_scale  = _infer_patch_perp_scale

    # EXP固有predictorを優先使用（変換は predictor 側で行う）
    if predictor is not None:
        try:
            return predictor(image_rgb, px, py, qx, qy, is_upper, pq_prime=pq_prime)
        except Exception as exc:
            logger.warning("EXP固有predictor の推論に失敗しました: %s", exc)
            return None

    try:
        import torch
        orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

        # pq_rescale モデルはパッチを P'Q' 座標系で抽出する。
        # pq_prime が orig_pq と異なるとき P'Q' を再計算してパッチ抽出に使う。
        if pq_prime is not None and abs(pq_prime - orig_pq) > 1e-6 and orig_pq > 1e-6:
            ox = (px + qx) / 2.0
            oy = (py + qy) / 2.0
            ux = (qx - px) / orig_pq
            uy = (qy - py) / orig_pq
            px_e = ox - (pq_prime / 2.0) * ux
            py_e = oy - (pq_prime / 2.0) * uy
            qx_e = ox + (pq_prime / 2.0) * ux
            qy_e = oy + (pq_prime / 2.0) * uy
        else:
            px_e, py_e, qx_e, qy_e = px, py, qx, qy

        patch = extract_patch(
            image_rgb, px_e, py_e, qx_e, qy_e,
            patch_size=ps, t_min=pt_min, t_max=pt_max, perp_scale=perp_scale,
        )
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        x = (x - mean_t) / std_t
        is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = [model(x, is_upper_t).item() for model in models]
        t_norm_label = float(sum(preds) / len(preds))

        # t_prime_norm → t_prime（pq_rescale 空間）
        t_prime = t_norm_label * (lt_max - lt_min) + lt_min

        # pq_rescale 逆変換: t_prime → t_raw
        # t_prime = 0.5 + (t_raw - 0.5) * orig_pq / pq_prime
        # => t_raw = 0.5 + (t_prime - 0.5) * pq_prime / orig_pq
        if pq_prime is not None and orig_pq > 1e-6:
            t_raw = 0.5 + (t_prime - 0.5) * pq_prime / orig_pq
        else:
            t_raw = t_prime

        return (t_raw - T_MIN) / (T_MAX - T_MIN)
    except TypeError as exc:
        logger.error("モデル呼び出しの引数エラー（バグ）: %s", exc)
        raise
    except Exception as exc:
        logger.warning("推論に失敗しました: %s", exc)
        return None


# サムネイルキャッシュ（image_path → base64文字列）— 上限付き OrderedDict で LRU 管理
_thumbnail_cache: OrderedDict[str, str] = OrderedDict()


# ---------------------------------------------------------------------------
# Image index (in-memory gallery cache built from DB)
# ---------------------------------------------------------------------------

def _is_numeric(val: Any) -> bool:
    """値が float に変換できるか確認する。"""
    if val is None or val == "":
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _build_image_index() -> None:
    """DB から画像パス単位のグループインデックスを構築する。"""
    global _image_index, _image_order, _gallery_sort_cache
    _image_index, _image_order = _db.get_image_index()
    _gallery_sort_cache = {"eye_scale": None, "norm_eye_scale": None, "lap_var": None}


def _ensure_gallery_sort_loaded(col: str) -> Dict[str, float]:
    """DB 由来のギャラリーソートキャッシュを返す（未ロードなら取得）。"""
    global _gallery_sort_cache
    if _gallery_sort_cache.get(col) is None:
        logger.info("gallery sort cache loading: %s", col)
        _gallery_sort_cache[col] = _db.get_col_min_per_image(col)
        logger.info("gallery sort cache ready: %s (%d images)", col, len(_gallery_sort_cache[col]))
    return _gallery_sort_cache.get(col) or {}


def _build_sample_feature_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    """サンプルに対応する eye_scale / norm_eye_scale / lap_var を DB カラムから返す。"""
    es  = sample.get("eye_scale")
    nes = sample.get("norm_eye_scale")
    lv  = sample.get("lap_var")
    return {
        "eye_scale":      round(es, 1)  if es  is not None else None,
        "norm_eye_scale": round(nes, 5) if nes is not None else None,
        "lap_var":        round(lv, 1)  if lv  is not None else None,
    }


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


class ModelSelectRequest(BaseModel):
    paths: Optional[List[str]] = None   # 個別ファイル指定（後方互換）
    exp_dir: Optional[str] = None       # 実験フォルダ指定（fold*.pt を自動収集）


class SkipImageRequest(BaseModel):
    skip: bool


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


DUMMY_T1_DIR = "__dummy_t1__"


def _make_dummy_t1_predictor():
    """常に t=1.0 を返すダミー予測器。"""
    def predict(image_rgb, px, py, qx, qy, is_upper, pq_prime=None):
        return 0.6666
    return predict


@app.get("/api/models/folders")
def get_model_folders():
    """モデル(.pt)が含まれるフォルダ一覧と現在アクティブなフォルダを返す。"""
    folders = []

    # ダミーモデル: t=1 (常に内眼角を予測)
    folders.append({
        "path": DUMMY_T1_DIR,
        "name": "t=1 (baseline)",
        "fold_count": 0,
        "has_ensemble": False,
        "file_count": 0,
        "has_infer_script": False,
        "is_dummy": True,
    })

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
    if _active_infer_mode == "dummy_t1":
        active_dir = DUMMY_T1_DIR
    elif _active_model_paths:
        parent = _active_model_paths[0].parent
        try:
            active_dir = str(parent.relative_to(_PROJECT_ROOT))
        except ValueError:
            active_dir = str(parent)
    elif _active_predictor_dir is not None:
        # EXP native predictor 使用時は _active_model_paths が空になるため
        # _active_predictor_dir からアクティブディレクトリを取得する
        try:
            active_dir = str(_active_predictor_dir.relative_to(_PROJECT_ROOT))
        except ValueError:
            active_dir = str(_active_predictor_dir)

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
        # ダミーモデル: t=1 常に内眼角を返す
        if body.exp_dir == DUMMY_T1_DIR:
            _reload_models([], predictor=_make_dummy_t1_predictor(), infer_mode="dummy_t1")
            return {
                "ok": True,
                "loaded_count": 0,
                "active": [DUMMY_T1_DIR],
                "infer_mode": "dummy_t1",
            }

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

        # infer_exp*.py が無い／create_predictor を実装していない dual 実験は
        # 共有の汎用 dual predictor ローダーへフォールバックする（単一ブランチなら None）
        if exp_native_predictor is None:
            exp_native_predictor = _load_generic_dual_predictor(resolved_dir)

        # experiment.json からパッチ・ラベルパラメータを読み込む
        exp_params = _load_exp_params(resolved_dir)

    elif body.paths:
        paths = [Path(p) for p in body.paths]
        invalid = [str(p) for p in paths if not _is_allowed_model_path(p)]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"許可されていないパスが含まれています: {invalid}",
            )
        # paths 指定の場合は最初のファイルの親ディレクトリから experiment.json を探す
        exp_params = _load_exp_params(paths[0].parent) if paths else {}
    else:
        raise HTTPException(status_code=400, detail="exp_dir または paths を指定してください")

    # predictor と infer_mode を一括で渡して TOCTOU を防止
    # exp_native_predictor がある場合は .pt の二重読みを避けるため paths=[] を渡す。
    # predictor は既に models_dir を参照して内部でモデルを保持しているため、
    # _reload_models に生の .pt パスを渡すと EyelinerCNN として読み込もうとして
    # 失敗し _active_predictor = None にリセットされるバグを防ぐ。
    active_predictor = exp_native_predictor
    active_mode = "exp_native" if exp_native_predictor is not None else "default"
    reload_paths = [] if exp_native_predictor is not None else paths
    _reload_models(
        reload_paths,
        predictor=active_predictor,
        infer_mode=active_mode,
        patch_params=exp_params,
        predictor_dir=resolved_dir if exp_native_predictor is not None else None,
    )
    if exp_native_predictor is not None:
        logger.info("EXP固有predictor をアクティブにしました")

    return {
        "ok": True,
        "loaded_count": len(_infer_models),
        "active": [str(p) for p in _active_model_paths],
        "infer_mode": _active_infer_mode,
        "patch_size": list(_infer_patch_size),
        "patch_t_min": _infer_patch_t_min,
        "patch_t_max": _infer_patch_t_max,
        "label_t_min": _infer_label_t_min,
        "label_t_max": _infer_label_t_max,
    }


@app.get("/api/samples")
def get_samples_summary():
    """サンプル集計を返す。大規模DB対応のため idx リストは含めない。"""
    total, annotated = _db.count_samples()
    return {
        "total": total,
        "annotated": annotated,
        "annotated_idxs": [],
        "skipped_idxs": [],
        "samples": [],
    }


# ---------------------------------------------------------------------------
# Server-side navigation (large-DB対応)
# ---------------------------------------------------------------------------

@app.get("/api/nav/adjacent/{idx}")
def nav_adjacent(idx: int, delta: int = 1):
    """idx から delta 方向の次の非スキップサンプル idx を返す。"""
    next_idx = _db.get_adjacent_idx(idx, delta)
    return {"idx": next_idx}


@app.get("/api/nav/next_eye/{idx}")
def nav_next_eye(idx: int):
    """idx が属する目グループの次の目グループ先頭 idx を返す。"""
    next_idx = _db.get_next_eye_start_idx(idx)
    return {"idx": next_idx}


@app.get("/api/nav/prev_eye/{idx}")
def nav_prev_eye(idx: int):
    """idx が属する目グループの前の目グループ先頭 idx を返す。"""
    prev_idx = _db.get_prev_eye_start_idx(idx)
    return {"idx": prev_idx}


@app.get("/api/nav/next_unannotated/{idx}")
def nav_next_unannotated(idx: int):
    """idx より後の未アノテーション・非スキップの idx を返す（末尾で折り返す）。"""
    next_idx = _db.get_next_unannotated_idx(idx)
    return {"idx": next_idx}


@app.get("/api/nav/random_unannotated")
def nav_random_unannotated():
    """未アノテーション・非スキップのランダムな idx を返す。"""
    rand_idx = _db.get_random_unannotated_idx()
    return {"idx": rand_idx}


@app.get("/api/nav/first")
def nav_first():
    """最初の非スキップサンプルの idx を返す。"""
    idx = _db.get_first_idx()
    return {"idx": idx if idx is not None else 0}


@app.get("/api/sample_groups")
def get_sample_groups():
    """(image_path, face_index, part_name) グループの idx 範囲を返す。"""
    groups = _db.get_sample_groups()
    return {"groups": groups}


@app.get("/api/resolve_idx/{idx_field}")
def resolve_idx(idx_field: int):
    """idx フィールド値から配列位置を解決する。
    DB では idx は PRIMARY KEY であり配列位置と一致する。
    """
    sample = _db.get_sample_by_idx(idx_field)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"idx={idx_field} のサンプルが見つかりません")
    return {"array_position": idx_field}


@app.get("/api/sample/{idx}")
def get_sample(idx: int):
    sample = _db.get_sample_by_idx(idx)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    px, py = sample["px"], sample["py"]
    qx, qy = sample["qx"], sample["qy"]
    # DB の t は生の物理 t 値（PQ 座標系）
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

        # パッチROI矩形を顔クロップに描画
        try:
            if pq_dist > 0:
                ux = (qx - px) / pq_dist
                uy = (qy - py) / pq_dist
                perp_x, perp_y = -uy, ux
                half_perp = pq_dist / 2
                tmax_x = T_MAX * px + (1 - T_MAX) * qx
                tmax_y = T_MAX * py + (1 - T_MAX) * qy
                tmin_x = T_MIN * px + (1 - T_MIN) * qx
                tmin_y = T_MIN * py + (1 - T_MIN) * qy
                corners_img = [
                    (tmax_x + half_perp * perp_x, tmax_y + half_perp * perp_y),
                    (tmin_x + half_perp * perp_x, tmin_y + half_perp * perp_y),
                    (tmin_x - half_perp * perp_x, tmin_y - half_perp * perp_y),
                    (tmax_x - half_perp * perp_x, tmax_y - half_perp * perp_y),
                ]
                corners_crop = np.array(
                    [[int((cx - x1) * 3), int((cy - y1) * 3)] for cx, cy in corners_img],
                    dtype=np.int32,
                )
                cv2.polylines(
                    face_crop_rgb, [corners_crop], isClosed=True,
                    color=(26, 120, 194), thickness=2, lineType=cv2.LINE_AA,
                )
        except Exception:
            pass  # パッチ矩形描画は非致命的

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
    orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
    pq_prime_val = _get_pq_prime(sample, orig_pq, _infer_pq_rescale_mode, _infer_pq_rescale_factor)
    t_pred = _predict_t(image_rgb, px, py, qx, qy, is_upper, pq_prime=pq_prime_val)

    total, annotated = _db.count_samples()

    return {
        "idx": idx,
        "total": total,
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
        "skip_flag": sample.get("skip_flag", False),
        **_build_sample_feature_fields(sample),
        "image_metadata": _db.get_image_labels(
            _db.DB_PATH, Path(sample["image_path"]).stem
        ),
    }


@app.post("/api/annotate/{idx}")
def annotate(idx: int, body: AnnotateRequest):
    if not (0.0 <= body.t <= 1.0):
        raise HTTPException(status_code=422, detail="t は [0, 1] の範囲で指定してください")

    # フロントエンドから受け取る body.t は t_norm [0,1]。
    # DB には生の物理 t 値で保存する。
    t_raw = round(body.t * (T_MAX - T_MIN) + T_MIN, 6)

    sample = _db.get_sample_by_idx(idx)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    action_type = "add" if sample.get("t") is None else "modify"
    _db.update_annotation(idx, t_raw)
    _build_image_index()
    logger.info("[annotate:%s] idx=%d t=%.4f", action_type, idx, body.t)
    return {"ok": True, "idx": idx, "t": body.t}


@app.post("/api/skip_image/{idx}")
def skip_image(idx: int, body: SkipImageRequest):
    """指定サンプルと同じ画像の全サンプルの skip_flag を一括更新する。"""
    sample = _db.get_sample_by_idx(idx)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    image_path = sample["image_path"]
    updated_count = _db.update_skip_flag(image_path, body.skip)
    _build_image_index()

    # Return affected idxs so frontend can update local state
    with _db._connect(_db.DB_PATH) as conn:
        affected = [r["idx"] for r in conn.execute(
            "SELECT idx FROM samples WHERE image_path = ?", (image_path,)
        ).fetchall()]

    logger.info("[skip] idx=%d skip=%s updated=%d", idx, body.skip, updated_count)
    return {"ok": True, "updated": updated_count, "skip": body.skip, "updated_idxs": affected}


@app.post("/api/apply_pred_eye/{idx}")
def apply_pred_eye(idx: int):
    """現在サンプルと同一目（image_path + face_index + part_name）の未ラベルサンプルに予測を一括適用する。"""
    with _models_lock:
        has_model = len(_infer_models) > 0 or _active_predictor is not None
    if not has_model:
        raise HTTPException(status_code=400, detail="推論モデルが読み込まれていません")

    current = _db.get_sample_by_idx(idx)
    if current is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    image_path = current["image_path"]
    face_index = current.get("face_index")
    part_name = current.get("part_name", "")

    targets = _db.get_unannotated_samples_for_eye(image_path, face_index, part_name)
    if not targets:
        return {"ok": True, "updated": 0, "message": "未ラベルのサンプルはありません"}

    try:
        image_rgb = _load_image_rgb(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # mean_pq_dist 逆変換用: targets はすべて同一眼グループなので一括計算
    eye_pq_dists = [
        math.sqrt((float(s["qx"]) - float(s["px"])) ** 2 + (float(s["qy"]) - float(s["py"])) ** 2)
        for s in targets
    ]
    updates: List[tuple] = []
    for s in targets:
        is_upper = 1.0 if s.get("sub_key", "") == "upper" else 0.0
        orig_pq = math.sqrt((float(s["qx"]) - float(s["px"])) ** 2 + (float(s["qy"]) - float(s["py"])) ** 2)
        pq_prime_val = _get_pq_prime(s, orig_pq, _infer_pq_rescale_mode, _infer_pq_rescale_factor, eye_pq_dists)
        t_norm = _predict_t(
            image_rgb, float(s["px"]), float(s["py"]),
            float(s["qx"]), float(s["qy"]), is_upper,
            pq_prime=pq_prime_val,
        )
        if t_norm is not None:
            t_raw = t_norm * (T_MAX - T_MIN) + T_MIN
            updates.append((s["idx"], t_raw))

    if not updates:
        raise HTTPException(status_code=500, detail="全サンプルの予測に失敗しました")

    _db.batch_update_t(updates)
    _build_image_index()
    logger.info("[apply_pred] idx=%d updated=%d", idx, len(updates))
    return {"ok": True, "updated": len(updates)}


@app.post("/api/reset_eye/{idx}")
def reset_eye(idx: int):
    """現在サンプルと同一目（image_path + face_index + part_name）の全サンプルの t を NULL にリセットする。"""
    current = _db.get_sample_by_idx(idx)
    if current is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    image_path = current["image_path"]
    face_index = current.get("face_index")
    part_name = current.get("part_name", "")

    updated = _db.reset_t_for_eye(image_path, face_index, part_name)
    _build_image_index()
    logger.info("[reset_eye] idx=%d updated=%d", idx, updated)
    return {"ok": True, "updated": updated}


@app.get("/api/gallery")
def get_gallery(
    page: int = 1,
    per_page: int = Query(80, ge=1, le=200),
    skip_filter: str = Query("all"),
    sort_by: str = Query("idx"),
):
    """ページネーション対応のギャラリー。メタデータのみ返す（サムネイルは別API）。

    sort_by:
      "idx"                  — デフォルト（DB挿入順）
      "eye_scale_asc/desc"   — eye_scale
      "norm_eye_scale_asc/desc" — 正規化 eye_scale
      "lap_var_asc/desc"     — lap_var
    """
    if _image_index is None or _image_order is None:
        _build_image_index()

    # skip フィルタリング
    filtered_order = list(_image_order)
    if skip_filter == "active":
        filtered_order = [p for p in _image_order if not _image_index[p].get("skipped", False)]
    elif skip_filter == "skipped":
        filtered_order = [p for p in _image_order if _image_index[p].get("skipped", False)]

    # ソート: (col, reverse, missing_sentinel)
    _SORT_COL_MAP = {
        "eye_scale_asc":       ("eye_scale",      False, float("inf")),
        "eye_scale_desc":      ("eye_scale",      True,  float("inf")),
        "norm_eye_scale_asc":  ("norm_eye_scale", False, float("inf")),
        "norm_eye_scale_desc": ("norm_eye_scale", True,  float("inf")),
        "lap_var_asc":         ("lap_var",        False, float("inf")),
        "lap_var_desc":        ("lap_var",        True,  float("inf")),
    }
    sort_col_info = _SORT_COL_MAP.get(sort_by)
    if sort_col_info:
        col, reverse, missing = sort_col_info
        sort_index = _ensure_gallery_sort_loaded(col)
        filtered_order = sorted(
            filtered_order,
            key=lambda p: sort_index.get(p, missing),
            reverse=reverse,
        )

    total_images = len(filtered_order)
    total_pages = max(1, (total_images + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = min(start + per_page, total_images)
    page_images = filtered_order[start:end]

    # 表示用キャッシュ（ロード済みのみ; バッジ表示用）
    es_index  = _gallery_sort_cache.get("eye_scale")       or {}
    nes_index = _gallery_sort_cache.get("norm_eye_scale")  or {}
    lv_index  = _gallery_sort_cache.get("lap_var")         or {}

    items = []
    for image_path in page_images:
        entry = _image_index[image_path]
        es  = es_index.get(image_path)
        nes = nes_index.get(image_path)
        lv  = lv_index.get(image_path)
        items.append({
            "image_key":       Path(image_path).name,
            "image_path":      image_path,
            "total":           entry["total"],
            "annotated":       entry["annotated"],
            "first_idx":       entry["first_pos"],
            "skipped":         entry.get("skipped", False),
            "eye_scale":       round(es,  1)  if es  is not None else None,
            "norm_eye_scale":  round(nes, 5)  if nes is not None else None,
            "lap_var":         round(lv,  1)  if lv  is not None else None,
        })

    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total_images": total_images,
        "total_pages": total_pages,
    }


@app.get("/api/sample_order")
def get_sample_order(sort_by: str = Query("idx")):
    """ソート順のサンプル idx リストを返す。

    sort_by: "idx" | "eye_scale_asc" | "eye_scale_desc"
           | "norm_eye_scale_asc" | "norm_eye_scale_desc"
           | "lap_var_asc" | "lap_var_desc"
    """
    if sort_by == "idx":
        return {"order": _db.get_sample_order_by_idx()}

    _COL_MAP = {
        "eye_scale_asc":       ("eye_scale",      "ASC"),
        "eye_scale_desc":      ("eye_scale",      "DESC"),
        "norm_eye_scale_asc":  ("norm_eye_scale", "ASC"),
        "norm_eye_scale_desc": ("norm_eye_scale", "DESC"),
        "lap_var_asc":         ("lap_var",        "ASC"),
        "lap_var_desc":        ("lap_var",        "DESC"),
    }
    if sort_by in _COL_MAP:
        col, direction = _COL_MAP[sort_by]
        return {"order": _db.get_sample_order_by_col(col, direction)}

    raise HTTPException(status_code=400, detail=f"Unknown sort_by: {sort_by}")


@app.get("/api/gallery/find_page")
def gallery_find_page(image_key: str = "", per_page: int = Query(80, ge=1, le=200)):
    """指定画像が含まれるページ番号を返す。"""
    if _image_order is None:
        _build_image_index()
    for i, path in enumerate(_image_order):
        if Path(path).name == image_key:
            return {"page": (i // per_page) + 1, "image_index": i}
    return {"page": 1, "image_index": 0}


@app.get("/api/thumbnail")
def get_thumbnail(image_path: str, max_side: int = 200):
    """個別サムネイル画像をJPEGで返す。ブラウザキャッシュを利用。"""
    # パストラバーサル防止: DB に登録されたパスのみ許可
    if not _db.image_path_exists(image_path):

        raise HTTPException(status_code=404, detail=f"画像が見つかりません: {image_path}")

    # LRUキャッシュを確認
    with _thumbnail_cache_lock:
        thumbnail_b64 = _thumbnail_cache.get(image_path)
        if thumbnail_b64 is not None:
            _thumbnail_cache.move_to_end(image_path)

    if thumbnail_b64 is None:
        try:
            image_rgb = _load_image_rgb(image_path)
            thumbnail_b64 = _to_base64_jpeg(image_rgb, max_side=max_side)
            with _thumbnail_cache_lock:
                if len(_thumbnail_cache) >= _MAX_THUMBNAIL_CACHE:
                    _thumbnail_cache.popitem(last=False)
                _thumbnail_cache[image_path] = thumbnail_b64
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"サムネイル生成失敗: {exc}")

    img_bytes = base64.b64decode(thumbnail_b64)
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


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
    eye_scale: Optional[float] = None,
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

    # mean_pq_dist 逆変換用: eye_samples は同一眼グループ全点なので一括計算
    eye_pq_dists = [
        math.sqrt((float(s["qx"]) - float(s["px"])) ** 2 + (float(s["qy"]) - float(s["py"])) ** 2)
        for s in eye_samples
    ]

    for s in sorted(eye_samples, key=lambda x: x["point_index"]):
        px, py = float(s["px"]), float(s["py"])
        qx, qy = float(s["qx"]), float(s["qy"])
        sub = s.get("sub_key", "upper")

        if use_pred:
            is_upper_val = 1.0 if sub == "upper" else 0.0
            orig_pq = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
            pq_prime_val = _get_pq_prime(s, orig_pq, _infer_pq_rescale_mode, _infer_pq_rescale_factor, eye_pq_dists)
            t_norm = _predict_t(image_rgb, px, py, qx, qy, is_upper_val, pq_prime=pq_prime_val)
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

    # 色設定: upper/lower ともに青 (BGR)
    color_upper = (0, 0, 255)
    color_lower = color_upper

    # 先に拡大してから描画（拡大後の座標で thickness=1 にすることで線を細く保つ）
    crop_h, crop_w = crop.shape[:2]
    scale = max(5, min(8, 1200 // max(crop_h, crop_w, 5)))
    if scale > 1:
        crop = cv2.resize(
            crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_LANCZOS4
        )

    def to_crop(rx: float, ry: float) -> tuple:
        return (int((rx - x1) * scale), int((ry - y1) * scale))

    # eye_scale に応じてドット・ライン太さをスケール
    _BASE_EYE_SCALE = 60.0
    _es_factor = max(1.0, (eye_scale or _BASE_EYE_SCALE) / _BASE_EYE_SCALE)
    dot_radius = max(4, min(20, round(7 * _es_factor)))
    line_thickness = max(1, min(5, round(_es_factor)))

    # upper: 実線
    upper_pts = [to_crop(rx, ry) for rx, ry in points_by_sub["upper"]]
    for i in range(len(upper_pts) - 1):
        cv2.line(crop, upper_pts[i], upper_pts[i + 1], color_upper, line_thickness, cv2.LINE_AA)
    for pt in upper_pts:
        cv2.circle(crop, pt, dot_radius, color_upper, -1)

    # lower: 破線
    lower_pts = [to_crop(rx, ry) for rx, ry in points_by_sub["lower"]]
    for i in range(len(lower_pts) - 1):
        _draw_dashed_line(crop, lower_pts[i], lower_pts[i + 1], color_lower, line_thickness)
    for pt in lower_pts:
        cv2.circle(crop, pt, dot_radius, color_lower, -1)

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
    current = _db.get_sample_by_idx(idx)
    if current is None:
        raise HTTPException(status_code=404, detail=f"サンプル {idx} が見つかりません")

    image_path = current["image_path"]
    part_name = current.get("part_name", "")
    image_samples = _db.get_samples_by_image_path(image_path)

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

    current_eye_scale = current.get("eye_scale")
    es = float(current_eye_scale) if current_eye_scale is not None else None

    return {
        "eye_true_b64": _render_eye_crop(image_rgb, eye_samples, use_pred=False, eye_scale=es),
        "eye_pred_b64": _render_eye_crop(image_rgb, eye_samples, use_pred=True, eye_scale=es) if has_model else None,
        "side": side,
        "has_model": has_model,
    }


_EXPORT_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
]


@app.get("/api/export")
def export_manifest():
    """アノテーション済みサンプルを manifest.csv としてダウンロード。"""
    annotated = _db.get_annotated_samples()

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
    sample = _db.get_sample_by_idx(idx)
    if sample is None:
        raise HTTPException(status_code=404, detail="インデックスが範囲外です")
    image_path = Path(sample["image_path"]).resolve()
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"画像ファイルが見つかりません: {image_path}")
    os.startfile(str(image_path))
    return {"status": "ok", "path": str(image_path)}



def _setup_file_logging() -> None:
    """ログファイルハンドラを登録する。lifespan で呼ばれるため任意の起動方法で有効。"""
    logging.basicConfig(level=logging.INFO)
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in root.handlers):
        return
    _file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "annotator.log",
        when="midnight",
        backupCount=90,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(_file_handler)


if __name__ == "__main__":
    import uvicorn
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    _setup_file_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)

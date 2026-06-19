"""CNN 推論器 (EyePredictor / DualEyePredictor) の LRU キャッシュ。

eye_crop_renderer.make_predictor は重いモデルロードを伴うため、
モデルディレクトリ文字列をキーに上限3件の LRU キャッシュで保持する。
同一キーへの同時ロードはロックで直列化し、二重ロードを防ぐ。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from tools.annotation_support.t_pred_webui import eye_crop_renderer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MAX_CACHE_SIZE = 3

# key (resolved model_dir 文字列) → 推論器インスタンス。
# OrderedDict で挿入順を保持し、アクセス時に末尾へ移動して LRU を実現する。
_cache: "OrderedDict[str, object]" = OrderedDict()
_cache_lock = threading.Lock()
# キーごとのロード用ロック（同一キーへの同時ロードを直列化）
_key_locks: dict = {}
_key_locks_lock = threading.Lock()


def _reset_for_test() -> None:
    """テスト間の状態汚染を防ぐためキャッシュを全クリアする（テスト専用）。"""
    with _cache_lock:
        _cache.clear()
    with _key_locks_lock:
        _key_locks.clear()


def _resolve_safe_model_dir(model_dir: str) -> Optional[Path]:
    """model_dir をプロジェクトルート配下のパスとして解決する。

    パストラバーサル対策: ルート外に解決される場合は None を返す。
    experiment.json が存在しない場合も None を返す。
    """
    try:
        resolved = Path(model_dir).resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(_PROJECT_ROOT):
        return None
    if not (resolved / "experiment.json").exists():
        return None
    return resolved


def _get_key_lock(key: str) -> threading.Lock:
    with _key_locks_lock:
        if key not in _key_locks:
            _key_locks[key] = threading.Lock()
        return _key_locks[key]


def get_predictor(model_dir: str, device: str = "cpu") -> Optional[object]:
    """model_dir に対応する推論器をキャッシュから取得、なければロードする。

    不正なパス（ルート外 or experiment.json 不在）、または experiment.json が
    壊れている等でモデルロード自体が失敗した場合も None を返す。
    """
    resolved = _resolve_safe_model_dir(model_dir)
    if resolved is None:
        return None
    key = str(resolved)

    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    # ロード処理は同一キーごとに直列化する（二重ロード防止）
    key_lock = _get_key_lock(key)
    with key_lock:
        with _cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]

        try:
            predictor = eye_crop_renderer.make_predictor(resolved, device=device)
        except Exception:
            logger.warning("モデルロード失敗: %s", resolved, exc_info=True)
            return None

        with _cache_lock:
            _cache[key] = predictor
            _cache.move_to_end(key)
            while len(_cache) > _MAX_CACHE_SIZE:
                _cache.popitem(last=False)

    return predictor

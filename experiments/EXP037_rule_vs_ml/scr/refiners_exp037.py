"""EXP037: ルールベース補正（face_makeup）と ML 補正（eye_contour_refiner.onnx）の共通インターフェース。

どちらも「RGB 画像 + P,Q 座標（N 点）」を受け取り、補正点 R（N 点）を返す。
処理時間の内訳（ルール: 前処理 / エッジ探索、ML: パッチ切り出し / 推論）も返す。

- ルールベース: face_makeup（commit 09034270）の EyelinerCorrector をそのまま呼ぶ。
  correct_points と同じく preprocess() を 1 回 → find_edge(return_debug_info=True) を点ごとに実行する。
  correct_points は 1708 点の拡張ランドマーク配列を前提とするため、P,Q を直接渡せる find_edge を使う
  （correct_points との一致は tests/test_exp037_refiners.py で検証）。
- ML: models/infer_example.py（EyeContourRefiner main）と同じ前処理を N 点まとめて 1 バッチで推論する。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from face_makeup.core.eyeliner_corrector import EyelinerCorrector

from eyeliner.data.patch_utils import (
    compute_pq_rescale,
    compute_t_from_point,
    extract_patch,
)
from eyeliner.training.dual_branch import branch_a_x0, branch_b_eye_scale

_CFG_A = branch_a_x0()
_CFG_B = branch_b_eye_scale()


@dataclass(frozen=True)
class RuleResult:
    points: np.ndarray      # (N, 2) 補正点。face_makeup の仕様で整数に切り捨て済み
    fallback: np.ndarray    # (N,) bool。エッジが見つからず P をそのまま返した点
    t_untruncated: np.ndarray  # (N,) 切り捨て前の検出位置をモデルの t 規約に直した値（参考。fallback は 1.0）
    preprocess_s: float     # bilateralFilter → グレースケール → 正規化
    search_s: float         # find_edge × N


@dataclass(frozen=True)
class MlResult:
    t: np.ndarray           # (N,) 生 t
    points: np.ndarray      # (N, 2) R = t·P + (1-t)·Q
    patch_s: float          # パッチ切り出し（branch A/B）+ 入力テンソル作成
    infer_s: float          # session.run


def eye_scale_of(pq: np.ndarray) -> float:
    """1 つの目の全 P,Q 点の bounding box 対角長（最小 1.0）。eyeliner.data.weights.compute_eye_meta と同じ定義。"""
    xs = np.concatenate([pq[:, 0], pq[:, 2]])
    ys = np.concatenate([pq[:, 1], pq[:, 3]])
    scale = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    return max(scale, 1.0)


def to_production_coords(pq: np.ndarray) -> np.ndarray:
    """face_makeup 本番と同じ座標の丸めを適用する。

    本番（renderer.py）は FaceLandmarks.to_image_coords で座標を np.round → int32 にしてから
    correct_points に渡すため、ルールベースへの入力はこの丸め後の座標にする。
    """
    return np.round(pq).astype(np.int32).astype(np.float64)


def refine_rule(image_rgb: np.ndarray, pq: np.ndarray) -> RuleResult:
    """face_makeup のルールベース補正で N 点を補正する。

    前処理キャッシュを持ち越さないよう、呼び出しごとに新しい EyelinerCorrector を使う
    （同じ画像で繰り返し計測しても毎回 bilateralFilter が走る）。
    """
    corrector = EyelinerCorrector()

    start = time.perf_counter()
    gray = corrector.preprocess(image_rgb)
    preprocessed = time.perf_counter()
    found = [corrector.find_edge(gray, px, py, qx, qy, return_debug_info=True) for px, py, qx, qy in pq]
    end = time.perf_counter()

    points = np.array([point for point, _ in found], dtype=np.float64).reshape(-1, 2)
    fallback = np.array([debug is None or debug["best_idx"] is None for _, debug in found], dtype=bool)
    t_untruncated = np.array(
        [1.0 if fell_back else 1.0 - float(debug["t_values"][debug["best_idx"]])
         for (_, debug), fell_back in zip(found, fallback)],
        dtype=np.float64,
    )
    return RuleResult(points, fallback, t_untruncated, preprocessed - start, end - preprocessed)


def rule_point_to_t(pq: np.ndarray, points: np.ndarray) -> np.ndarray:
    """補正点を P-Q 直線に射影し、モデルの規約 R = t·P + (1-t)·Q の t に変換する。"""
    return np.array(
        [compute_t_from_point(px, py, qx, qy, rx, ry) for (px, py, qx, qy), (rx, ry) in zip(pq, points)],
        dtype=np.float64,
    )


def _patches(image_rgb: np.ndarray, pq: np.ndarray, eye_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patches_a, patches_b = [], []
    for (px, py, qx, qy), scale in zip(pq, eye_scale):
        patches_a.append(extract_patch(
            image_rgb, px, py, qx, qy,
            patch_size=_CFG_A.patch_size, t_min=_CFG_A.patch_t_min, t_max=_CFG_A.patch_t_max,
            perp_scale=_CFG_A.patch_perp_scale,
        ))
        bpx, bpy, bqx, bqy, _, _ = compute_pq_rescale(
            px, py, qx, qy, 0.0, {"eye_scale": scale}, _CFG_B.pq_rescale_mode, _CFG_B.pq_rescale_factor,
        )
        patches_b.append(extract_patch(
            image_rgb, bpx, bpy, bqx, bqy,
            patch_size=_CFG_B.patch_size, t_min=_CFG_B.patch_t_min, t_max=_CFG_B.patch_t_max,
            perp_scale=_CFG_B.patch_perp_scale,
        ))
    to_nchw = lambda patches: np.ascontiguousarray(np.stack(patches).transpose(0, 3, 1, 2), dtype=np.float32)
    return to_nchw(patches_a), to_nchw(patches_b)


def refine_ml(
    session,
    image_rgb: np.ndarray,
    pq: np.ndarray,
    is_upper: np.ndarray,
    eye_scale: np.ndarray,
) -> MlResult:
    """eye_contour_refiner.onnx で N 点を 1 バッチで補正する。eye_scale は点ごと（同じ目なら同じ値）。"""
    column = lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1)

    start = time.perf_counter()
    patch_a, patch_b = _patches(image_rgb, pq, eye_scale)
    feed = {
        "patch_a": patch_a, "patch_b": patch_b,
        "is_upper": column(is_upper),
        "px": column(pq[:, 0]), "py": column(pq[:, 1]),
        "qx": column(pq[:, 2]), "qy": column(pq[:, 3]),
        "eye_scale": column(eye_scale),
    }
    prepared = time.perf_counter()
    t = session.run(None, feed)[0].reshape(-1).astype(np.float64)
    end = time.perf_counter()

    points = t[:, None] * pq[:, :2] + (1.0 - t[:, None]) * pq[:, 2:]
    return MlResult(t, points, prepared - start, end - prepared)

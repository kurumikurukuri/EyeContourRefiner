"""スタンドアロン眼クロップレンダラー。

アノテーターサーバーに依存せず、CNN モデルを直接ロードして
Catmull-Rom スプライン曲線付きの眼クロップ画像を生成する。

upper 0-6: 実線ブルー
lower 0-6: 破線ブルー
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from eyeliner.data.patch_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_patch,
    reconstruct_point,
)

_BLUE_RGB = (0, 0, 255)


# ---------------------------------------------------------------------------
# Catmull-Rom スプライン
# ---------------------------------------------------------------------------

def _catmull_rom_chain(
    pts: List[Tuple[float, float]],
    n_per_seg: int = 20,
) -> List[Tuple[int, int]]:
    """Catmull-Rom スプラインで補間した pixel 座標リストを返す。

    両端にファントム点を追加してC1連続を確保する。
    """
    n = len(pts)
    if n == 0:
        return []
    if n == 1:
        return [(int(round(pts[0][0])), int(round(pts[0][1])))]
    if n == 2:
        result = []
        for j in range(n_per_seg + 1):
            t = j / n_per_seg
            x = pts[0][0] * (1 - t) + pts[1][0] * t
            y = pts[0][1] * (1 - t) + pts[1][1] * t
            result.append((int(round(x)), int(round(y))))
        return result

    p_start = (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1])
    p_end = (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1])
    aug = [p_start] + list(pts) + [p_end]

    result: List[Tuple[int, int]] = []
    for i in range(1, len(aug) - 2):
        p0, p1, p2, p3 = aug[i - 1], aug[i], aug[i + 1], aug[i + 2]
        for j in range(n_per_seg):
            tt = j / n_per_seg
            tt2 = tt * tt
            tt3 = tt2 * tt
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * tt
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * tt2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * tt3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * tt
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * tt2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * tt3
            )
            result.append((int(round(x)), int(round(y))))

    result.append((int(round(pts[-1][0])), int(round(pts[-1][1]))))
    return result


# ---------------------------------------------------------------------------
# 曲線描画
# ---------------------------------------------------------------------------

def _draw_solid_curve(
    img: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


def _draw_dashed_curve(
    img: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    dash_px: int = 8,
    gap_px: int = 5,
) -> None:
    """Catmull-Rom チェーン上をアーク長ベースのダッシュで描画する。"""
    acc = 0.0
    cycle = dash_px + gap_px
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 0.5:
            continue
        t_seg = 0.0
        while t_seg < seg:
            phase = acc % cycle
            if phase < dash_px:
                end = min(seg, t_seg + (dash_px - phase))
                a = t_seg / seg
                b = end / seg
                cv2.line(
                    img,
                    (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
                    (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)),
                    color, thickness, cv2.LINE_AA,
                )
                dt = end - t_seg
            else:
                dt = min(seg - t_seg, cycle - phase)
            acc += dt
            t_seg += dt


# ---------------------------------------------------------------------------
# CNN 推論器
# ---------------------------------------------------------------------------

class EyePredictor:
    """単一ブランチ EyelinerCNN アンサンブルを直接ロードする推論器。

    experiment.json からモデルパラメータを自動読み取りする。
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from eyeliner.models.ensemble import create_ensemble_from_dir

        model_dir = Path(model_dir)
        exp_json = model_dir / "experiment.json"
        cfg: Dict[str, Any] = {}
        if exp_json.exists():
            with open(exp_json, encoding="utf-8") as f:
                cfg = json.load(f).get("hyperparameters", {})

        self._patch_size = (int(cfg.get("patch_h", 64)), int(cfg.get("patch_w", 128)))
        self._patch_t_min = float(cfg.get("patch_t_min", -3.0))
        self._patch_t_max = float(cfg.get("patch_t_max", 4.0))
        self._label_t_min = float(cfg.get("label_t_min", -1.0))
        self._label_t_max = float(cfg.get("label_t_max", 2.0))
        self._use_is_upper = bool(cfg.get("use_is_upper", True))
        self._device = device

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1).to(device)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1).to(device)

        ensemble, _ = create_ensemble_from_dir(model_dir)
        ensemble.eval()
        self._ensemble = ensemble.to(device)
        self.loaded_count = len(list(ensemble._models))

    def predict_t_raw(
        self,
        image_rgb: np.ndarray,
        px: float, py: float,
        qx: float, qy: float,
        is_upper: float,
        eye_scale: Optional[float] = None,
    ) -> Optional[float]:
        """1点の生 t 値（物理 t 空間）を予測する。失敗時は None。eye_scale は未使用（デュアルブランチ互換引数）。"""
        import torch

        try:
            patch = extract_patch(
                image_rgb, px, py, qx, qy,
                patch_size=self._patch_size,
                t_min=self._patch_t_min,
                t_max=self._patch_t_max,
            )
            x = torch.from_numpy(patch.transpose(2, 0, 1)).float().to(self._device)
            x = (x - self._mean) / self._std
            x = x.unsqueeze(0)
            is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=self._device)
            with torch.no_grad():
                t_norm = self._ensemble(x, is_upper_t).item()
            return t_norm * (self._label_t_max - self._label_t_min) + self._label_t_min
        except Exception:
            return None


class DualEyePredictor:
    """デュアルブランチ EyelinerCNN アンサンブルの推論器。

    experiment.json から fusion_mode / rescale_b を読み取り、
    枝A（生PQ）・枝B（P'Q'）のパッチを抽出して生 t を返す。
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch

        model_dir = Path(model_dir)
        exp_json = model_dir / "experiment.json"
        if not exp_json.exists():
            raise FileNotFoundError(f"experiment.json が見つかりません: {exp_json}")
        with open(exp_json, encoding="utf-8") as f:
            hp = json.load(f).get("hyperparameters", {})
        if "fusion_mode" not in hp:
            raise ValueError(f"デュアルブランチ実験ではありません: {model_dir}")

        self._fusion_mode = hp["fusion_mode"]
        self._rescale_b = hp.get("rescale_b", "eye_scale")
        self._device = device

        from eyeliner.models.dual_ensemble import load_dual_fold_models
        from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t as _fer

        self._cfg_a, self._cfg_b = build_branch_configs(self._rescale_b)
        self._fold_ensemble_raw_t = _fer
        models = load_dual_fold_models(model_dir, self._fusion_mode)
        self._models = [m.to(device).eval() for m in models]
        self.loaded_count = len(self._models)

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1).to(device)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1).to(device)

    def predict_t_raw(
        self,
        image_rgb: np.ndarray,
        px: float, py: float,
        qx: float, qy: float,
        is_upper: float,
        eye_scale: Optional[float] = None,
    ) -> Optional[float]:
        """デュアルブランチで生 t を予測する。eye_scale は枝B の P'Q' 計算に使用。"""
        import torch
        import math as _math

        try:
            orig_pq = _math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

            patch_a = extract_patch(
                image_rgb, px, py, qx, qy,
                patch_size=self._cfg_a.patch_size,
                t_min=self._cfg_a.patch_t_min,
                t_max=self._cfg_a.patch_t_max,
                perp_scale=self._cfg_a.patch_perp_scale,
            )
            x_a = torch.from_numpy(patch_a.transpose(2, 0, 1)).float().unsqueeze(0).to(self._device)
            x_a = (x_a - self._mean) / self._std

            # pq_prime: eye_scale * factor（未指定時は原 PQ 距離で代用）
            if eye_scale is not None and eye_scale > 1e-6:
                pq_prime_val = eye_scale * self._cfg_b.pq_rescale_factor
            else:
                pq_prime_val = orig_pq

            if orig_pq > 1e-6:
                ox = (px + qx) / 2.0
                oy = (py + qy) / 2.0
                ux = (qx - px) / orig_pq
                uy = (qy - py) / orig_pq
                px_b = ox - (pq_prime_val / 2.0) * ux
                py_b = oy - (pq_prime_val / 2.0) * uy
                qx_b = ox + (pq_prime_val / 2.0) * ux
                qy_b = oy + (pq_prime_val / 2.0) * uy
            else:
                px_b, py_b, qx_b, qy_b = px, py, qx, qy

            patch_b = extract_patch(
                image_rgb, px_b, py_b, qx_b, qy_b,
                patch_size=self._cfg_b.patch_size,
                t_min=self._cfg_b.patch_t_min,
                t_max=self._cfg_b.patch_t_max,
                perp_scale=self._cfg_b.patch_perp_scale,
            )
            x_b = torch.from_numpy(patch_b.transpose(2, 0, 1)).float().unsqueeze(0).to(self._device)
            x_b = (x_b - self._mean) / self._std

            is_upper_t = torch.tensor([[is_upper]], dtype=torch.float32, device=self._device)
            pq_dist_t  = torch.tensor([[orig_pq]],     dtype=torch.float32, device=self._device)
            pq_prime_t = torch.tensor([[pq_prime_val]], dtype=torch.float32, device=self._device)

            with torch.no_grad():
                t_raw = self._fold_ensemble_raw_t(
                    self._models, x_a, x_b, is_upper_t,
                    self._fusion_mode, self._cfg_a, self._cfg_b,
                    pq_dist_t, pq_prime_t,
                )
            return float(t_raw.item())
        except Exception:
            return None


def make_predictor(model_dir: Path, device: str = "cpu"):
    """experiment.json を参照してシングル/デュアルブランチを自動判別し推論器を返す。"""
    exp_json = Path(model_dir) / "experiment.json"
    if exp_json.exists():
        with open(exp_json, encoding="utf-8") as f:
            hp = json.load(f).get("hyperparameters", {})
        if "fusion_mode" in hp:
            return DualEyePredictor(model_dir, device)
    return EyePredictor(model_dir, device)


# ---------------------------------------------------------------------------
# レンダリング
# ---------------------------------------------------------------------------

def load_eye_region_rgb(
    image_path: str,
    eye_samples: List[Dict[str, Any]],
    margin_factor: float = 1.5,
) -> Tuple[np.ndarray, float, float]:
    """眼周辺の小領域だけをロードして (crop, ox, oy) を返す。

    フル画像をロード後に即 P/Q bbox 周辺を切り出してフル画像を解放するため、
    大解像度画像でのメモリ消費を最小化する。
    ox, oy は crop の左上隅の元画像座標。
    """
    import gc

    pxs = [float(s["px"]) for s in eye_samples] + [float(s["qx"]) for s in eye_samples]
    pys = [float(s["py"]) for s in eye_samples] + [float(s["qy"]) for s in eye_samples]
    span = max(max(pxs) - min(pxs), max(pys) - min(pys), 1.0)
    margin = span * margin_factor

    full = cv2.imread(image_path)
    if full is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    h, w = full.shape[:2]
    cx1 = max(0, int(min(pxs) - margin))
    cy1 = max(0, int(min(pys) - margin))
    cx2 = min(w, int(max(pxs) + margin))
    cy2 = min(h, int(max(pys) + margin))

    # BGR→RGB かつ copy() でフル画像との共有を断ち、フル画像を即解放
    crop_bgr = full[cy1:cy2, cx1:cx2].copy()
    del full
    gc.collect()

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    del crop_bgr
    return crop_rgb, float(cx1), float(cy1)


def render_eye_crop_b64(
    image_rgb: np.ndarray,
    eye_samples: List[Dict[str, Any]],
    predictor: EyePredictor,
    image_offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[str]:
    """
    CNN 予測に基づく Catmull-Rom 曲線付きアイクロップを base64 PNG で返す。

    image_rgb: 眼周辺の小クロップ（または元画像全体）。
    eye_samples: 元画像座標系の P/Q 点 (px, py, qx, qy)。
    image_offset: image_rgb の左上隅の元画像座標 (ox, oy)。
      load_eye_region_rgb() と組み合わせてフル画像のメモリ保持を回避する。
    R 点が1点も得られない場合は None。
    """
    if not eye_samples:
        return None

    ox, oy = image_offset

    points_by_sub: Dict[str, List[Tuple[float, float]]] = {"upper": [], "lower": []}

    for s in sorted(eye_samples, key=lambda x: x["point_index"]):
        # CNN 推論は image_rgb の座標系（ox,oy オフセット済み）で行う
        px, py = float(s["px"]) - ox, float(s["py"]) - oy
        qx, qy = float(s["qx"]) - ox, float(s["qy"]) - oy
        sub = str(s.get("sub_key", "upper"))
        is_upper = 1.0 if sub == "upper" else 0.0

        # eye_scale はデュアルブランチ枝B の P'Q' 計算に必須。
        # スキャン (t_pred_scan) と同じ入力を与え、プレビュー曲線が
        # テーブルの min/max_t_pred と整合するようにする。
        es = float(s["eye_scale"]) if s.get("eye_scale") is not None else None

        t_raw = predictor.predict_t_raw(image_rgb, px, py, qx, qy, is_upper, es)
        if t_raw is None:
            continue

        # R 点は元画像座標系で保持
        rx, ry = reconstruct_point(px + ox, py + oy, qx + ox, qy + oy, t_raw)
        bucket = sub if sub in points_by_sub else "upper"
        points_by_sub[bucket].append((rx, ry))

    all_r = points_by_sub["upper"] + points_by_sub["lower"]
    if not all_r:
        return None

    # 元画像座標系でバウンディングボックスを計算
    all_bbox = (
        all_r
        + [(float(s["px"]), float(s["py"])) for s in eye_samples]
        + [(float(s["qx"]), float(s["qy"])) for s in eye_samples]
    )
    xs = [p[0] for p in all_bbox]
    ys = [p[1] for p in all_bbox]

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = max(span * 0.1, 10.0)

    # image_rgb の範囲内に収める（image_rgb は小クロップ）
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, int(min(xs) - margin - ox))
    y1 = max(0, int(min(ys) - margin - oy))
    x2 = min(w_img, int(max(xs) + margin - ox))
    y2 = min(h_img, int(max(ys) + margin - oy))
    if x1 >= x2 or y1 >= y2:
        return None

    crop = image_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    scale = max(5, min(8, 1200 // max(crop_h, crop_w, 5)))
    if scale > 1:
        crop = cv2.resize(
            crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_LANCZOS4
        )

    # 表示座標: 元画像座標 → image_rgb座標 → crop座標 → scale
    def to_crop(rx: float, ry: float) -> Tuple[float, float]:
        return ((rx - ox - x1) * scale, (ry - oy - y1) * scale)

    eye_scale = float(eye_samples[0].get("eye_scale") or 60.0)
    es_factor = max(1.0, eye_scale / 60.0)
    dot_r = max(4, min(20, round(7 * es_factor)))
    line_t = max(1, min(5, round(es_factor)))

    upper_pts_f = [to_crop(rx, ry) for rx, ry in points_by_sub["upper"]]
    chain_u = _catmull_rom_chain(upper_pts_f)
    _draw_solid_curve(crop, chain_u, _BLUE_RGB, line_t)
    for pt in upper_pts_f:
        cv2.circle(crop, (int(round(pt[0])), int(round(pt[1]))), dot_r, _BLUE_RGB, -1)

    lower_pts_f = [to_crop(rx, ry) for rx, ry in points_by_sub["lower"]]
    chain_l = _catmull_rom_chain(lower_pts_f)
    _draw_dashed_curve(crop, chain_l, _BLUE_RGB, line_t)
    for pt in lower_pts_f:
        cv2.circle(crop, (int(round(pt[0])), int(round(pt[1]))), dot_r, _BLUE_RGB, -1)

    return _to_base64_png(crop)


def _to_base64_png(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()

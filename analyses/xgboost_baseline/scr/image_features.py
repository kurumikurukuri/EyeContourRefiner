"""画像から目領域の局所特徴量を抽出するモジュール。

カテゴリ A: 輝度ヒストグラム統計量 (mean, std, skewness, entropy, p10, p90)
カテゴリ B: ボケ・エッジ鮮鋭度 (Laplacian var, Sobel mean/std, Canny density)
カテゴリ C: テクスチャ (HOG mean/std, LBP uniform ratio, LBP entropy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

EYE_CROP_MARGIN = 12
HOG_WIN_SIZE = (64, 32)  # (width, height) — canonical eye crop size for HOG


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def load_image_gray(image_path: str | Path) -> np.ndarray | None:
    path = Path(image_path)
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Eye crop
# ---------------------------------------------------------------------------
def _compute_eye_bbox(
    landmark_group: pd.DataFrame,
) -> tuple[int, int, int, int]:
    xs = np.concatenate(
        [landmark_group["px"].values, landmark_group["qx"].values]
    )
    ys = np.concatenate(
        [landmark_group["py"].values, landmark_group["qy"].values]
    )
    x_min = int(np.floor(np.nanmin(xs) - EYE_CROP_MARGIN))
    y_min = int(np.floor(np.nanmin(ys) - EYE_CROP_MARGIN))
    x_max = int(np.ceil(np.nanmax(xs) + EYE_CROP_MARGIN))
    y_max = int(np.ceil(np.nanmax(ys) + EYE_CROP_MARGIN))
    return x_min, y_min, x_max, y_max


def _crop_region(
    image: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray | None:
    h, w = image.shape
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)
    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]


# ---------------------------------------------------------------------------
# A: Histogram features
# ---------------------------------------------------------------------------
def _histogram_entropy(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0, 255))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _histogram_features(crop: np.ndarray) -> dict:
    values = crop.astype(float).ravel()
    return {
        "eye_gray_mean": float(np.mean(values)),
        "eye_gray_std": float(np.std(values)),
        "eye_gray_skewness": float(pd.Series(values).skew()),
        "eye_gray_entropy": _histogram_entropy(values),
        "eye_gray_p10": float(np.percentile(values, 10)),
        "eye_gray_p90": float(np.percentile(values, 90)),
    }


# ---------------------------------------------------------------------------
# B: Blur / edge sharpness
# ---------------------------------------------------------------------------
def _blur_features(crop: np.ndarray) -> dict:
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    lap_var = float(lap.var())

    sobel_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    canny = cv2.Canny(crop, 50, 150)
    canny_density = float(np.count_nonzero(canny) / canny.size)

    return {
        "eye_laplacian_var": lap_var,
        "eye_sobel_mean": float(sobel_mag.mean()),
        "eye_sobel_std": float(sobel_mag.std()),
        "eye_canny_density": canny_density,
    }


# ---------------------------------------------------------------------------
# C: Texture (HOG + LBP)
# ---------------------------------------------------------------------------
def _hog_features(crop: np.ndarray) -> dict:
    resized = cv2.resize(crop, HOG_WIN_SIZE, interpolation=cv2.INTER_LINEAR)
    hog = cv2.HOGDescriptor(
        _winSize=HOG_WIN_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    desc = hog.compute(resized).ravel()
    if len(desc) == 0:
        return {"eye_hog_mean": 0.0, "eye_hog_std": 0.0}
    return {
        "eye_hog_mean": float(np.mean(desc)),
        "eye_hog_std": float(np.std(desc)),
    }


def _lbp_features(crop: np.ndarray) -> dict:
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(crop, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins - 1))
    hist = hist.astype(float)
    hist /= hist.sum()
    uniform_ratio = float(hist[:-1].sum()) if len(hist) > 1 else 0.0
    hist_nz = hist[hist > 0]
    lbp_entropy = (
        float(-np.sum(hist_nz * np.log2(hist_nz))) if len(hist_nz) > 0 else 0.0
    )
    return {
        "eye_lbp_uniform_ratio": uniform_ratio,
        "eye_lbp_entropy": lbp_entropy,
    }


# ---------------------------------------------------------------------------
# Per-eye extraction
# ---------------------------------------------------------------------------
def extract_eye_features(
    image: np.ndarray,
    landmark_group: pd.DataFrame,
) -> dict:
    x_min, y_min, x_max, y_max = _compute_eye_bbox(landmark_group)
    crop = _crop_region(image, x_min, y_min, x_max, y_max)
    if crop is None or crop.size < 16:
        return {}
    feats: dict = {}
    feats.update(_histogram_features(crop))
    feats.update(_blur_features(crop))
    feats.update(_hog_features(crop))
    feats.update(_lbp_features(crop))
    return feats


# ---------------------------------------------------------------------------
# Batch build (with cache)
# ---------------------------------------------------------------------------
def build_image_features(
    manifest: pd.DataFrame,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """マニフェストから画像・目単位の局所特徴量を計算する。

    結果は (image_path, face_index, part_name) をキーとする DataFrame。
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        logger.info("loading image features from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    eye_key = ["image_path", "face_index", "part_name"]
    groups = manifest.groupby(eye_key)

    rows: list[dict] = []
    n_total = groups.ngroups
    n_processed = 0
    n_failed = 0

    for (img_path, face_idx, part), grp in groups:
        img = load_image_gray(img_path)
        if img is None:
            n_failed += 1
            continue

        feats = extract_eye_features(img, grp)
        if not feats:
            n_failed += 1
            continue

        feats["image_path"] = img_path
        feats["face_index"] = face_idx
        feats["part_name"] = part
        rows.append(feats)
        n_processed += 1

        if n_processed % 1000 == 0:
            logger.info(
                "image features: %d/%d (%d failed)",
                n_processed,
                n_total,
                n_failed,
            )

    logger.info(
        "image features: %d processed, %d failed, %d total groups",
        n_processed,
        n_failed,
        n_total,
    )

    df = pd.DataFrame(rows)

    if cache_path is not None and len(df) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("image features cached to %s", cache_path)

    return df

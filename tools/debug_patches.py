"""デバッグ用：CNNへの入力パッチと元画像クロップを数件保存するスクリプト。

使い方:
    python tools/debug_patches.py --manifest experiments/EXP014_xxx/manifests/manifest_train.csv

    # 件数・出力先を指定
    python tools/debug_patches.py \\
        --manifest experiments/EXP014_xxx/manifests/manifest_train.csv \\
        --n 8 \\
        --output_dir debug_patches

    # パッチオプションを合わせる（実験設定と揃える場合）
    python tools/debug_patches.py \\
        --manifest experiments/EXP014_xxx/manifests/manifest_train.csv \\
        --patch_t_min -0.5 --patch_t_max 1.5 \\
        --patch_h 64 --patch_w 128 \\
        --norm_type imagenet

出力:
    {output_dir}/
        000_crop.png        ... extract_patch() の出力（正規化前、P/Q マーカー付き）
        000_cnn_input.png   ... 正規化後をデノーマライズした画像（CNN への入力を可視化）
        000_region.png      ... 元画像上にクロップ領域（四角形）・P・Q を描画した画像
        001_crop.png
        ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eyeliner.data.patch_utils import (
    NORM_TYPE_CHOICES,
    T_MIN, T_MAX,
    PATCH_SIZE,
    extract_patch,
    get_norm_stats,
)
from eyeliner.data.dataset import load_manifest

logger = logging.getLogger(__name__)


def _build_affine_matrix(
    px: float, py: float, qx: float, qy: float,
    patch_h: int, patch_w: int,
    t_min: float, t_max: float,
    perp_scale: float,
) -> np.ndarray:
    """extract_patch() と完全一致するアフィン行列"""
    dx, dy = qx - px, qy - py
    norm = float(np.hypot(dx, dy))
    if norm == 0.0:
        return None

    ux, uy = dx / norm, dy / norm

    scale = (patch_w - 1) / ((t_max - t_min) * norm)

    # t = t_max の点（patch左端）
    sx = t_max * px + (1.0 - t_max) * qx
    sy = t_max * py + (1.0 - t_max) * qy

    actual_perp = (patch_h - 1) / (perp_scale * norm)

    M = np.float32([
        [scale * ux,        scale * uy,        -scale * (ux * sx + uy * sy)],
        [-uy * actual_perp, ux * actual_perp,   actual_perp * (uy * sx - ux * sy) + patch_h // 2],
    ])
    return M


def _draw_region_on_image(
    image_rgb: np.ndarray,
    px: float, py: float,
    qx: float, qy: float,
    patch_h: int, patch_w: int,
    t_min: float, t_max: float,
    perp_scale: float,
) -> np.ndarray:
    out = image_rgb.copy()

    M = _build_affine_matrix(
        px, py, qx, qy,
        patch_h, patch_w,
        t_min, t_max,
        perp_scale,
    )

    if M is not None:
        # --- パッチ空間でマスク作成 ---
        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (patch_w - 1, patch_h - 1), 255, -1)

        # --- 画像サイズにwarp（forwardと一致） ---
        mask_img = cv2.warpAffine(
            mask,
            cv2.invertAffineTransform(M),  # patch→image
            (image_rgb.shape[1], image_rgb.shape[0]),
            flags=cv2.INTER_NEAREST,
        )

        # --- 輪郭抽出 ---
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 描画（黄色） ---
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    # --- P / Q ---
    cv2.circle(out, (int(px), int(py)), 6, (0, 0, 255), -1)
    cv2.putText(out, "P", (int(px) + 6, int(py) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.circle(out, (int(qx), int(qy)), 6, (0, 255, 0), -1)
    cv2.putText(out, "Q", (int(qx) + 6, int(qy) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # PQ方向
    cv2.arrowedLine(out, (int(px), int(py)), (int(qx), int(qy),), (255, 255, 0), 2)

    return out


def _draw_pq_markers(patch_u8: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """パッチに t=1(P) と t=0(Q) の位置を描画"""
    out = patch_u8.copy()
    h, w = out.shape[:2]

    # P (t=1)
    p_x = int(round((t_max - 1.0) / (t_max - t_min) * (w - 1)))
    cv2.line(out, (p_x, 0), (p_x, h - 1), (0, 0, 255), 1)

    # Q (t=0)
    q_x = int(round((t_max - 0.0) / (t_max - t_min) * (w - 1)))
    cv2.line(out, (q_x, 0), (q_x, h - 1), (0, 255, 0), 1)

    return out


def save_debug_patches(
    manifest_path: Path,
    n: int,
    output_dir: Path,
    patch_t_min: float,
    patch_t_max: float,
    patch_perp_scale: float,
    patch_h: int,
    patch_w: int,
    norm_type: str,
    norm_stats_json: Path | None,
) -> None:
    records = load_manifest(manifest_path)
    if not records:
        logger.error("manifest にサンプルが見つかりません: %s", manifest_path)
        sys.exit(1)

    records = records[:n]
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_mean, norm_std = get_norm_stats(norm_type, stats_json_path=norm_stats_json)
    mean_arr = np.array(norm_mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(norm_std, dtype=np.float32).reshape(1, 1, 3)

    patch_size = (patch_h, patch_w)
    image_cache: dict[str, np.ndarray] = {}

    for idx, rec in enumerate(records):
        image_path = rec["image_path"]
        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])
        t_actual = float(rec["t"])

        if image_path not in image_cache:
            bgr = cv2.imread(image_path)
            if bgr is None:
                logger.warning("[%03d] 読み込み失敗: %s", idx, image_path)
                continue
            image_cache[image_path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_rgb = image_cache[image_path]

        # --- crop ---
        raw_patch = extract_patch(
            image_rgb, px, py, qx, qy,
            patch_size=patch_size,
            t_min=patch_t_min,
            t_max=patch_t_max,
            perp_scale=patch_perp_scale,
        )

        crop_u8 = (raw_patch * 255).clip(0, 255).astype(np.uint8)
        crop_u8 = _draw_pq_markers(crop_u8, patch_t_min, patch_t_max)

        cv2.imwrite(
            str(output_dir / f"{idx:03d}_crop.png"),
            cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR),
        )

        # --- CNN input (確認用) ---
        normalized = (raw_patch - mean_arr) / std_arr
        denorm = normalized * std_arr + mean_arr

        cnn_u8 = (denorm * 255).clip(0, 255).astype(np.uint8)
        cnn_u8 = _draw_pq_markers(cnn_u8, patch_t_min, patch_t_max)

        cv2.imwrite(
            str(output_dir / f"{idx:03d}_cnn_input.png"),
            cv2.cvtColor(cnn_u8, cv2.COLOR_RGB2BGR),
        )

        # --- region ---
        region_img = _draw_region_on_image(
            image_rgb, px, py, qx, qy,
            patch_h, patch_w,
            patch_t_min, patch_t_max,
            patch_perp_scale,
        )

        cv2.imwrite(
            str(output_dir / f"{idx:03d}_region.png"),
            cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR),
        )

        logger.info("[%03d] 保存完了 t=%.3f img=%s",
                    idx, t_actual, Path(image_path).name)

    logger.info("完了: %d 件を %s に保存", len(records), output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("debug_patches"))
    parser.add_argument("--patch_t_min", type=float, default=T_MIN)
    parser.add_argument("--patch_t_max", type=float, default=T_MAX)
    parser.add_argument("--patch_perp_scale", type=float, default=1.0)
    parser.add_argument("--patch_h", type=int, default=PATCH_SIZE[0])
    parser.add_argument("--patch_w", type=int, default=PATCH_SIZE[1])
    parser.add_argument("--norm_type", choices=NORM_TYPE_CHOICES, default="imagenet")
    parser.add_argument("--norm_stats_json", type=Path, default=None)

    args = parser.parse_args()

    save_debug_patches(
        manifest_path=args.manifest,
        n=args.n,
        output_dir=args.output_dir,
        patch_t_min=args.patch_t_min,
        patch_t_max=args.patch_t_max,
        patch_perp_scale=args.patch_perp_scale,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        norm_type=args.norm_type,
        norm_stats_json=args.norm_stats_json,
    )


if __name__ == "__main__":
    main()
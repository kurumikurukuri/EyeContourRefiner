"""eye_contour_refiner.onnx の自己完結型リファレンス実装。

依存: numpy, opencv-python, onnxruntime のみ（このリポジトリの eye_contour/ パッケージから
パッチ抽出・座標変換の関数を再利用するが、torch はエクスポート済み.onnxを使う限り不要）。

使い方:
    # 動作確認（ダミー画像・ダミー座標、実画像不要）
    python models/infer_example.py --dry-run

    # 実画像で1点推論
    python models/infer_example.py \\
        --image path/to/face.jpg \\
        --px 120.5 --py 88.0 --qx 95.2 --qy 90.1 --is-upper \\
        --eye-points "118,80 122,79 ... "  # 目1つ分の14点(px,py qx,qy)をスペース区切りで

入出力の詳細な契約は README.md を参照。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eye_contour.data.patch_utils import (  # noqa: E402
    compute_eye_scale,
    compute_pq_rescale,
    extract_patch,
    reconstruct_point,
)
from eye_contour.training.dual_branch import branch_a_x0, branch_b_eye_scale  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).parent / "eye_contour_refiner.onnx"


def predict_point(
    session: ort.InferenceSession,
    image_rgb: np.ndarray,
    px: float, py: float, qx: float, qy: float,
    is_upper: bool,
    eye_scale: float,
) -> tuple[float, float, float]:
    """1点分の推論を行い (t, rx, ry) を返す。

    image_rgb: RGB画像 (H, W, 3) uint8。
    px, py, qx, qy: ランドマーク点P, Qの元の画像ピクセル座標（リスケール前）。
    eye_scale: 対象の目の14点から計算した eye_scale（compute_eye_scale参照）。
    """
    cfg_a = branch_a_x0()
    cfg_b = branch_b_eye_scale()

    # branch A: 生PQ座標系のままパッチ抽出
    patch_a = extract_patch(
        image_rgb, px, py, qx, qy,
        patch_size=cfg_a.patch_size, t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max,
        perp_scale=cfg_a.patch_perp_scale,
    )
    # branch B: eye_scaleでリスケールしたP'Q'座標系でパッチ抽出
    rec = {"eye_scale": eye_scale}
    pb_px, pb_py, pb_qx, pb_qy, _, _ = compute_pq_rescale(
        px, py, qx, qy, 0.0, rec, cfg_b.pq_rescale_mode, cfg_b.pq_rescale_factor,
    )
    patch_b = extract_patch(
        image_rgb, pb_px, pb_py, pb_qx, pb_qy,
        patch_size=cfg_b.patch_size, t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max,
        perp_scale=cfg_b.patch_perp_scale,
    )

    x_a = patch_a.transpose(2, 0, 1)[None].astype(np.float32)   # (1,3,H,W)
    x_b = patch_b.transpose(2, 0, 1)[None].astype(np.float32)
    feed = {
        "patch_a": x_a, "patch_b": x_b,
        "is_upper": np.array([[1.0 if is_upper else 0.0]], dtype=np.float32),
        "px": np.array([[px]], dtype=np.float32), "py": np.array([[py]], dtype=np.float32),
        "qx": np.array([[qx]], dtype=np.float32), "qy": np.array([[qy]], dtype=np.float32),
        "eye_scale": np.array([[eye_scale]], dtype=np.float32),
    }
    t = float(session.run(None, feed)[0].reshape(-1)[0])
    rx, ry = reconstruct_point(px, py, qx, qy, t)
    return t, rx, ry


def _parse_points(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        x_str, y_str = tok.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="eye_contour_refiner.onnx 推論サンプル")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--dry-run", action="store_true", help="ダミー画像・ダミー座標で動作確認のみ行う")
    ap.add_argument("--image", type=Path, help="入力画像")
    ap.add_argument("--px", type=float)
    ap.add_argument("--py", type=float)
    ap.add_argument("--qx", type=float)
    ap.add_argument("--qy", type=float)
    ap.add_argument("--is-upper", action="store_true")
    ap.add_argument("--eye-points", type=str,
                    help='対象の目の全14点(P,Q x 7)を"px,py px,py ..."形式で。'
                         "eye_scale計算に使う。省略時は--px..--qyから概算")
    args = ap.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    if args.dry_run:
        rng = np.random.RandomState(0)
        image = rng.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        px, py, qx, qy = 300.0, 200.0, 260.0, 205.0
        eye_scale = compute_eye_scale([(px, py), (qx, qy)])  # ダミーなので2点のみ
        t, rx, ry = predict_point(session, image, px, py, qx, qy, is_upper=True, eye_scale=eye_scale)
        print(f"[dry-run] t={t:.4f}  R=({rx:.1f}, {ry:.1f})  (eye_scale={eye_scale:.1f} はダミー、"
              "実運用では対象の目の14点全部から計算すること)")
        return 0

    if args.image is None or None in (args.px, args.py, args.qx, args.qy):
        ap.error("--dry-run を指定しない場合は --image --px --py --qx --qy が必須です")

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.eye_points:
        eye_scale = compute_eye_scale(_parse_points(args.eye_points))
    else:
        print("[warn] --eye-points 未指定のためP,Qの2点だけからeye_scaleを概算します"
              "（実運用では目1つ分の14点全部を使うこと）", file=sys.stderr)
        eye_scale = compute_eye_scale([(args.px, args.py), (args.qx, args.qy)])

    t, rx, ry = predict_point(
        session, image, args.px, args.py, args.qx, args.qy, args.is_upper, eye_scale,
    )
    print(f"t={t:.4f}  R=({rx:.1f}, {ry:.1f})  eye_scale={eye_scale:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

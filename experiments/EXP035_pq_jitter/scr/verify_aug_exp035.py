"""EXP035 PQ 回転摂動 augmentation の実装検証（学習済みモデルは使わない）。

実際の EyelinerPatchDualDataset.__getitem__ 経路で次を確認する。

  1. 合成画像: R の位置に赤点を描き、θ を振っても赤点がラベル t の示すパッチ位置
     （枝A: t、枝B: t'、いずれも中央行）に留まるか = 画像とラベルの整合
  2. 実データ: ラベル y_a / y_b / |PQ| / |P'Q'| が摂動で不変か、θ の分布、
     パッチの変化が R から離れるほど大きいか（R 中心回転になっているか）、
     サンプル領域が画像外へはみ出す割合
  3. 実データの可視化: θ = -10, -5, 0, +5, +10° のパッチを枝A・枝B で並べ、R の期待位置に印

使い方:
    python experiments/EXP035_pq_jitter/scr/verify_aug_exp035.py
    python experiments/EXP035_pq_jitter/scr/verify_aug_exp035.py --n-samples 500 --rot-deg 10
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
sys.path.insert(0, str(ROOT))

from eyeliner.data.augmentation import AUG_PQ_JITTER  # noqa: E402
from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.patch_utils import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    reconstruct_point,
)
from eyeliner.data.source_degradation import compute_patch_source_bbox  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.training.dual_branch import build_branch_configs  # noqa: E402

THETAS_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)


class FixedAngleRng:
    """augment_pq_jitter に固定 θ を返す RNG（uniform のみ使われる、prob=1.0 前提）。"""

    def __init__(self, theta_deg: float) -> None:
        self.theta_deg = theta_deg

    def uniform(self, _a: float, _b: float) -> float:
        return self.theta_deg

    def random(self) -> float:
        return 0.0


class RecordingRng(random.Random):
    """実際の θ 系列を記録する RNG。"""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.thetas: list[float] = []

    def uniform(self, a: float, b: float) -> float:
        v = super().uniform(a, b)
        self.thetas.append(v)
        return v


def make_dataset(records: list[dict], jitter: bool, rot_deg: float) -> EyelinerPatchDualDataset:
    """PQ 回転摂動のみ有効（他 aug は全 OFF）の dataset。jitter=False は評価時と同一。"""
    cfg_a, cfg_b = build_branch_configs("eye_scale")
    return EyelinerPatchDualDataset(
        records, cfg_a, cfg_b,
        augment_data=jitter,
        aug_flags=AUG_PQ_JITTER if jitter else 0,
        pq_jitter_rot_deg=rot_deg if jitter else 0.0,
        pq_jitter_prob=1.0,
        preload_images=False,
    )


def to_image(x) -> np.ndarray:
    """正規化済み (3,H,W) tensor → RGB [0,1] (H,W,3)。"""
    arr = x.numpy().transpose(1, 2, 0)
    return np.clip(arr * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0.0, 1.0)


def expected_patch_xy(t: float, t_min: float, t_max: float, size: tuple[int, int]) -> tuple[float, float]:
    """extract_patch の仕様（docstring）から、t の点が来るパッチ座標を求める。"""
    h, w = size
    return (t_max - t) / (t_max - t_min) * (w - 1), float(h // 2)


def label_to_t(y: float, cfg) -> float:
    return y * (cfg.label_t_max - cfg.label_t_min) + cfg.label_t_min


# ---------------------------------------------------------------------------
# 1. 合成画像
# ---------------------------------------------------------------------------

def synthetic_record(t: float) -> tuple[dict, np.ndarray]:
    px, py, qx, qy = 220.0, 200.0, 180.0, 206.0
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    rx, ry = reconstruct_point(px, py, qx, qy, t)
    # 疑似まぶたの縁（緑線）を R を通る斜め線として描き、R に赤点を置く
    cv2.line(img, (int(rx - 80), int(ry + 25)), (int(rx + 80), int(ry - 25)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (int(round(rx)), int(round(ry))), 4, (255, 0, 0), -1, cv2.LINE_AA)
    rec = {
        "image_path": "synthetic.png", "px": px, "py": py, "qx": qx, "qy": qy, "t": t,
        "sub_key": "upper", "point_index": 3, "eye_scale": 150.0,
        "pq_ratio": math.hypot(qx - px, qy - py) / 150.0, "weight": 1.0, "val_weight": 1.0,
    }
    return rec, img


def red_centroid(patch: np.ndarray) -> tuple[float, float]:
    redness = np.clip(patch[..., 0] - patch[..., 1:].max(axis=-1), 0.0, None)
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    total = redness.sum()
    if total <= 0:
        return float("nan"), float("nan")
    return float((xs * redness).sum() / total), float((ys * redness).sum() / total)


def check_synthetic(out_dir: Path) -> bool:
    print("\n[1] 合成画像: R の赤点がラベル位置に留まるか（許容誤差 0.5 patch px）")
    ok = True
    tiles = []
    for t in (0.1, 0.4, 0.9):
        rec, img = synthetic_record(t)
        row = []
        for theta in THETAS_DEG:
            ds = make_dataset([rec], jitter=True, rot_deg=10.0)
            ds._image_cache["synthetic.png"] = img
            ds._img_diag_map["synthetic.png"] = 400.0
            ds._pq_jitter_rng = FixedAngleRng(theta)
            out = ds[0]
            for name, x, y, cfg in (("A", out[0], out[4], ds.cfg_a), ("B", out[1], out[5], ds.cfg_b)):
                patch = to_image(x)
                t_label = label_to_t(float(y), cfg)
                ex, ey = expected_patch_xy(t_label, cfg.patch_t_min, cfg.patch_t_max, cfg.patch_size)
                cx, cy = red_centroid(patch)
                err = math.hypot(cx - ex, cy - ey)
                ok &= err < 0.5
                print(f"  t={t:.1f} θ={theta:+5.1f}° 枝{name}: 期待=({ex:5.1f},{ey:4.1f}) "
                      f"赤点=({cx:5.1f},{cy:4.1f}) 誤差={err:.2f}px")
                row.append((patch, ex, ey))
        tiles.append(row)
    save_grid(tiles, out_dir / "aug_check_synthetic.png")
    print(f"  → {'OK' if ok else 'NG'}")
    return ok


# ---------------------------------------------------------------------------
# 2. 実データ数値検証
# ---------------------------------------------------------------------------

def load_real_records(n: int, seed: int) -> list[dict]:
    records = load_manifest(MANIFEST_DIR / "manifest_train.csv")
    compute_eye_meta(records)  # eye_scale は眼単位の全点から求めるため、抽出前に全件で計算
    rng = random.Random(seed)
    return [dict(r) for r in rng.sample(records, min(n, len(records)))]


def footprint_outside(rec: dict, px, py, qx, qy, cfg, img_shape) -> bool:
    bbox = compute_patch_source_bbox(px, py, qx, qy, cfg.patch_t_min, cfg.patch_t_max,
                                     cfg.patch_perp_scale, margin_px=0)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    h, w = img_shape[:2]
    return x0 < 0 or y0 < 0 or x1 > w or y1 > h


def radial_diff_profile(diffs: list[np.ndarray], r_cols: list[float], cfg) -> list[tuple[str, float]]:
    """枝A パッチの |Δ| を、R からの画像空間距離（|PQ| 単位）で集計する。

    パッチは沿線方向と垂直方向でスケールが違うため、画素距離ではなく |PQ| 単位に戻す。
    R 中心の回転なら画素の移動量 ≈ θ·距離 なので、|Δ| は R 近傍ほど小さくなるはず。
    """
    h, w = cfg.patch_size
    along = (w - 1) / (cfg.patch_t_max - cfg.patch_t_min)   # patch px / |PQ|
    perp = (h - 1) / cfg.patch_perp_scale                    # patch px / |PQ|
    bins = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
    rows, cols = np.mgrid[0:h, 0:w]
    acc = {b: [] for b in bins}
    for d, rc in zip(diffs, r_cols):
        dist = np.hypot((cols - rc) / along, (rows - h // 2) / perp)
        pix = d.mean(axis=2)
        for b in bins:
            m = (dist >= b[0]) & (dist < b[1])
            if m.any():
                acc[b].append(float(pix[m].mean()))
    return [(f"{b[0]}-{b[1]}|PQ|", float(np.mean(v)) if v else float("nan")) for b, v in acc.items()]


def check_real(records: list[dict], rot_deg: float, seed: int) -> bool:
    print(f"\n[2] 実データ（train manifest から {len(records)} 件、θmax=±{rot_deg}°）")
    clean_ds = make_dataset(records, jitter=False, rot_deg=0.0)
    jit_ds = make_dataset(records, jitter=True, rot_deg=rot_deg)
    rng = RecordingRng(seed)
    jit_ds._pq_jitter_rng = rng

    label_ok = True
    diffs, r_cols = [], []
    for i in range(len(records)):
        c, j = clean_ds[i], jit_ds[i]
        for k in (4, 5, 12, 13):  # y_a, y_b, |PQ|, |P'Q'|
            label_ok &= bool(abs(float(c[k]) - float(j[k])) < 1e-4)
        diffs.append(np.abs(to_image(j[0]) - to_image(c[0])))
        t_a = label_to_t(float(c[4]), clean_ds.cfg_a)
        r_cols.append(expected_patch_xy(t_a, clean_ds.cfg_a.patch_t_min, clean_ds.cfg_a.patch_t_max,
                                        clean_ds.cfg_a.patch_size)[0])

    th = np.asarray(rng.thetas)
    print(f"  ラベル y_a / y_b / |PQ| / |P'Q'| 不変: {'OK' if label_ok else 'NG'}")
    print(f"  θ: n={len(th)}（=サンプル数 {len(records)}）  min={th.min():+.2f}  max={th.max():+.2f}  "
          f"mean={th.mean():+.2f}  std={th.std():.2f}（一様分布の理論値 {rot_deg / math.sqrt(3):.2f}）")
    print("  枝A パッチの |Δ|（R からの距離別、R 中心回転なら R 近傍ほど小さい）:")
    for label, v in radial_diff_profile(diffs, r_cols, clean_ds.cfg_a):
        print(f"    {label:>8}: {v:.4f}")
    report_outside(records, rot_deg, jit_ds)
    theta_ok = len(th) == len(records) and th.min() >= -rot_deg and th.max() <= rot_deg
    return label_ok and theta_ok


def report_outside(records: list[dict], rot_deg: float, ds) -> None:
    """サンプル領域が画像外に出る割合（BORDER_REFLECT_101 の反射が入る）を θ=0 と ±θmax で比べる。"""
    from eyeliner.data.augmentation import augment_pq_jitter
    from eyeliner.data.patch_utils import compute_pq_rescale

    counts = {0.0: [0, 0], rot_deg: [0, 0], -rot_deg: [0, 0]}
    for rec in records:
        img = cv2.imread(str(ROOT / rec["image_path"]))  # 絶対パスならそのまま使われる
        if img is None:
            continue
        for theta in counts:
            px, py, qx, qy, t = augment_pq_jitter(
                float(rec["px"]), float(rec["py"]), float(rec["qx"]), float(rec["qy"]), float(rec["t"]),
                aug_flags=AUG_PQ_JITTER, rot_deg=abs(theta) or 1.0, prob=1.0, rng=FixedAngleRng(theta),
            )
            bx = compute_pq_rescale(px, py, qx, qy, t, rec, ds.cfg_b.pq_rescale_mode, ds.cfg_b.pq_rescale_factor)
            counts[theta][0] += footprint_outside(rec, px, py, qx, qy, ds.cfg_a, img.shape)
            counts[theta][1] += footprint_outside(rec, *bx[:4], ds.cfg_b, img.shape)
    n = len(records)
    print("  サンプル領域が画像外に出る割合（枝A / 枝B）:")
    for theta, (a, b) in counts.items():
        print(f"    θ={theta:+5.1f}°: {100 * a / n:5.2f}% / {100 * b / n:5.2f}%")


# ---------------------------------------------------------------------------
# 3. 実データ可視化
# ---------------------------------------------------------------------------

def pick_by_t(records: list[dict], targets=(0.1, 0.5, 0.9, 1.1)) -> list[dict]:
    return [min(records, key=lambda r, tt=tt: abs(float(r["t"]) - tt)) for tt in targets]


def visualize_real(records: list[dict], out_dir: Path) -> None:
    rows = []
    for rec in pick_by_t(records):
        row_a, row_b = [], []
        for theta in THETAS_DEG:
            ds = make_dataset([rec], jitter=True, rot_deg=10.0)
            ds._pq_jitter_rng = FixedAngleRng(theta)
            out = ds[0]
            for x, y, cfg, row in ((out[0], out[4], ds.cfg_a, row_a), (out[1], out[5], ds.cfg_b, row_b)):
                ex, ey = expected_patch_xy(label_to_t(float(y), cfg), cfg.patch_t_min, cfg.patch_t_max,
                                           cfg.patch_size)
                row.append((to_image(x), ex, ey))
        rows.extend([row_a, row_b])
        print(f"  idx={rec.get('idx')} t={float(rec['t']):.3f}  http://127.0.0.1:8000/?idx={rec.get('idx')}")
    save_grid(rows, out_dir / "aug_check_real.png")


def save_grid(rows: list[list[tuple[np.ndarray, float, float]]], path: Path, scale: int = 3) -> None:
    """パッチを格子状に並べ、R の期待位置に十字（マゼンタ）を描いて保存する。"""
    tiles = []
    for row in rows:
        cells = []
        for patch, ex, ey in row:
            big = cv2.resize((patch * 255).astype(np.uint8), None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
            cx, cy = int(round(ex * scale + scale / 2)), int(round(ey * scale + scale / 2))
            cv2.drawMarker(big, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
            cells.append(cv2.copyMakeBorder(big, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tiles.append(np.hstack(cells))
    grid = np.vstack(tiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  保存: {path}（列: θ = {', '.join(f'{t:+.0f}°' for t in THETAS_DEG)}）")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="EXP035 PQ 回転摂動の実装検証")
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--rot-deg", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=EXP_DIR / "outputs" / "aug_check")
    args = ap.parse_args()

    ok_synth = check_synthetic(args.out_dir)
    records = load_real_records(args.n_samples, args.seed)
    ok_real = check_real(records, args.rot_deg, args.seed)
    print("\n[3] 実データ可視化（各サンプル 2 行: 上=枝A、下=枝B）")
    visualize_real(records, args.out_dir)
    print(f"\n総合: {'OK' if ok_synth and ok_real else 'NG'}")
    return 0 if ok_synth and ok_real else 1


if __name__ == "__main__":
    raise SystemExit(main())

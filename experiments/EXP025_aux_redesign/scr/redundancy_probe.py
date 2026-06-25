"""EXP025 Stage A: aux 候補特徴の冗長性プローブ（安価版）。

「特徴 X が入力から予測できる ⟹ X は冗長」という predictability ベースの冗長性検定。
容量3軸が飽和済み（EXP022）＝モデルは容量不足ではないため、
「patch から X を予測できる ⟹ 容量十分なモデルは既に X を抽出済み ⟹ 明示スカラーで足しても無駄」
が意思決定ルールとして成立する。

2種類のプローブ（いずれも再学習なし＝安価）:
  Probe 1 (scalar→scalar): core スカラー(polar_cos/sin, pq_ratio, norm_eye_scale) から
      候補を回帰。線形 + GBM。core 入力との冗長性を測る。
  Probe 2 (frozen CNN linear probe): 学習済み bc32_pw8(EXP024) の penultimate 表現
      (features_a||features_b) から候補を Ridge 回帰。patch 内に情報があるかを測る。

判定は R²（5-fold CV, out-of-fold）。high=冗長(drop) / low=独立(t-arm へ昇格)。
注意: これは「入力との冗長性」を測るのであって「t への relevance」ではない。
      昇格候補の relevance は別途 t-arm（dual k-fold + paired 検定）で検証する。

使い方:
    python experiments/EXP025_aux_redesign/scr/redundancy_probe.py
"""
from __future__ import annotations

# スレッド数を絞る（torch+cv2+sklearn の OpenMP 競合による intermittent segfault を回避）。
# 環境変数は numpy/torch/sklearn の import 前に設定する必要がある。
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

cv2.setNumThreads(1)
torch.set_num_threads(1)

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.training.dual_branch import build_branch_configs  # noqa: E402
from eyeliner.models.dual_ensemble import load_dual_fold_models  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import cross_val_score, GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# 冗長性 R² の CV はこの件数で十分安定。全件(8372)は低RAM環境で OOM するため抑制。
SAMPLE_N = 4000
SAMPLE_SEED = 42

DB_PATH = ROOT / "data" / "samples.db"
MANIFEST_TEST = ROOT / "experiments" / "EXP024_retrain_bc32_pw8" / "manifests" / "manifest_test.csv"
MODEL_DIR = ROOT / "experiments" / "EXP024_retrain_bc32_pw8" / "outputs" / "bc32_pw8"
FUSION_MODE = "independent"
RESCALE_B = "eye_scale"

# DB から join する列（manifest に無い: 顔pose / blur / scale）
DB_COLS = ["face_yaw", "face_pitch", "face_roll", "norm_eye_scale", "lap_var"]

# 画像単位 GroupKFold: 同一画像の全点行を同一 fold に固め、グループリーク
# （per-image/per-eye 定数 = face_pose/lap_var/norm_eye_scale を「画像同定」で記憶する）を防ぐ。
CV = GroupKFold(n_splits=5)


# ---------------------------------------------------------------------------
# データ構築
# ---------------------------------------------------------------------------
def fetch_db_meta(idxs: list[int]) -> dict[int, dict]:
    """samples.db から idx→{DB_COLS} を取得する（manifest に無い列の join）。"""
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    out: dict[int, dict] = {}
    cols = ", ".join(["idx", *DB_COLS])
    for i in range(0, len(idxs), 900):
        chunk = idxs[i : i + 900]
        ph = ",".join("?" * len(chunk))
        for row in cur.execute(f"SELECT {cols} FROM samples WHERE idx IN ({ph})", chunk):
            out[int(row[0])] = dict(zip(DB_COLS, row[1:]))
    con.close()
    return out


def polar(point_index: int, is_upper: bool) -> tuple[float, float]:
    """dataset_dual と同一式の点番号極座標 (cos, sin)。"""
    degree = (point_index + 1) * 22.5 if is_upper else (point_index + 9) * 22.5
    rad = math.radians(degree)
    return math.cos(rad), math.sin(rad)


def build_scalar_features(records: list[dict], db_meta: dict[int, dict]) -> dict[str, np.ndarray]:
    """core 予測子と候補ターゲットのスカラー配列を構築する。"""
    cols: dict[str, list[float]] = {
        "polar_cos": [], "polar_sin": [], "pq_ratio": [], "norm_eye_scale": [],
        "face_yaw_sin": [], "face_yaw_cos": [],
        "face_pitch_sin": [], "face_pitch_cos": [],
        "face_roll_sin": [], "face_roll_cos": [],
        "log_lap_var": [],
        "pq_angle_cos": [], "pq_angle_sin": [], "t": [],
    }
    for rec in records:
        idx = int(rec["idx"])
        meta = db_meta.get(idx, {})
        pi = int(rec.get("point_index", 3))
        is_upper = rec.get("sub_key", "") == "upper"
        pc, ps = polar(pi, is_upper)

        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])
        pq = math.hypot(qx - px, qy - py) or 1.0

        yaw_rad = math.radians(float(meta.get("face_yaw") or 0.0))
        pitch_rad = math.radians(float(meta.get("face_pitch") or 0.0))
        roll_rad = math.radians(float(meta.get("face_roll") or 0.0))

        cols["polar_cos"].append(pc)
        cols["polar_sin"].append(ps)
        cols["pq_ratio"].append(float(rec.get("pq_ratio", 1.0)))
        cols["norm_eye_scale"].append(float(meta.get("norm_eye_scale") or 0.0))
        cols["face_yaw_sin"].append(math.sin(yaw_rad))
        cols["face_yaw_cos"].append(math.cos(yaw_rad))
        cols["face_pitch_sin"].append(math.sin(pitch_rad))
        cols["face_pitch_cos"].append(math.cos(pitch_rad))
        cols["face_roll_sin"].append(math.sin(roll_rad))
        cols["face_roll_cos"].append(math.cos(roll_rad))
        cols["log_lap_var"].append(math.log1p(float(meta.get("lap_var") or 0.0)))
        cols["pq_angle_cos"].append((qx - px) / pq)
        cols["pq_angle_sin"].append((qy - py) / pq)
        cols["t"].append(float(rec["t"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


# ---------------------------------------------------------------------------
# Probe 2: frozen CNN 表現抽出
# ---------------------------------------------------------------------------
def extract_cnn_features(records: list[dict]) -> np.ndarray:
    """学習済み bc32_pw8 fold1 の features_a||features_b を抽出する (N, 2*cnn_flat)。

    抽出は脆い（低RAM + cv2/torch スレッド競合で intermittent segfault）。idx 順を
    検証キーに .npy キャッシュし、同一サブサンプルでの再実行は再抽出せず即返す。
    CPU 強制（CUDA コンテキストの system RAM 予約を回避）。
    """
    idxs = np.asarray([int(r["idx"]) for r in records])
    feat_cache = OUTPUT_DIR / f"cnn_feats_n{SAMPLE_N}_s{SAMPLE_SEED}.npy"
    idx_cache = OUTPUT_DIR / f"cnn_feats_n{SAMPLE_N}_s{SAMPLE_SEED}_idx.npy"
    if feat_cache.exists() and idx_cache.exists() and np.array_equal(np.load(idx_cache), idxs):
        print("  CNN 表現: キャッシュ再利用")
        return np.load(feat_cache)

    device = torch.device("cpu")
    cfg_a, cfg_b = build_branch_configs(RESCALE_B)
    # preload_images=False: 全画像一括ロードは OOM するため遅延読込（__getitem__ がミス時に imread）
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b, augment_data=False, preload_images=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = load_dual_fold_models(MODEL_DIR, FUSION_MODE)[0].to(device)
    model.eval()

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            fa = model.features_a(batch[0]).reshape(batch[0].size(0), -1)
            fb = model.features_b(batch[1]).reshape(batch[1].size(0), -1)
            feats.append(torch.cat([fa, fb], dim=1).numpy().astype(np.float32))
    X = np.concatenate(feats, axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(feat_cache, X)
    np.save(idx_cache, idxs)
    return X


# ---------------------------------------------------------------------------
# プローブ実行
# ---------------------------------------------------------------------------
def r2_cv(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """画像単位 GroupKFold 5-fold CV の平均 R²（out-of-fold）。"""
    scores = cross_val_score(estimator, X, y, groups=groups, scoring="r2", cv=CV, n_jobs=1)
    return float(np.mean(scores))


ABBR = {
    "polar_cos": "pcos", "polar_sin": "psin", "pq_ratio": "pqr", "norm_eye_scale": "nes",
    "face_yaw_sin": "yawS", "face_yaw_cos": "yawC",
    "face_pitch_sin": "pitS", "face_pitch_cos": "pitC",
    "face_roll_sin": "rolS", "face_roll_cos": "rolC",
    "log_lap_var": "lap", "t": "t",
}


def correlations(sc: dict[str, np.ndarray], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pearson / Spearman 相関行列を返す（names 順, 行=列=names）。"""
    M = np.column_stack([sc[n] for n in names])
    pear = np.corrcoef(M, rowvar=False)
    spear = spearmanr(M).correlation
    return pear, spear


def print_corr_matrix(title: str, mat: np.ndarray, names: list[str]) -> None:
    labels = [ABBR[n] for n in names]
    print(f"\n  {title}")
    print("        " + "".join(f"{l:>7}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"  {lab:>6}" + "".join(f"{mat[i, j]:>7.2f}" for j in range(len(labels))))


def verdict(scalar_r2: float, cnn_r2: float) -> str:
    hi = max(scalar_r2, cnn_r2)
    if hi >= 0.9:
        return "REDUNDANT(drop)"
    if hi < 0.5:
        return "INDEPENDENT(t-arm)"
    return "PARTIAL(borderline)"


def main() -> int:
    # Windows コンソールは cp932 で '²' 等を encode できずクラッシュするため UTF-8 に再設定。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print(f"\n{'='*72}")
    print("  EXP025 Stage A: aux 候補特徴 冗長性プローブ（安価版）")
    print(f"{'='*72}")

    records = load_manifest(MANIFEST_TEST)
    compute_eye_meta(records)
    if SAMPLE_N and len(records) > SAMPLE_N:
        rng = np.random.default_rng(SAMPLE_SEED)
        sel = rng.choice(len(records), size=SAMPLE_N, replace=False)
        records = [records[i] for i in sorted(sel.tolist())]
        print(f"  サブサンプル: {SAMPLE_N} 件（seed={SAMPLE_SEED}）")
    idxs = [int(r["idx"]) for r in records]
    db_meta = fetch_db_meta(idxs)
    print(f"  test サンプル: {len(records)}  / DB join 取得: {len(db_meta)}")

    sc = build_scalar_features(records, db_meta)
    # core も含め全特徴を対称に検定する。scalar 側は leave-one-out
    # （各特徴を「残り全特徴」から予測）＝集合内の相互冗長性を測る。
    # pq_angle は廃止済みのため除外。
    FEATURES = [
        "polar_cos", "polar_sin", "pq_ratio", "norm_eye_scale",  # 旧 core（要検証）
        "face_yaw_sin", "face_yaw_cos",                          # 昇格候補（sin/cos化）
        "face_pitch_sin", "face_pitch_cos",
        "face_roll_sin", "face_roll_cos",
        "log_lap_var",
    ]

    # 画像単位グループ（GroupKFold 用）。同一画像の全点行を同一 fold へ。
    img_ids = {p: i for i, p in enumerate({r["image_path"] for r in records})}
    groups = np.asarray([img_ids[r["image_path"]] for r in records])
    print(f"  グループ（画像）数: {len(img_ids)}")

    print("  CNN 表現を抽出中（frozen bc32_pw8 fold1）...")
    X_cnn = extract_cnn_features(records)
    print(f"  CNN 表現次元: {X_cnn.shape}")

    lin = lambda: make_pipeline(StandardScaler(), LinearRegression())
    gbm = lambda: HistGradientBoostingRegressor(max_iter=200, random_state=42)
    ridge = lambda: make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 6))
    )

    results: dict[str, dict] = {}
    for name in FEATURES:
        y = sc[name]
        others = [g for g in FEATURES if g != name]
        X_oth = np.column_stack([sc[g] for g in others])
        s_lin = r2_cv(lin(), X_oth, y, groups)        # 他スカラーからの予測（LOO）
        s_gbm = r2_cv(gbm(), X_oth, y, groups)
        scalar_r2 = max(s_lin, s_gbm)
        cnn_r2 = r2_cv(ridge(), X_cnn, y, groups)     # patch 表現からの予測
        results[name] = {
            "scalar_r2_loo_linear": round(s_lin, 4),
            "scalar_r2_loo_gbm": round(s_gbm, 4),
            "scalar_r2_loo": round(scalar_r2, 4),
            "cnn_r2_ridge": round(cnn_r2, 4),
            "verdict": verdict(scalar_r2, cnn_r2),
        }
        print(f"  [{name:14}] scalarLOO_R²(lin/gbm)={s_lin:+.3f}/{s_gbm:+.3f}  "
              f"cnn_R²={cnn_r2:+.3f}  → {results[name]['verdict']}")

    # ---------- サマリ表 ----------
    print(f"\n  {'feature':14} {'scalarLOO_R²':>13} {'cnn_R²':>8}  verdict")
    print(f"  {'-'*55}")
    for name, r in results.items():
        print(f"  {name:14} {r['scalar_r2_loo']:>13.3f} {r['cnn_r2_ridge']:>8.3f}  {r['verdict']}")

    print("\n  解釈: scalarLOO_R²=他スカラーとの相互冗長性 / cnn_R²=patch内に情報があるか。")
    print("        high(≥0.9)→冗長で捨てる / low(<0.5)→独立 / 中間→borderline。")
    print("        ※ これは冗長性のみ。relevance(t依存)は t-arm で別途検証。")

    # ---------- 相関 ----------
    corr_names = FEATURES + ["t"]
    pear, spear = correlations(sc, corr_names)
    print_corr_matrix("Pearson 相関行列", pear, corr_names)
    print_corr_matrix("Spearman 相関行列", spear, corr_names)
    ti = corr_names.index("t")
    print("\n  各特徴 vs t の相関（relevance の手がかり, redundancy とは別軸）:")
    print(f"  {'feature':14} {'pearson':>9} {'spearman':>9}")
    for i, n in enumerate(FEATURES):
        print(f"  {n:14} {pear[i, ti]:>9.3f} {spear[i, ti]:>9.3f}")

    # ---------- nonlinear relevance: t を各特徴単体から予測（対称依存を拾う） ----------
    print("\n  nonlinear relevance（t を予測, 画像GroupKFold R²。lin≈corr² / gbm=非線形）:")
    print(f"  {'feature':14} {'lin_R²':>8} {'gbm_R²':>8}")
    t_vec = sc["t"]
    relevance: dict[str, dict] = {}
    for n in FEATURES:
        Xf = sc[n].reshape(-1, 1)
        rl = r2_cv(lin(), Xf, t_vec, groups)
        rg = r2_cv(gbm(), Xf, t_vec, groups)
        relevance[n] = {"lin_r2": round(rl, 4), "gbm_r2": round(rg, 4)}
        print(f"  {n:14} {rl:>8.3f} {rg:>8.3f}")
    X_all = np.column_stack([sc[n] for n in FEATURES])
    j_lin = r2_cv(lin(), X_all, t_vec, groups)
    j_gbm = r2_cv(gbm(), X_all, t_vec, groups)
    relevance["_joint_all"] = {"lin_r2": round(j_lin, 4), "gbm_r2": round(j_gbm, 4)}
    print(f"  {'[ALL joint]':14} {j_lin:>8.3f} {j_gbm:>8.3f}  ← aux スカラー単独で t を説明できる上限")

    def mat_to_dict(mat: np.ndarray) -> dict:
        return {a: {b: round(float(mat[i, j]), 4) for j, b in enumerate(corr_names)}
                for i, a in enumerate(corr_names)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "manifest": str(MANIFEST_TEST.relative_to(ROOT)),
        "n_samples": len(records),
        "features": FEATURES,
        "scalar_probe": "leave-one-out (predict each from all others)",
        "cnn_feature_dim": int(X_cnn.shape[1]),
        "results": results,
        "relevance_to_t": relevance,
        "pearson": mat_to_dict(pear),
        "spearman": mat_to_dict(spear),
        "corr_with_t": {n: {"pearson": round(float(pear[i, ti]), 4),
                            "spearman": round(float(spear[i, ti]), 4)}
                        for i, n in enumerate(FEATURES)},
    }
    (OUTPUT_DIR / "redundancy_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  保存: {OUTPUT_DIR / 'redundancy_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

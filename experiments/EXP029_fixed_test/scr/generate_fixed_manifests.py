"""EXP029: 固定テストセット manifest 生成スクリプト.

パターン A: OOF 高誤差画像を fold 別 0/1 ナップサック選択 (画像単位, リークなし)
パターン B: bin 定常分布 (Greedy KL divergence 最小化)
パターン C: KDE ベース分布一致 (Greedy KL divergence 最小化)
パターン D: ランダム画像選択 (ベースライン)
パターン E: 一様分布 (各 t-band = 1/6, Greedy KL 最小化)

使用例:
  python experiments/EXP029_fixed_test/scr/generate_fixed_manifests.py
  python experiments/EXP029_fixed_test/scr/generate_fixed_manifests.py --patterns A
  python experiments/EXP029_fixed_test/scr/generate_fixed_manifests.py --patterns A B --max-eye-groups 350
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from annotator import db as _db  # noqa: E402

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "nose_x", "nose_y",
]

T_BANDS: list[tuple[str, tuple[float, float]]] = [
    ("t<0",     (-np.inf, 0.0)),
    ("0-0.3",   (0.0, 0.3)),
    ("0.3-0.7", (0.3, 0.7)),
    ("0.7-1.0", (0.7, 1.0)),
    ("1.0-1.2", (1.0, 1.2)),
    ("t>=1.2",  (1.2, np.inf)),
]

DEFAULT_OOF_PATH = (
    _PROJECT_ROOT / "experiments" / "EXP023_oof_mislabel" / "outputs"
    / "bc32_pw8_nested" / "oof_predictions.csv"
)

logger = logging.getLogger(__name__)


# ===================================================================
# データ読み込み
# ===================================================================

def load_db_rows(db_path: Path) -> pd.DataFrame:
    """DB から annotated サンプル (t IS NOT NULL, skip_flag=0) を読み込む."""
    samples = _db.get_annotated_samples(db_path)
    rows: list[dict[str, Any]] = []
    for s in samples:
        if s.get("skip_flag", 0):
            continue
        rows.append({
            "idx":         int(s["idx"]),
            "image_path":  str(s["image_path"]).replace("\\", "/"),
            "face_index":  int(s.get("face_index", 0)),
            "part_name":   str(s["part_name"]),
            "sub_key":     str(s.get("sub_key", "")),
            "point_index": int(s.get("point_index", 0)),
            "px":          float(s["px"]),
            "py":          float(s["py"]),
            "qx":          float(s["qx"]),
            "qy":          float(s["qy"]),
            "t":           float(s["t"]) if s.get("t") is not None else None,
            "nose_x":      float(s["nose_x"]) if s.get("nose_x") is not None else None,
            "nose_y":      float(s["nose_y"]) if s.get("nose_y") is not None else None,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["t"])
    logger.info(f"Loaded {len(df)} annotated rows (skip_flag=0) from DB")
    return df


def load_oof_predictions(oof_path: Path) -> pd.DataFrame:
    """OOF 予測 CSV を読み込む.

    Returns DataFrame with columns: idx, image_path, part_name, sub_key,
    point_index, t_true, t_pred, fold_idx
    """
    df = pd.read_csv(oof_path)
    required = {"idx", "t_true", "t_pred", "fold_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF CSV missing columns: {missing}")
    logger.info(f"Loaded {len(df)} OOF predictions from {oof_path}")
    return df


# ===================================================================
# manifest 入出力
# ===================================================================

def write_manifest(df: pd.DataFrame, out_path: Path) -> None:
    """DataFrame を manifest CSV として書き出す."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = df[FIELDNAMES].sort_values("idx").to_dict(orient="records")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows → {out_path}")


# ===================================================================
# 検証
# ===================================================================

def _count_eye_groups(df: pd.DataFrame) -> int:
    """DataFrame 内の eye-group 数を返す（1 eye-group = upper+lower 合算 14 点）."""
    return df.groupby(
        ["image_path", "face_index", "part_name"]
    ).ngroups


def _get_eye_group_set(df: pd.DataFrame) -> set[tuple]:
    """DataFrame 内の eye-group key の集合を返す（1 eye-group = upper+lower 合算）."""
    return set(
        (image_path, face_index, part_name)
        for (image_path, face_index, part_name), _ in df.groupby(
            ["image_path", "face_index", "part_name"]
        )
    )


def validate_manifests(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    db_df: pd.DataFrame,
    pattern_name: str,
) -> dict[str, Any]:
    """生成された train/test manifest を検証する.

    Returns:
        検証結果 dict
    """
    n_total = len(db_df)
    n_train = len(train_df)
    n_test = len(test_df)

    errors: list[str] = []

    # 1. 行数
    if n_train + n_test != n_total:
        errors.append(
            f"train({n_train}) + test({n_test}) != total({n_total})"
        )

    # 2. idx の重複なし
    train_idx = set(train_df["idx"])
    test_idx = set(test_df["idx"])
    overlap = train_idx & test_idx
    if overlap:
        errors.append(f"{len(overlap)} overlapping idx between train and test")

    # 3. test 側の eye-group 完全性（各 14 点: upper 7 + lower 7）
    eg_incomplete = []
    for (_image_path, _face_index, _part_name), group in test_df.groupby(
        ["image_path", "face_index", "part_name"]
    ):
        if len(group) != 14:
            eg_incomplete.append(
                f"{_image_path}/{_face_index}/{_part_name}: {len(group)} pts"
            )
    if eg_incomplete:
        errors.append(f"{len(eg_incomplete)} incomplete eye-groups")

    # 4. 画像レベルのリーク検査
    train_images = set(train_df["image_path"])
    test_images = set(test_df["image_path"])
    leaked_images = train_images & test_images
    if leaked_images:
        errors.append(
            f"{len(leaked_images)} images split across train and test (LEAKAGE!)"
        )

    n_test_eg = _count_eye_groups(test_df)

    result = {
        "pattern": pattern_name,
        "n_train": n_train,
        "n_test": n_test,
        "n_total": n_total,
        "n_test_eye_groups": n_test_eg,
        "n_test_images": len(test_images),
        "errors": errors,
        "valid": len(errors) == 0,
    }

    if errors:
        logger.warning(f"[{pattern_name}] Validation errors: {errors}")
    else:
        logger.info(
            f"[{pattern_name}] ✓ {n_train} train + {n_test} test = {n_total} total, "
            f"{n_test_eg} eye-groups in {len(test_images)} images"
        )

    return result


# ===================================================================
# 貪欲選択（eye-group の平均残差による）
# ===================================================================

def _greedy_select_by_avg_error(
    image_info: dict[str, dict[str, Any]],
    capacity: int,
    descending: bool = True,
) -> list[str]:
    """avg_residual 降順(または昇順) greedy + gap fill.

    1. avg_residual 降順(descending=True, hard)または昇順(descending=False, easy)に
       ソートし、capacity を超えないよう貪欲選択
    2. 残り容量に収まる最良 avg_residual の未選択画像で隙間を埋める

    Args:
        image_info: {image_path: {eye_group_count, residual_sum, avg_residual}}
        capacity: 最大 eye-group 数
        descending: True なら高誤差(hard)優先、False なら低誤差(easy)優先

    Returns:
        選択された image_path のリスト
    """
    sorted_images = sorted(
        image_info.items(),
        key=lambda kv: kv[1]["avg_residual"],
        reverse=descending,
    )

    selected: list[str] = []
    selected_set: set[str] = set()
    total_eg = 0

    # フェーズ1: avg_residual の降順/昇順で貪欲選択
    for img_path, info in sorted_images:
        eg_count = info["eye_group_count"]
        if total_eg + eg_count <= capacity:
            selected.append(img_path)
            selected_set.add(img_path)
            total_eg += eg_count

    # フェーズ2: 残り容量を最良の avg_residual の画像で埋める
    remaining = capacity - total_eg
    if remaining > 0:
        best_fill: Optional[str] = None
        best_fill_score = -1.0 if descending else float("inf")
        for img_path, info in sorted_images:
            if img_path in selected_set:
                continue
            if info["eye_group_count"] <= remaining:
                better = (
                    info["avg_residual"] > best_fill_score if descending
                    else info["avg_residual"] < best_fill_score
                )
                if better:
                    best_fill_score = info["avg_residual"]
                    best_fill = img_path
        if best_fill is not None:
            selected.append(best_fill)
            total_eg += image_info[best_fill]["eye_group_count"]

    return selected


# ===================================================================
# t-band ヘルパ
# ===================================================================

def _t_to_band(t: float) -> int:
    """t 値を band index (0-5) に変換."""
    for i, (_name, (lo, hi)) in enumerate(T_BANDS):
        if lo <= t < hi:
            return i
    return 5  # t>=1.2


def _compute_target_dist(
    db_df: pd.DataFrame,
    method: str = "empirical",
    kde_bandwidth: Optional[float] = None,
    random_seed: int = 42,
) -> np.ndarray:
    """目標 t-band 分布 (6次元, sum=1) を計算.

    Args:
        db_df: 全 annotated データ
        method: "empirical" → 全点の band 比率
                "kde" → KDE からサンプリングして band 比率
        kde_bandwidth: KDE bandwidth (None → Scott 則)
        random_seed: KDE サンプリング用シード

    Returns:
        target_dist: shape (6,) numpy array, sum=1
    """
    t_values = db_df["t"].values.astype(np.float64)

    if method == "empirical":
        bands = np.array([_t_to_band(float(t)) for t in t_values])
        counts = np.bincount(bands, minlength=6)
        dist = counts.astype(np.float64) / counts.sum()
        return dist

    elif method == "kde":
        from scipy.stats import gaussian_kde

        t_clipped = t_values[(t_values >= 0.0) & (t_values <= 1.2)]
        if len(t_clipped) < 100:
            logger.warning(
                "Too few points in [0,1.2] for KDE; falling back to empirical"
            )
            return _compute_target_dist(db_df, method="empirical")

        kde = gaussian_kde(t_clipped, bw_method=kde_bandwidth)
        rng = np.random.RandomState(random_seed)
        samples = kde.resample(len(t_values), seed=rng).flatten()
        samples = np.clip(samples, float(t_values.min()), float(t_values.max()))
        bands = np.array([_t_to_band(float(t)) for t in samples])
        counts = np.bincount(bands, minlength=6)
        dist = counts.astype(np.float64) / counts.sum()
        return dist

    else:
        raise ValueError(f"Unknown method: {method}")


# ===================================================================
# 分布距離指標
# ===================================================================

def _distribution_distance(
    counts: list[int],
    total: int,
    target: np.ndarray,
    metric: str = "kl",
    eps: float = 1e-10,
) -> float:
    """現在の band 分布と目標分布の距離を計算.

    Args:
        counts: 現在の band カウント [6]
        total: 総点数
        target: 目標分布 [6], sum=1
        metric: "kl" | "l2"
        eps: KL 計算時のゼロ除算防止

    Returns:
        距離スコア (小さいほど良い)
    """
    if total == 0:
        return float("inf")

    p = np.array(counts, dtype=np.float64) / total
    target = np.asarray(target, dtype=np.float64)

    if metric == "kl":
        mask = (p > 0) & (target > eps)
        if not mask.any():
            return float("inf")
        return float(np.sum(p[mask] * np.log(p[mask] / target[mask])))

    elif metric == "l2":
        return float(np.sum((p - target) ** 2))

    else:
        raise ValueError(f"Unknown metric: {metric}")


# ===================================================================
# 画像情報の構築
# ===================================================================

def _build_per_image_info(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """画像ごとの band カウントと eye-group 情報を構築.

    Args:
        df: columns include image_path, face_index, part_name, sub_key, t

    Returns:
        {image_path: {band_cnt: [6], eye_group_cnt: int, eye_group_keys: list}}
    """
    images: dict[str, dict[str, Any]] = {}
    for image_path, img_group in df.groupby("image_path"):
        band_cnt = [0] * 6
        for t in img_group["t"]:
            band_cnt[_t_to_band(float(t))] += 1

        eg_keys = sorted(set(
            (int(row["face_index"]), str(row["part_name"]))
            for _, row in img_group.iterrows()
        ))
        eye_group_cnt = len(eg_keys)

        images[image_path] = {
            "band_cnt": band_cnt,
            "eye_group_cnt": eye_group_cnt,
            "eye_group_keys": eg_keys,
        }

    return images


# ===================================================================
# パターンA: OOF 貪欲選択（画像単位・リークなし）
# ===================================================================

def _prepare_oof_fold_image_info(
    db_df: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> dict[int, dict[str, dict[str, Any]]]:
    """OOF 予測と DB をマージし、fold別・画像別の avg_residual/eye_group_count を計算する.

    OOF の avg_residual でランク付けし、DB の eye_group_count で
    容量管理するため、選択後も eye-group 数がずれない。

    Returns:
        {fold_idx: {image_path: {eye_group_count, residual_sum, avg_residual}}}
    """
    # 画像ごとの DB eye-group 数を事前計算
    db_eg_counts: dict[str, int] = {}
    for image_path, group in db_df.groupby("image_path"):
        db_eg_counts[image_path] = group.groupby(
            ["face_index", "part_name"]
        ).ngroups

    # face_index を得るため OOF と DB を結合
    db_lookup = db_df[["idx", "face_index"]].copy()
    merged = oof_df.merge(db_lookup, on="idx", how="inner")

    n_missing = len(oof_df) - len(merged)
    if n_missing > 0:
        logger.warning(
            f"OOF: {n_missing} OOF rows not found in current DB"
        )

    merged["abs_error"] = (merged["t_true"] - merged["t_pred"]).abs()

    # fold の整合確認
    fold_per_image = merged.groupby("image_path")["fold_idx"].nunique()
    inconsistent = fold_per_image[fold_per_image > 1]
    if len(inconsistent) > 0:
        logger.warning(
            f"OOF: {len(inconsistent)} images have rows in multiple folds "
            f"(nested CV assumption violated); using majority fold"
        )
        majority_fold = (
            merged.groupby("image_path")["fold_idx"]
            .agg(lambda x: x.mode().iloc[0])
            .rename("fold_idx_majority")
        )
        merged = merged.merge(majority_fold, on="image_path")
        merged["fold_idx"] = merged["fold_idx_majority"].astype(int)
        merged = merged.drop(columns=["fold_idx_majority"])

    result: dict[int, dict[str, dict[str, Any]]] = {}
    for fold_idx in sorted(merged["fold_idx"].unique()):
        fold_data = merged[merged["fold_idx"] == fold_idx]

        image_info: dict[str, dict[str, Any]] = {}
        for image_path, group in fold_data.groupby("image_path"):
            residual_sum = float(group["abs_error"].sum())
            oof_eg_count = group.groupby(
                ["face_index", "part_name"]
            ).ngroups
            avg_residual = residual_sum / oof_eg_count if oof_eg_count > 0 else 0.0
            # 容量管理を正確にするため DB の eye-group 数を使う
            db_eg = db_eg_counts.get(image_path, oof_eg_count)
            image_info[image_path] = {
                "eye_group_count": db_eg,
                "residual_sum": residual_sum,
                "avg_residual": avg_residual,
            }
        result[int(fold_idx)] = image_info

    return result


def generate_pattern_a_hard(
    db_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    n_per_fold: int = 70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン oof_error_hard: fold 別 avg_residual 降順 greedy 選択 (画像単位).

    Args:
        db_df: DB annotated rows (skip_flag=0)
        oof_df: OOF predictions (idx, t_true, t_pred, fold_idx, ...)
        n_per_fold: fold あたりの目標 eye-group 数

    Returns:
        (train_df, test_df)
    """
    fold_info = _prepare_oof_fold_image_info(db_df, oof_df)
    folds = sorted(fold_info.keys())
    logger.info(f"Pattern oof_error_hard: processing {len(folds)} folds: {folds}")

    all_test_image_paths: set[str] = set()

    for fold_idx in folds:
        image_info = fold_info[fold_idx]

        if not image_info:
            logger.warning(f"Pattern oof_error_hard fold {fold_idx}: no images found")
            continue

        selected = _greedy_select_by_avg_error(image_info, n_per_fold, descending=True)

        total_eg = sum(
            image_info[img]["eye_group_count"] for img in selected
        )
        total_residual = sum(
            image_info[img]["residual_sum"] for img in selected
        )

        logger.info(
            f"  Fold {fold_idx}: {len(selected)}/{len(image_info)} images, "
            f"{total_eg}/{n_per_fold} eye-groups, "
            f"residual_sum={total_residual:.4f}"
        )

        all_test_image_paths.update(selected)

    # 合計確認（DB の件数を使うため n_per_fold * n_folds 以下になるはず）
    test_df = db_df[db_df["image_path"].isin(all_test_image_paths)].copy()
    actual_eg = _count_eye_groups(test_df)
    max_total_eg = n_per_fold * len(folds)

    if actual_eg > max_total_eg:
        logger.warning(
            f"Pattern oof_error_hard: {actual_eg} eye-groups > target {max_total_eg} "
            f"(unexpected; DB counts should match)"
        )

    logger.info(
        f"Pattern oof_error_hard: {len(all_test_image_paths)} test images, "
        f"{actual_eg} eye-groups"
    )

    train_df = db_df[~db_df["image_path"].isin(all_test_image_paths)].copy()

    return train_df, test_df


def generate_pattern_a_easy(
    db_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    n_per_fold: int = 70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン oof_error_easy: fold 別 avg_residual 昇順 greedy 選択 (低誤差サンプル).

    generate_pattern_a (hard) と対になる、モデルが得意なサンプルを集めたテストセット。

    Args:
        db_df: DB annotated rows (skip_flag=0)
        oof_df: OOF predictions (idx, t_true, t_pred, fold_idx, ...)
        n_per_fold: fold あたりの目標 eye-group 数

    Returns:
        (train_df, test_df)
    """
    fold_info = _prepare_oof_fold_image_info(db_df, oof_df)
    folds = sorted(fold_info.keys())
    logger.info(f"Pattern oof_error_easy: processing {len(folds)} folds: {folds}")

    all_test_image_paths: set[str] = set()

    for fold_idx in folds:
        image_info = fold_info[fold_idx]

        if not image_info:
            logger.warning(f"Pattern oof_error_easy fold {fold_idx}: no images found")
            continue

        selected = _greedy_select_by_avg_error(image_info, n_per_fold, descending=False)

        total_eg = sum(
            image_info[img]["eye_group_count"] for img in selected
        )
        total_residual = sum(
            image_info[img]["residual_sum"] for img in selected
        )

        logger.info(
            f"  Fold {fold_idx}: {len(selected)}/{len(image_info)} images, "
            f"{total_eg}/{n_per_fold} eye-groups, "
            f"residual_sum={total_residual:.4f}"
        )

        all_test_image_paths.update(selected)

    test_df = db_df[db_df["image_path"].isin(all_test_image_paths)].copy()
    actual_eg = _count_eye_groups(test_df)

    logger.info(
        f"Pattern oof_error_easy: {len(all_test_image_paths)} test images, "
        f"{actual_eg} eye-groups"
    )

    train_df = db_df[~db_df["image_path"].isin(all_test_image_paths)].copy()

    return train_df, test_df


def generate_pattern_a_half(
    db_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    n_per_fold_each: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン oof_error_half: fold 別に hard 側 + easy 側を半々ずつ greedy 選択.

    generate_pattern_a_hard (hard, 容量 n_per_fold_each*2 相当の上位半分) と
    generate_pattern_a_easy (easy, 下位半分) それぞれと部分集合として重なるように、
    同一の greedy アルゴリズムを容量 n_per_fold_each ずつ両方向に適用する。
    これにより oof_error_hard ∩ oof_error_half、oof_error_easy ∩ oof_error_half が
    非空になり、paired t-test で「hard専用モデル vs half-halfモデル」等の
    共通サンプルでの比較が可能になる。

    Args:
        db_df: DB annotated rows (skip_flag=0)
        oof_df: OOF predictions (idx, t_true, t_pred, fold_idx, ...)
        n_per_fold_each: fold あたり・方向あたりの目標 eye-group 数
            (hard 側・easy 側それぞれこの数だけ選択される)

    Returns:
        (train_df, test_df)
    """
    fold_info = _prepare_oof_fold_image_info(db_df, oof_df)
    folds = sorted(fold_info.keys())
    logger.info(f"Pattern oof_error_half: processing {len(folds)} folds: {folds}")

    all_test_image_paths: set[str] = set()

    for fold_idx in folds:
        image_info = fold_info[fold_idx]

        if not image_info:
            logger.warning(f"Pattern oof_error_half fold {fold_idx}: no images found")
            continue

        hard_selected = _greedy_select_by_avg_error(
            image_info, n_per_fold_each, descending=True
        )
        hard_set = set(hard_selected)

        # easy 側は hard 側で選ばれた画像を除外してから選定（重複防止）
        easy_candidates = {
            img: info for img, info in image_info.items() if img not in hard_set
        }
        easy_selected = _greedy_select_by_avg_error(
            easy_candidates, n_per_fold_each, descending=False
        )

        selected = hard_selected + easy_selected
        total_eg = sum(image_info[img]["eye_group_count"] for img in selected)

        logger.info(
            f"  Fold {fold_idx}: hard={len(hard_selected)} + easy={len(easy_selected)} images, "
            f"{total_eg}/{n_per_fold_each * 2} eye-groups"
        )

        all_test_image_paths.update(selected)

    test_df = db_df[db_df["image_path"].isin(all_test_image_paths)].copy()
    actual_eg = _count_eye_groups(test_df)

    logger.info(
        f"Pattern oof_error_half: {len(all_test_image_paths)} test images, "
        f"{actual_eg} eye-groups"
    )

    train_df = db_df[~db_df["image_path"].isin(all_test_image_paths)].copy()

    return train_df, test_df


# ===================================================================
# パターンB: bin 定常分布（Greedy KL 最小化）
# ===================================================================

def generate_pattern_b(
    db_df: pd.DataFrame,
    max_eye_groups: int = 350,
    dist_metric: str = "kl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン B: 全データの t-band 比率を目標とした Greedy 分布一致選択.

    Args:
        db_df: DB annotated rows (skip_flag=0)
        max_eye_groups: 目標 eye-group 数
        dist_metric: 分布距離指標 ("kl" | "l2")

    Returns:
        (train_df, test_df)
    """
    target_dist = _compute_target_dist(db_df, method="empirical")
    band_names = [b[0] for b in T_BANDS]
    logger.info(
        f"Pattern B target: {dict(zip(band_names, np.round(target_dist, 4)))}"
    )

    images = _build_per_image_info(db_df)
    image_list = sorted(images.keys())

    selected_ids: set[str] = set()
    current_bands = [0] * 6
    current_total_eg = 0

    while current_total_eg < max_eye_groups and len(selected_ids) < len(image_list):
        best_id: Optional[str] = None
        best_score = float("inf")

        # 最初の 1 件は KL(0||Q) = inf を避けるため L2 を使う
        m = "l2" if len(selected_ids) == 0 else dist_metric

        for img_id in image_list:
            if img_id in selected_ids:
                continue
            img = images[img_id]
            candidate_bands = [
                current_bands[k] + img["band_cnt"][k] for k in range(6)
            ]
            candidate_total = current_total_eg + img["eye_group_cnt"]

            score = _distribution_distance(
                candidate_bands, candidate_total, target_dist, metric=m
            )
            if score < best_score:
                best_score = score
                best_id = img_id

        if best_id is None:
            break

        selected_ids.add(best_id)
        img = images[best_id]
        for k in range(6):
            current_bands[k] += img["band_cnt"][k]
        current_total_eg += img["eye_group_cnt"]

    logger.info(
        f"Pattern B: {len(selected_ids)} images, {current_total_eg} eye-groups"
    )

    test_df = db_df[db_df["image_path"].isin(selected_ids)].copy()
    train_df = db_df[~db_df["image_path"].isin(selected_ids)].copy()

    return train_df, test_df


# ===================================================================
# パターンC: KDE ベース分布一致（Greedy KL 最小化）
# ===================================================================

def generate_pattern_c(
    db_df: pd.DataFrame,
    max_eye_groups: int = 350,
    dist_metric: str = "kl",
    kde_bandwidth: Optional[float] = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン C: KDE 密度推定 → 理論分布を目標とした Greedy 選択.

    Args:
        db_df: DB annotated rows (skip_flag=0)
        max_eye_groups: 目標 eye-group 数
        dist_metric: 分布距離指標 ("kl" | "l2")
        kde_bandwidth: KDE bandwidth (None → Scott 則)
        random_seed: KDE サンプリング用シード

    Returns:
        (train_df, test_df)
    """
    target_dist = _compute_target_dist(
        db_df, method="kde", kde_bandwidth=kde_bandwidth, random_seed=random_seed
    )
    band_names = [b[0] for b in T_BANDS]
    logger.info(
        f"Pattern C target (KDE): {dict(zip(band_names, np.round(target_dist, 4)))}"
    )

    # パターンBと同じ貪欲アルゴリズム
    images = _build_per_image_info(db_df)
    image_list = sorted(images.keys())

    selected_ids: set[str] = set()
    current_bands = [0] * 6
    current_total_eg = 0

    while current_total_eg < max_eye_groups and len(selected_ids) < len(image_list):
        best_id: Optional[str] = None
        best_score = float("inf")

        m = "l2" if len(selected_ids) == 0 else dist_metric

        for img_id in image_list:
            if img_id in selected_ids:
                continue
            img = images[img_id]
            candidate_bands = [
                current_bands[k] + img["band_cnt"][k] for k in range(6)
            ]
            candidate_total = current_total_eg + img["eye_group_cnt"]

            score = _distribution_distance(
                candidate_bands, candidate_total, target_dist, metric=m
            )
            if score < best_score:
                best_score = score
                best_id = img_id

        if best_id is None:
            break

        selected_ids.add(best_id)
        img = images[best_id]
        for k in range(6):
            current_bands[k] += img["band_cnt"][k]
        current_total_eg += img["eye_group_cnt"]

    logger.info(
        f"Pattern C: {len(selected_ids)} images, {current_total_eg} eye-groups"
    )

    test_df = db_df[db_df["image_path"].isin(selected_ids)].copy()
    train_df = db_df[~db_df["image_path"].isin(selected_ids)].copy()

    return train_df, test_df


# ===================================================================
# パターンD: ランダム画像選択（ベースライン）
# ===================================================================

def generate_pattern_d(
    db_df: pd.DataFrame,
    max_eye_groups: int = 350,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン D: ランダム画像選択 (ベースライン).

    Args:
        db_df: DB annotated rows (skip_flag=0)
        max_eye_groups: 目標 eye-group 数
        random_seed: 乱数シード

    Returns:
        (train_df, test_df)
    """
    rng = np.random.RandomState(random_seed)
    image_paths = sorted(db_df["image_path"].unique())
    rng.shuffle(image_paths)

    selected_ids: set[str] = set()
    current_total_eg = 0

    for img_id in image_paths:
        img_df = db_df[db_df["image_path"] == img_id]
        eg_count = img_df.groupby(
            ["face_index", "part_name"]
        ).ngroups

        if current_total_eg + eg_count > max_eye_groups + 1:
            break

        selected_ids.add(img_id)
        current_total_eg += eg_count

        if current_total_eg >= max_eye_groups:
            break

    logger.info(
        f"Pattern D: {len(selected_ids)} images, {current_total_eg} eye-groups"
    )

    test_df = db_df[db_df["image_path"].isin(selected_ids)].copy()
    train_df = db_df[~db_df["image_path"].isin(selected_ids)].copy()

    return train_df, test_df


# ===================================================================
# パターンE: 一様 t-band 分布（Greedy KL 最小化）
# ===================================================================

def generate_pattern_e(
    db_df: pd.DataFrame,
    max_eye_groups: int = 350,
    dist_metric: str = "kl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """パターン E: 一様分布 (各 t-band = 1/6) を目標とした Greedy 選択.

    極端な t 値を含む eye-group を積極的に選択し、
    テストセットの t-band 分布を可能な限り平坦化する。

    Args:
        db_df: DB annotated rows (skip_flag=0)
        max_eye_groups: 目標 eye-group 数
        dist_metric: 分布距離指標 ("kl" | "l2")

    Returns:
        (train_df, test_df)
    """
    # 一様目標: 各帯 1/6 ≈ 16.67%
    target_dist = np.ones(6, dtype=np.float64) / 6.0
    band_names = [b[0] for b in T_BANDS]
    logger.info(
        f"Pattern E target (uniform): {dict(zip(band_names, np.round(target_dist, 4)))}"
    )

    images = _build_per_image_info(db_df)
    image_list = sorted(images.keys())

    selected_ids: set[str] = set()
    current_bands = [0] * 6
    current_total_eg = 0

    while current_total_eg < max_eye_groups and len(selected_ids) < len(image_list):
        best_id: Optional[str] = None
        best_score = float("inf")

        m = "l2" if len(selected_ids) == 0 else dist_metric

        for img_id in image_list:
            if img_id in selected_ids:
                continue
            img = images[img_id]
            candidate_bands = [
                current_bands[k] + img["band_cnt"][k] for k in range(6)
            ]
            candidate_total = current_total_eg + img["eye_group_cnt"]

            score = _distribution_distance(
                candidate_bands, candidate_total, target_dist, metric=m
            )
            if score < best_score:
                best_score = score
                best_id = img_id

        if best_id is None:
            break

        selected_ids.add(best_id)
        img = images[best_id]
        for k in range(6):
            current_bands[k] += img["band_cnt"][k]
        current_total_eg += img["eye_group_cnt"]

    logger.info(
        f"Pattern E: {len(selected_ids)} images, {current_total_eg} eye-groups"
    )

    # 達成された分布を集計
    final_p = np.array(current_bands, dtype=np.float64) / sum(current_bands)
    logger.info(
        f"Pattern E achieved: {dict(zip(band_names, np.round(final_p, 4)))}"
    )

    test_df = db_df[db_df["image_path"].isin(selected_ids)].copy()
    train_df = db_df[~db_df["image_path"].isin(selected_ids)].copy()

    return train_df, test_df


# ===================================================================
# パターン間重複行列
# ===================================================================

def compute_overlap_matrix(
    test_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """パターン間の eye-group 重複行列を計算."""
    patterns = sorted(test_dfs.keys())

    eg_sets: dict[str, set[tuple]] = {}
    for name, df in test_dfs.items():
        eg_sets[name] = _get_eye_group_set(df)

    data: dict[str, dict[str, int]] = {}
    for p1 in patterns:
        data[p1] = {}
        for p2 in patterns:
            data[p1][p2] = len(eg_sets[p1] & eg_sets[p2])

    return pd.DataFrame(data, index=patterns, columns=patterns)


# ===================================================================
# メイン
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP029: Generate fixed test set manifests (5 patterns)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_PROJECT_ROOT / "data" / "samples.db",
        help="Path to samples.db",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "experiments" / "EXP029_fixed_test" / "manifests",
        help="Output directory for manifest CSVs",
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=DEFAULT_OOF_PATH,
        help="Path to OOF predictions CSV (Pattern A only)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["oof_error_hard", "db_replica", "random"],
        help="Patterns to generate (default: A B C D E)",
    )
    parser.add_argument(
        "--max-eye-groups",
        type=int,
        default=350,
        help="Target total eye-groups for test set (default: 350)",
    )
    parser.add_argument(
        "--n-per-fold",
        type=int,
        default=70,
        help="Target eye-groups per fold for Pattern A (default: 70)",
    )
    parser.add_argument(
        "--dist-metric",
        type=str,
        choices=["kl", "l2"],
        default="kl",
        help="Distribution distance metric for Patterns B/C (default: kl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 入力検証
    if not args.db.exists():
        logger.error(f"DB not found: {args.db}")
        sys.exit(1)

    valid_patterns = {
        "oof_error_hard", "oof_error_easy", "oof_error_half", "db_replica", "random",
    }
    for p in args.patterns:
        if p not in valid_patterns:
            logger.error(f"Invalid pattern: {p}. Must be one of {valid_patterns}")
            sys.exit(1)

    # データ読み込み
    logger.info("Loading DB...")
    db_df = load_db_rows(args.db)

    oof_patterns = {"oof_error_hard", "oof_error_easy", "oof_error_half"}
    oof_df = None
    if oof_patterns & set(args.patterns):
        if not args.oof.exists():
            logger.error(f"OOF predictions not found: {args.oof}")
            sys.exit(1)
        logger.info("Loading OOF predictions...")
        oof_df = load_oof_predictions(args.oof)

    # 各パターンの生成
    test_dfs: dict[str, pd.DataFrame] = {}
    results: list[dict[str, Any]] = []

    for pattern in args.patterns:
        logger.info(f"{'='*60}")
        logger.info(f"Generating Pattern {pattern}...")

        if pattern == "oof_error_hard":
            assert oof_df is not None
            train_df, test_df = generate_pattern_a_hard(
                db_df, oof_df, n_per_fold=args.n_per_fold,
            )
        elif pattern == "oof_error_easy":
            assert oof_df is not None
            train_df, test_df = generate_pattern_a_easy(
                db_df, oof_df, n_per_fold=args.n_per_fold,
            )
        elif pattern == "oof_error_half":
            assert oof_df is not None
            train_df, test_df = generate_pattern_a_half(
                db_df, oof_df, n_per_fold_each=args.n_per_fold // 2,
            )
        elif pattern == "db_replica":
            train_df, test_df = generate_pattern_b(
                db_df,
                max_eye_groups=args.max_eye_groups,
                dist_metric=args.dist_metric,
            )
        elif pattern == "random":
            train_df, test_df = generate_pattern_d(
                db_df,
                max_eye_groups=args.max_eye_groups,
                random_seed=args.seed,
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        # 検証
        result = validate_manifests(train_df, test_df, db_df, pattern)
        results.append(result)

        if not result["valid"]:
            logger.error(
                f"Pattern {pattern} validation FAILED, skipping write."
            )
            continue

        # manifest の書き出し
        write_manifest(train_df, args.out / f"manifest_train_{pattern}.csv")
        write_manifest(test_df, args.out / f"manifest_test_{pattern}.csv")

        test_dfs[pattern] = test_df

    # パターン間重複行列
    if len(test_dfs) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("Cross-pattern eye-group overlap:")
        overlap = compute_overlap_matrix(test_dfs)
        for p1 in sorted(test_dfs.keys()):
            parts = [
                f"{p1}↔{p2}: {overlap.loc[p1, p2]}"
                for p2 in sorted(test_dfs.keys())
                if p1 != p2
            ]
            logger.info(f"  {p1} (n={overlap.loc[p1, p1]}): {' | '.join(parts)}")

    # サマリ
    logger.info(f"\n{'='*60}")
    logger.info("Summary:")
    for r in results:
        status = "✓" if r["valid"] else "✗"
        logger.info(
            f"  [{status}] Pattern {r['pattern']}: "
            f"test={r['n_test']} rows, {r['n_test_eye_groups']} eye-groups, "
            f"{r['n_test_images']} images"
        )

    # パターン oof_error_hard: 画像レベルのリークがないことの確認
    if "oof_error_hard" in test_dfs:
        train_imgs = set(db_df["image_path"]) - set(test_dfs["oof_error_hard"]["image_path"])
        test_imgs = set(test_dfs["oof_error_hard"]["image_path"])
        leak = train_imgs & test_imgs
        if leak:
            logger.error(
                f"Pattern oof_error_hard LEAKAGE: {len(leak)} images in both train & test!"
            )
        else:
            logger.info("Pattern oof_error_hard: image-level leakage check passed ✓")


if __name__ == "__main__":
    main()

"""EXP029: Fixed test manifests EDA (Exploratory Data Analysis).

5 パターン (A/B/C/D/E) のテスト manifest を比較分析し、
t 値分布・画像特性・パターン間重複を可視化する。

出力:
  outputs/eda_report.md   — サマリーレポート
  outputs/eda_*.png       — プロット群
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1]
_MANIFEST_DIR = _EXP_DIR / "manifests"
_OUTPUT_DIR = _EXP_DIR / "outputs"
sys.path.insert(0, str(_PROJECT_ROOT))

from annotator import db as _db  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATTERNS = ["oof_error_hard", "db_replica", "random"]
PATTERN_LABELS = {
    "oof_error_hard": "OOF high-error (greedy)",
    "db_replica": "DB replica (KL greedy)",
    "random": "Random (baseline)",
}
COLORS = {
    "oof_error_hard": "#E74C3C", "db_replica": "#3498DB", "random": "#F39C12",
}
DB_COLOR = "#7F8C8D"

T_BANDS: list[tuple[str, tuple[float, float]]] = [
    ("t<0",     (-np.inf, 0.0)),
    ("0-0.3",   (0.0, 0.3)),
    ("0.3-0.7", (0.3, 0.7)),
    ("0.7-1.0", (0.7, 1.0)),
    ("1.0-1.2", (1.0, 1.2)),
    ("t>=1.2",  (1.2, np.inf)),
]

# Matplotlib 日本語フォント設定
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


# ===================================================================
# Helpers
# ===================================================================

def _t_to_band(t: float) -> int:
    for i, (_name, (lo, hi)) in enumerate(T_BANDS):
        if lo <= t < hi:
            return i
    return 5


def _t_to_band_name(t: float) -> str:
    return T_BANDS[_t_to_band(t)][0]


def _eye_group_key(row: pd.Series) -> tuple:
    return (row["image_path"], row["face_index"],
            row["part_name"], row["sub_key"])


def load_manifest(pattern: str) -> pd.DataFrame:
    path = _MANIFEST_DIR / f"manifest_test_{pattern}.csv"
    return pd.read_csv(path)


def load_db_reference() -> pd.DataFrame:
    """DB の全 annotated (skip_flag=0) 行を読み込む."""
    samples = _db.get_annotated_samples()
    rows = []
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
            "t":           float(s["t"]) if s.get("t") is not None else None,
            "face_yaw":    float(s["face_yaw"]) if s.get("face_yaw") is not None else None,
            "face_pitch":  float(s["face_pitch"]) if s.get("face_pitch") is not None else None,
            "face_roll":   float(s["face_roll"]) if s.get("face_roll") is not None else None,
            "eye_scale":   float(s["eye_scale"]) if s.get("eye_scale") is not None else None,
            "norm_eye_scale": float(s["norm_eye_scale"]) if s.get("norm_eye_scale") is not None else None,
            "lap_var":     float(s["lap_var"]) if s.get("lap_var") is not None else None,
        })
    return pd.DataFrame(rows).dropna(subset=["t"])


def count_eye_groups(df: pd.DataFrame) -> int:
    """1 eye-group = upper+lower 合算 14 点."""
    return df.groupby(["image_path", "face_index", "part_name"]).ngroups


def count_images(df: pd.DataFrame) -> int:
    return df["image_path"].nunique()


def compute_band_distribution(df: pd.DataFrame) -> np.ndarray:
    """t 値の 6-band 分布を返す (shape=6, sum=1)."""
    t_vals = df["t"].values.astype(np.float64)
    bands = np.array([_t_to_band(float(t)) for t in t_vals])
    counts = np.bincount(bands, minlength=6)
    return counts.astype(np.float64) / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(P||Q)."""
    mask = (p > 0) & (q > eps)
    if not mask.any():
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((p - q) ** 2))


# ===================================================================
# Plotting functions
# ===================================================================

def plot_t_distribution(
    data: dict[str, pd.DataFrame],
    db_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """t 値分布: パターン別 histogram + DB reference KDE overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    t_range = (0.0, 1.2)
    x = np.linspace(t_range[0], t_range[1], 500)

    # DB KDE
    db_t = db_df["t"].values.astype(np.float64)
    db_t_clipped = db_t[(db_t >= t_range[0]) & (db_t <= t_range[1])]
    db_kde = gaussian_kde(db_t_clipped)
    db_y = db_kde(x)

    for idx, pattern in enumerate(PATTERNS):
        ax = axes[idx]
        df = data[pattern]
        t_vals = df["t"].values.astype(np.float64)

        # Histogram
        ax.hist(t_vals, bins=40, range=t_range, density=True,
                alpha=0.5, color=COLORS[pattern],
                label=f"Pattern {pattern} (n={len(t_vals)})")

        # DB reference KDE
        ax.plot(x, db_y, color=DB_COLOR, linewidth=2, linestyle="--",
                label=f"DB all (n={len(db_t_clipped)})")

        ax.set_title(f"Pattern {pattern}: {PATTERN_LABELS[pattern]}")
        ax.set_xlabel("t value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(t_range)

    # Hide unused subplot
    if len(PATTERNS) < len(axes):
        for idx in range(len(PATTERNS), len(axes)):
            axes[idx].set_visible(False)

    fig.suptitle("t-value Distribution by Pattern vs DB Reference",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_t_distribution_overlay(
    data: dict[str, pd.DataFrame],
    db_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """全パターンの t 値 KDE を 1 枚に重ね書き."""
    fig, ax = plt.subplots(figsize=(12, 6))

    t_range = (0.0, 1.2)
    x = np.linspace(t_range[0], t_range[1], 500)

    # DB reference
    db_t = db_df["t"].values.astype(np.float64)
    db_t_clipped = db_t[(db_t >= t_range[0]) & (db_t <= t_range[1])]
    db_kde = gaussian_kde(db_t_clipped)
    ax.plot(x, db_kde(x), color=DB_COLOR, linewidth=2.5, linestyle="--",
            label=f"DB all (reference)")

    for pattern in PATTERNS:
        t_vals = data[pattern]["t"].values.astype(np.float64)
        t_clipped = t_vals[(t_vals >= t_range[0]) & (t_vals <= t_range[1])]
        if len(t_clipped) < 3:
            continue
        kde = gaussian_kde(t_clipped)
        ax.plot(x, kde(x), color=COLORS[pattern], linewidth=1.8,
                label=f"Pattern {pattern}")

    ax.set_title("t-value KDE: All Patterns vs DB Reference",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_xlim(t_range)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_band_distribution(
    data: dict[str, pd.DataFrame],
    db_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """t-band 比率の grouped bar chart."""
    band_names = [b[0] for b in T_BANDS]
    db_dist = compute_band_distribution(db_df)

    x = np.arange(len(band_names))
    width = 0.18
    n_patterns = len(PATTERNS) + 1  # +1 for DB

    fig, ax = plt.subplots(figsize=(14, 6))

    # DB reference bars
    ax.bar(x - width * 2, db_dist * 100, width, color=DB_COLOR,
           alpha=0.7, label="DB all (reference)")

    for i, pattern in enumerate(PATTERNS):
        dist = compute_band_distribution(data[pattern])
        offset = width * (i - 1)
        ax.bar(x + offset, dist * 100, width, color=COLORS[pattern],
               alpha=0.85, label=f"Pattern {pattern}")

    ax.set_title("t-band Distribution: Patterns vs DB Reference",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t band")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _enrich_with_db(
    manifest_df: pd.DataFrame,
    db_df: pd.DataFrame,
) -> pd.DataFrame:
    """Manifest に DB の追加カラム (eye_scale, face_angle, lap_var) を JOIN."""
    extra_cols = ["idx", "eye_scale", "norm_eye_scale",
                  "face_yaw", "face_pitch", "face_roll", "lap_var"]
    # DB に該当カラムがない場合はスキップ
    available = [c for c in extra_cols if c in db_df.columns]
    return manifest_df.merge(db_df[available], on="idx", how="left")


def plot_image_characteristics(
    data: dict[str, pd.DataFrame],
    db_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """画像特性 (eye_scale, face_yaw, lap_var) の分布比較."""
    metrics = [
        ("eye_scale", "Eye Scale (px)"),
        ("norm_eye_scale", "Normalized Eye Scale"),
        ("face_yaw", "Face Yaw (°)"),
        ("face_pitch", "Face Pitch (°)"),
        ("face_roll", "Face Roll (°)"),
        ("lap_var", "Laplacian Variance (blur)"),
    ]

    # Enrich manifest data with DB columns
    enriched: dict[str, pd.DataFrame] = {}
    for p in PATTERNS:
        enriched[p] = _enrich_with_db(data[p], db_df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        if col not in db_df.columns:
            ax.set_title(f"{title}\n(column not available)")
            continue

        # DB reference distribution (per-image median)
        db_vals = db_df.groupby("image_path")[col].median().dropna().values
        if len(db_vals) > 0:
            ax.hist(db_vals, bins=40, density=True, alpha=0.3,
                    color=DB_COLOR, label="DB all")

        for pattern in PATTERNS:
            vals = enriched[pattern].groupby("image_path")[col].median().dropna().values
            if len(vals) < 3:
                continue
            kde = gaussian_kde(vals)
            lo, hi = np.percentile(db_vals, [1, 99]) if len(db_vals) > 0 else (vals.min(), vals.max())
            x = np.linspace(lo, hi, 300)
            ax.plot(x, kde(x), color=COLORS[pattern], linewidth=1.5,
                    label=f"Pattern {pattern}")

        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplot
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle("Image Characteristic Distributions by Pattern",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_heatmap(
    data: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """パターン間 eye-group 重複ヒートマップ."""
    patterns = PATTERNS
    n = len(patterns)

    # eye-group sets
    eg_sets: dict[str, set[tuple]] = {}
    for p in patterns:
        eg_sets[p] = set(
            (image_path, face_index, part_name)
            for (image_path, face_index, part_name), _
            in data[p].groupby(["image_path", "face_index", "part_name"])
        )

    matrix = np.zeros((n, n), dtype=int)
    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            matrix[i, j] = len(eg_sets[p1] & eg_sets[p2])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=350)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(patterns)
    ax.set_yticklabels(patterns)
    ax.set_title("Eye-group Overlap Matrix", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            text_color = "white" if matrix[i, j] > 175 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.85, label="Overlapping eye-groups")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Report generation
# ===================================================================

def generate_report(
    data: dict[str, pd.DataFrame],
    db_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Markdown レポートを生成."""
    band_names = [b[0] for b in T_BANDS]
    db_dist = compute_band_distribution(db_df)

    lines: list[str] = []
    def w(text: str = "") -> None:
        lines.append(text)

    w("# EXP029 Fixed Test Manifests — EDA Report")
    w()
    w(f"**生成日**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**データソース**: `data/samples.db` (skip_flag=0, {len(db_df):,} rows)")
    w()

    # ------- 1. 基本統計量 -------
    w("## 1. 基本統計量")
    w()
    w("| Pattern | Description | Test rows | Eye-groups | Images | eg/img | Train rows |")
    w("|---------|-------------|-----------|------------|--------|--------|------------|")
    for p in PATTERNS:
        df = data[p]
        eg = count_eye_groups(df)
        imgs = count_images(df)
        train_rows = len(db_df) - len(df)
        w(f"| **{p}** | {PATTERN_LABELS[p]} | {len(df):,} | {eg} | {imgs} | {eg/imgs:.2f} | {train_rows:,} |")
    w()

    # part_name / sub_key 分布
    w("### part_name / sub_key 分布")
    w()
    w("| Pattern | eyeliner_left | eyeliner_right | upper | lower |")
    w("|---------|---------------|----------------|-------|-------|")
    for p in PATTERNS:
        df = data[p]
        eg_df = df.groupby(["image_path", "face_index", "part_name", "sub_key"]).size().reset_index()
        n_left = (eg_df["part_name"] == "eyeliner_left").sum()
        n_right = (eg_df["part_name"] == "eyeliner_right").sum()
        n_upper = (eg_df["sub_key"] == "upper").sum()
        n_lower = (eg_df["sub_key"] == "lower").sum()
        w(f"| {p} | {n_left} | {n_right} | {n_upper} | {n_lower} |")
    w()

    # ------- 2. t 値分布 -------
    w("## 2. t 値分布の比較")
    w()
    w("### t-band 比率")
    w()
    header = "| Pattern | " + " | ".join(band_names) + " | KL(pat||DB) | L2(pat,DB) |"
    w(header)
    w("|" + "|".join(["---------"] * (len(band_names) + 3)) + "|")
    for p in PATTERNS:
        dist = compute_band_distribution(data[p])
        kl = kl_divergence(dist, db_dist)
        l2 = l2_distance(dist, db_dist)
        vals = " | ".join(f"{v*100:.2f}%" for v in dist)
        w(f"| **{p}** | {vals} | {kl:.4f} | {l2:.6f} |")
    # DB reference
    db_vals = " | ".join(f"{v*100:.2f}%" for v in db_dist)
    w(f"| *DB all* | {db_vals} | — | — |")
    w()

    w("![t-KDE overlay](eda_t_kde_overlay.png)")
    w()
    w("![t-band bars](eda_band_distribution.png)")
    w()

    # ------- 3. 画像特性 -------
    w("## 3. 画像特性バイアス")
    w()
    w("各パターンのテスト画像が特定の画像特性（目の大きさ、顔角度、ブレ）に偏っていないかを確認。")
    w()
    cols = ["eye_scale", "norm_eye_scale", "face_yaw", "face_pitch", "face_roll", "lap_var"]
    col_labels = ["Eye Scale", "Norm Eye Scale", "Face Yaw", "Face Pitch", "Face Roll", "Lap Var"]
    w("### per-image median の KS 検定 (各パターン vs DB)")
    w()
    w("| Pattern | " + " | ".join(col_labels) + " |")
    w("|" + "|".join(["---------"] * (len(cols) + 1)) + "|")

    from scipy.stats import ks_2samp

    # Enrich with DB for image characteristic columns
    enriched_data: dict[str, pd.DataFrame] = {}
    for p in PATTERNS:
        enriched_data[p] = _enrich_with_db(data[p], db_df)

    for p in PATTERNS:
        row = f"| **{p}**"
        for col in cols:
            if col not in db_df.columns:
                row += " | —"
                continue
            pat_vals = enriched_data[p].groupby("image_path")[col].median().dropna().values
            db_vals = db_df.groupby("image_path")[col].median().dropna().values
            if len(pat_vals) < 5 or len(db_vals) < 5:
                row += " | —"
            else:
                stat, pval = ks_2samp(pat_vals, db_vals)
                if pval < 0.01:
                    row += f" | **p={pval:.3f}** ⚠"
                else:
                    row += f" | p={pval:.3f}"
        w(row)
    w()
    w("> ⚠ = p < 0.01（分布に有意差あり）。サンプル数が少ないため参考値。")
    w()

    w("![image characteristics](eda_image_characteristics.png)")
    w()

    # ------- 4. パターン間比較 -------
    w("## 4. パターン間比較")
    w()
    w("### Eye-group 重複行列")
    w()

    eg_sets: dict[str, set[tuple]] = {}
    for p in PATTERNS:
        eg_sets[p] = set(
            (image_path, face_index, part_name)
            for (image_path, face_index, part_name), _
            in data[p].groupby(["image_path", "face_index", "part_name"])
        )

    # Overlap matrix
    w("| | " + " | ".join(PATTERNS) + " | Unique |")
    w("|" + "|".join(["---"] * (len(PATTERNS) + 2)) + "|")
    union_all = set()
    for p1 in PATTERNS:
        row = f"| **{p1}**"
        s1 = eg_sets[p1]
        union_all |= s1
        unique_in_p1 = len(s1)
        for p2 in PATTERNS:
            if p1 == p2:
                row += f" | {unique_in_p1}"
            else:
                row += f" | {len(s1 & eg_sets[p2])}"
        # Count eye-groups unique to this pattern (not in any other)
        others = set().union(*(eg_sets[p2] for p2 in PATTERNS if p2 != p1))
        unique_count = len(s1 - others)
        row += f" | {unique_count}"
        w(row)
    w(f"| *Union* | | | | | | {len(union_all)} |")
    w()

    w("![overlap heatmap](eda_overlap_heatmap.png)")
    w()

    # ------- 5. 各パターン詳細 -------
    w("## 5. 各パターン詳細プロット")
    w()
    for p in PATTERNS:
        w(f"### Pattern {p}")
        w(f"![t-dist {p}](eda_t_dist_{p}.png)")
        w()

    w("---")
    w("*Generated by `scr/eda_manifests.py`*")

    # Write report
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("Loading manifests...")
    data: dict[str, pd.DataFrame] = {}
    for p in PATTERNS:
        data[p] = load_manifest(p)
        eg = count_eye_groups(data[p])
        imgs = count_images(data[p])
        print(f"  Pattern {p}: {len(data[p])} rows, {eg} eye-groups, {imgs} images")

    print("\nLoading DB reference...")
    db_df = load_db_reference()
    print(f"  DB: {len(db_df)} rows, {count_images(db_df)} images")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Individual t-dist plots ---
    print("\nGenerating per-pattern t-distribution plots...")
    for p in PATTERNS:
        fig, ax = plt.subplots(figsize=(10, 5))
        t_range = (0.0, 1.2)
        x = np.linspace(t_range[0], t_range[1], 500)

        db_t = db_df["t"].values.astype(np.float64)
        db_t_clipped = db_t[(db_t >= t_range[0]) & (db_t <= t_range[1])]
        db_kde = gaussian_kde(db_t_clipped)

        t_vals = data[p]["t"].values.astype(np.float64)
        ax.hist(t_vals, bins=40, range=t_range, density=True,
                alpha=0.5, color=COLORS[p],
                label=f"Pattern {p} (n={len(t_vals)})")
        ax.plot(x, db_kde(x), color=DB_COLOR, linewidth=2, linestyle="--",
                label=f"DB all (n={len(db_t_clipped)})")
        ax.set_title(f"Pattern {p}: {PATTERN_LABELS[p]}")
        ax.set_xlabel("t value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_xlim(t_range)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / f"eda_t_dist_{p}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Combined plots ---
    print("Generating t-KDE overlay...")
    plot_t_distribution_overlay(data, db_df, _OUTPUT_DIR / "eda_t_kde_overlay.png")

    print("Generating band distribution chart...")
    plot_band_distribution(data, db_df, _OUTPUT_DIR / "eda_band_distribution.png")

    print("Generating image characteristics plot...")
    plot_image_characteristics(data, db_df, _OUTPUT_DIR / "eda_image_characteristics.png")

    print("Generating overlap heatmap...")
    plot_overlap_heatmap(data, _OUTPUT_DIR / "eda_overlap_heatmap.png")

    # --- Report ---
    print("Generating report...")
    generate_report(data, db_df, _OUTPUT_DIR / "eda_report.md")

    print(f"\nDone! Output → {_OUTPUT_DIR}")
    print("  eda_report.md")
    for p in PATTERNS:
        print(f"  eda_t_dist_{p}.png")
    print("  eda_t_kde_overlay.png")
    print("  eda_band_distribution.png")
    print("  eda_image_characteristics.png")
    print("  eda_overlap_heatmap.png")


if __name__ == "__main__":
    main()

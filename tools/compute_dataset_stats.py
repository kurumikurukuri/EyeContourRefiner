"""
データセット固有の正規化統計を計算し、ImageNet統計と比較するスクリプト。

出力:
  - コンソール: 統計値の比較表
  - tasks/research/YYYY-MM-DD_dataset_normalization_stats.json : 計算結果
  - tasks/research/YYYY-MM-DD_dataset_normalization_stats.png  : ヒストグラム比較図
"""

from __future__ import annotations
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.patch_utils import IMAGENET_MEAN, IMAGENET_STD, extract_patch


def load_manifest(path: Path) -> List[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                float(row["t"])
                records.append(row)
            except (ValueError, KeyError):
                continue
    return records


def collect_patches(records: List[dict], max_patches: int | None = None) -> np.ndarray:
    """全レコードからパッチを抽出し、(N, H, W, 3) float32 配列で返す。

    max_patches が指定された場合はランダムサブサンプリングする。
    """
    if max_patches is not None and len(records) > max_patches:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(records), max_patches, replace=False)
        records = [records[i] for i in indices]

    image_cache: dict[str, np.ndarray] = {}
    patches: list[np.ndarray] = []
    missing = 0

    for rec in records:
        path = rec["image_path"]
        if path not in image_cache:
            bgr = cv2.imread(path)
            if bgr is None:
                missing += 1
                continue
            image_cache[path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image = image_cache.get(path)
        if image is None:
            continue

        patch = extract_patch(
            image,
            float(rec["px"]), float(rec["py"]),
            float(rec["qx"]), float(rec["qy"]),
        )
        patches.append(patch)

    if missing:
        print(f"  警告: {missing} 枚の画像が読み込めませんでした")

    return np.stack(patches, axis=0)  # (N, H, W, 3)


def compute_stats(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """パッチ配列からチャンネル別 mean・std を計算する。

    patches: (N, H, W, 3) float32, values in [0, 1].
    Returns: mean (3,), std (3,) — チャンネル順 R, G, B.
    """
    # (N, H, W, 3) → (N*H*W, 3)
    flat = patches.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    return mean, std


def print_comparison(dataset_mean, dataset_std):
    imagenet_mean = np.array(IMAGENET_MEAN)
    imagenet_std  = np.array(IMAGENET_STD)

    header = f"{'':>10}  {'R':>8}  {'G':>8}  {'B':>8}"
    sep = "-" * 45
    print(sep)
    print(header)
    print(sep)
    print(f"{'Dataset μ':>10}  {dataset_mean[0]:.4f}    {dataset_mean[1]:.4f}    {dataset_mean[2]:.4f}")
    print(f"{'ImageNet μ':>10}  {imagenet_mean[0]:.4f}    {imagenet_mean[1]:.4f}    {imagenet_mean[2]:.4f}")
    print(f"{'Δμ':>10}  {dataset_mean[0]-imagenet_mean[0]:+.4f}   {dataset_mean[1]-imagenet_mean[1]:+.4f}   {dataset_mean[2]-imagenet_mean[2]:+.4f}")
    print(sep)
    print(f"{'Dataset σ':>10}  {dataset_std[0]:.4f}    {dataset_std[1]:.4f}    {dataset_std[2]:.4f}")
    print(f"{'ImageNet σ':>10}  {imagenet_std[0]:.4f}    {imagenet_std[1]:.4f}    {imagenet_std[2]:.4f}")
    print(f"{'Δσ':>10}  {dataset_std[0]-imagenet_std[0]:+.4f}   {dataset_std[1]-imagenet_std[1]:+.4f}   {dataset_std[2]-imagenet_std[2]:+.4f}")
    print(sep)


def plot_histograms(patches: np.ndarray, dataset_mean, dataset_std, out_path: Path) -> None:
    """チャンネル別ピクセル値ヒストグラムと正規化後分布を描画する。"""
    imagenet_mean = np.array(IMAGENET_MEAN)
    imagenet_std  = np.array(IMAGENET_STD)

    channel_names = ["R", "G", "B"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    flat = patches.reshape(-1, 3)  # (N*H*W, 3)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Dataset Patch Statistics vs ImageNet Normalization", fontsize=14, y=1.01)

    for ch, (name, color) in enumerate(zip(channel_names, colors)):
        vals = flat[:, ch]

        # --- 左列: 生ピクセル値分布 ---
        ax = axes[ch, 0]
        ax.hist(vals, bins=100, color=color, alpha=0.7, density=True)
        ax.axvline(dataset_mean[ch],  color="black",  linestyle="--", label=f"Dataset μ={dataset_mean[ch]:.3f}")
        ax.axvline(imagenet_mean[ch], color="orange", linestyle="--", label=f"ImageNet μ={imagenet_mean[ch]:.3f}")
        ax.set_title(f"{name} channel — raw [0,1]")
        ax.set_xlabel("pixel value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

        # --- 中列: ImageNet正規化後 ---
        ax = axes[ch, 1]
        normed_in = (vals - imagenet_mean[ch]) / imagenet_std[ch]
        ax.hist(normed_in, bins=100, color=color, alpha=0.7, density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="μ=0 target")
        # 標準正規分布を重ねる
        x_range = np.linspace(normed_in.min(), normed_in.max(), 200)
        ax.plot(x_range, np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi),
                color="black", linewidth=1, linestyle=":", label="N(0,1)")
        ax.set_title(f"{name} — ImageNet normalized\nμ={normed_in.mean():.3f}, σ={normed_in.std():.3f}")
        ax.set_xlabel("normalized value")
        ax.legend(fontsize=8)

        # --- 右列: Dataset正規化後 ---
        ax = axes[ch, 2]
        normed_ds = (vals - dataset_mean[ch]) / dataset_std[ch]
        ax.hist(normed_ds, bins=100, color=color, alpha=0.7, density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="μ=0 target")
        ax.plot(x_range, np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi),
                color="black", linewidth=1, linestyle=":", label="N(0,1)")
        ax.set_title(f"{name} — Dataset normalized\nμ={normed_ds.mean():.3f}, σ={normed_ds.std():.3f}")
        ax.set_xlabel("normalized value")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ヒストグラム保存: {out_path}")


def plot_sample_patches(patches: np.ndarray, out_path: Path, n: int = 16) -> None:
    """ランダムサンプルパッチを並べて描画する。"""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(patches), min(n, len(patches)), replace=False)
    cols = 8
    rows = (len(idx) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.2))
    for i, ax in enumerate(axes.flat):
        if i < len(idx):
            patch = patches[idx[i]]  # (H, W, 3) float32
            ax.imshow(np.clip(patch, 0, 1))
        ax.axis("off")

    plt.suptitle(f"Sample patches (n={len(idx)})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  サンプルパッチ画像保存: {out_path}")


def main():
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "data" / "manifest.csv"
    out_dir = repo_root / "tasks" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    json_out   = out_dir / f"{today}_dataset_normalization_stats.json"
    hist_out   = out_dir / f"{today}_dataset_normalization_stats.png"
    sample_out = out_dir / f"{today}_dataset_sample_patches.png"

    print("=== データセット正規化統計計算 ===\n")
    print(f"マニフェスト: {manifest_path}")

    records = load_manifest(manifest_path)
    print(f"レコード数: {len(records)}")

    print("\nパッチ抽出中...")
    patches = collect_patches(records)
    n_patches = len(patches)
    print(f"抽出パッチ数: {n_patches}  形状: {patches.shape}")

    print("\n統計計算中...")
    dataset_mean, dataset_std = compute_stats(patches)

    print("\n--- 統計比較 ---")
    print_comparison(dataset_mean, dataset_std)

    # 分布の歪み度・尖度も出力
    flat = patches.reshape(-1, 3)
    from scipy import stats as scipy_stats
    print("\n--- 分布形状 ---")
    for ch, name in enumerate(["R", "G", "B"]):
        vals = flat[:, ch]
        sk = scipy_stats.skew(vals)
        ku = scipy_stats.kurtosis(vals)
        print(f"  {name}: min={vals.min():.3f}  max={vals.max():.3f}  "
              f"skew={sk:.3f}  kurtosis={ku:.3f}")

    # --- 可視化 ---
    print("\n可視化生成中...")
    plot_histograms(patches, dataset_mean, dataset_std, hist_out)
    plot_sample_patches(patches, sample_out)

    # --- 結果保存 ---
    results = {
        "n_patches": int(n_patches),
        "patch_size": list(patches.shape[1:3]),
        "dataset": {
            "mean": dataset_mean.tolist(),
            "std":  dataset_std.tolist(),
        },
        "imagenet": {
            "mean": IMAGENET_MEAN,
            "std":  IMAGENET_STD,
        },
        "delta": {
            "mean": (dataset_mean - np.array(IMAGENET_MEAN)).tolist(),
            "std":  (dataset_std  - np.array(IMAGENET_STD)).tolist(),
        },
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON保存: {json_out}")

    print("\n=== 完了 ===")
    print(f"Dataset mean : R={dataset_mean[0]:.4f}  G={dataset_mean[1]:.4f}  B={dataset_mean[2]:.4f}")
    print(f"Dataset std  : R={dataset_std[0]:.4f}  G={dataset_std[1]:.4f}  B={dataset_std[2]:.4f}")
    print("\n上記の値を train.py の --norm_type dataset で使用できます。")


if __name__ == "__main__":
    main()

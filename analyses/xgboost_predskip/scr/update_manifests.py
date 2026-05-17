"""既存マニフェストに新規列を差分追加する。

Split 割り当て（train/val/test/unlabeled）は変更しない。
以下の 3 ソースから新列を取得する:
  1. analyses/samples.csv     - サンプル単位の特徴量（idx でジョイン）
  2. image_features.parquet   - 目領域の局所画像特徴量（eye_gray_mean → min_eye_brightness 派生）
  3. data/images.db           - 画像単位の特徴量（将来 FaceMesh 実行時に使用）

使い方:
  # parquet を事前ビルドしてからマニフェストを更新
  python -m analyses.xgboost_predskip.scr.update_manifests --build-image-features

  # parquet がすでにある場合はそのまま更新
  python -m analyses.xgboost_predskip.scr.update_manifests

  # dry-run で追加予定列を確認
  python -m analyses.xgboost_predskip.scr.update_manifests --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent
SAMPLES_CSV = ROOT / "analyses" / "samples.csv"
REGISTRY_DB = ROOT / "data" / "images.db"

DEFAULT_MANIFESTS_DIR = Path("analyses/xgboost_predskip/manifests")
DEFAULT_IMAGE_CACHE = Path("analyses/xgboost_predskip/outputs/image_features.parquet")
OI_LABELS_PATH = ROOT / "data" / ".open_images_cache" / "oi_image_labels.csv"

# images.db から取り込む画像単位の列（FaceMesh実行済みの場合に有効）
IMAGE_DB_COLS = [
    "min_eye_brightness",
]

# Open Images ラベル名 → 特徴量カラム名
OI_TARGET_LABELS: dict[str, str] = {
    "/m/01bl7v": "is_boy",
    "/m/05r655": "is_girl",
    "/m/04yx4": "is_man",
    "/m/03bt1vf": "is_woman",
    "/m/01g317": "is_person",
    "/m/017ftj": "is_sunglasses",
    "/m/0jyfg": "is_glasses",
    "/m/02_n6y": "is_goggles",
}


# ---------------------------------------------------------------------------
# Source 1: samples.csv
# ---------------------------------------------------------------------------
def _load_samples_new_cols(mf: pd.DataFrame) -> pd.DataFrame | None:
    """samples.csv にあってマニフェストにない列を返す。"""
    if not SAMPLES_CSV.exists():
        log.warning("samples.csv が見つかりません: %s", SAMPLES_CSV)
        return None
    samples = pd.read_csv(SAMPLES_CSV)
    new_cols = [c for c in samples.columns if c != "idx" and c not in mf.columns]
    if not new_cols or "idx" not in mf.columns:
        return None
    return samples[["idx"] + new_cols]


# ---------------------------------------------------------------------------
# Source 2: image_features.parquet → min_eye_brightness
# ---------------------------------------------------------------------------
def _derive_brightness_from_parquet(cache_path: Path) -> pd.DataFrame | None:
    """image_features.parquet から min_eye_brightness を image_path 単位で集約する。"""
    if not cache_path.exists():
        log.info("image_features.parquet が見つかりません: %s", cache_path)
        return None
    parquet = pd.read_parquet(cache_path, columns=["image_path", "eye_gray_mean"])
    if "eye_gray_mean" not in parquet.columns:
        log.warning("parquet に eye_gray_mean 列がありません。")
        return None
    brightness = (
        parquet.groupby("image_path")["eye_gray_mean"]
        .min()
        .reset_index()
        .rename(columns={"eye_gray_mean": "min_eye_brightness"})
    )
    log.info("parquet から min_eye_brightness 導出: %d 画像", len(brightness))
    return brightness


def _build_image_features_parquet(
    manifests_dir: Path,
    cache_path: Path,
) -> bool:
    """image_features.parquet を事前ビルドする。すでに存在する場合はスキップ。"""
    if cache_path.exists():
        log.info("image_features.parquet は既存のためスキップ: %s", cache_path)
        return True
    try:
        from analyses.xgboost_baseline.scr.image_features import build_image_features
    except ImportError as e:
        log.error("image_features モジュールのインポートに失敗: %s", e)
        return False

    manifest_files = list(manifests_dir.glob("manifest_*.csv"))
    if not manifest_files:
        log.error("manifest ファイルが見つかりません: %s", manifests_dir)
        return False

    all_mf = pd.concat([pd.read_csv(p) for p in manifest_files], ignore_index=True)
    log.info("image features 計算開始: %d 行", len(all_mf))
    build_image_features(all_mf, cache_path=cache_path)
    return cache_path.exists()


# ---------------------------------------------------------------------------
# Source 3: Open Images image-level class labels
# ---------------------------------------------------------------------------
def _load_oi_class_features() -> pd.DataFrame | None:
    """oi_image_labels.csv から is_boy/girl/man/woman/person を画像単位で返す。"""
    if not OI_LABELS_PATH.exists():
        log.warning("oi_image_labels.csv が見つかりません: %s", OI_LABELS_PATH)
        return None
    labels = pd.read_csv(OI_LABELS_PATH)
    filtered = labels[labels["label_name"].isin(OI_TARGET_LABELS)].copy()
    filtered["feature"] = filtered["label_name"].map(OI_TARGET_LABELS)
    pivot = (
        filtered.groupby(["image_id", "feature"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
        .astype(int)
    )
    for feat in OI_TARGET_LABELS.values():
        if feat not in pivot.columns:
            pivot[feat] = 0
    pivot = pivot[sorted(OI_TARGET_LABELS.values())].reset_index()
    log.info("OI class features: %d 画像カバー", len(pivot))
    return pivot


# ---------------------------------------------------------------------------
# Source 4: images.db
# ---------------------------------------------------------------------------
def _load_image_db_features(exclude_cols: set[str]) -> pd.DataFrame | None:
    """images.db から exclude_cols にない列を読み込む。"""
    if not REGISTRY_DB.exists():
        return None
    conn = sqlite3.connect(str(REGISTRY_DB))
    available = {
        row[1]
        for row in conn.execute("PRAGMA table_info(images)").fetchall()
    }
    cols = [c for c in IMAGE_DB_COLS if c in available and c not in exclude_cols]
    if not cols:
        conn.close()
        return None
    df = pd.read_sql(f"SELECT image_path, {', '.join(cols)} FROM images", conn)
    conn.close()
    df = df.rename(columns={"image_path": "_image_basename"})
    return df


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------
def update_manifest(
    manifest_path: Path,
    samples_new: pd.DataFrame | None,
    brightness_df: pd.DataFrame | None,
    img_db: pd.DataFrame | None,
    oi_class_df: pd.DataFrame | None = None,
    dry_run: bool = False,
) -> int:
    mf = pd.read_csv(manifest_path)
    added: list[str] = []

    # 1. samples.csv 差分列
    if samples_new is not None:
        new_cols = [c for c in samples_new.columns if c != "idx" and c not in mf.columns]
        if new_cols:
            subset = samples_new[["idx"] + new_cols]
            if not dry_run:
                mf = mf.merge(subset, on="idx", how="left")
            added.extend(new_cols)
            log.info("  samples.csv 追加予定: %s", new_cols)

    # 2. min_eye_brightness（parquet由来）
    if brightness_df is not None and "min_eye_brightness" not in mf.columns:
        if not dry_run:
            mf = mf.merge(brightness_df, on="image_path", how="left")
        added.append("min_eye_brightness")
        log.info("  parquet 追加予定: min_eye_brightness")

    # 3. images.db 差分列（FaceMesh実行済み列のみ）
    if img_db is not None and "image_path" in mf.columns:
        new_img_cols = [
            c for c in img_db.columns
            if c != "_image_basename" and c not in mf.columns
        ]
        if new_img_cols:
            if not dry_run:
                mf["_image_basename"] = mf["image_path"].apply(lambda p: Path(p).name)
                mf = mf.merge(img_db[["_image_basename"] + new_img_cols], on="_image_basename", how="left")
                mf = mf.drop(columns=["_image_basename"])
            added.extend(new_img_cols)
            log.info("  images.db 追加予定: %s", new_img_cols)

    # 4. Open Images クラス特徴量
    if oi_class_df is not None and "image_path" in mf.columns:
        oi_feat_cols = list(OI_TARGET_LABELS.values())
        new_oi_cols = [c for c in oi_feat_cols if c not in mf.columns]
        if new_oi_cols:
            if not dry_run:
                mf["_image_id"] = mf["image_path"].apply(lambda p: Path(p).stem)
                subset = oi_class_df[["image_id"] + new_oi_cols].rename(columns={"image_id": "_image_id"})
                mf = mf.merge(subset, on="_image_id", how="left")
                mf = mf.drop(columns=["_image_id"])
                for col in new_oi_cols:
                    mf[col] = mf[col].fillna(0).astype(int)
            added.extend(new_oi_cols)
            log.info("  OI class 追加予定: %s", new_oi_cols)

    if not added:
        log.info("  %s: 追加列なし（スキップ）", manifest_path.name)
        return 0

    if dry_run:
        log.info("  [dry-run] %s: 追加予定 %d 列 → %s", manifest_path.name, len(added), added)
    else:
        mf.to_csv(manifest_path, index=False)
        log.info("  %s: %d 列追加 → 計 %d 列", manifest_path.name, len(added), len(mf.columns))
    return len(added)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="マニフェストに新規列を差分追加する")
    parser.add_argument(
        "--manifests-dir", type=Path, default=DEFAULT_MANIFESTS_DIR,
    )
    parser.add_argument(
        "--image-cache-path", type=Path, default=DEFAULT_IMAGE_CACHE,
    )
    parser.add_argument(
        "--build-image-features", action="store_true",
        help="image_features.parquet を事前ビルドしてから更新する",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests_dir = args.manifests_dir
    cache_path = args.image_cache_path

    if not manifests_dir.exists():
        log.error("manifests_dir が見つかりません: %s", manifests_dir)
        sys.exit(1)

    manifest_files = sorted(manifests_dir.glob("manifest_*.csv"))
    if not manifest_files:
        log.error("manifest ファイルが見つかりません: %s", manifests_dir)
        sys.exit(1)

    # --- parquet ビルド ---
    if args.build_image_features:
        ok = _build_image_features_parquet(manifests_dir, cache_path)
        if not ok:
            log.error("image_features.parquet のビルドに失敗しました。")
            sys.exit(1)

    # --- 各ソースをロード ---
    # samples.csv は最初のマニフェストで差分確認
    first_mf = pd.read_csv(manifest_files[0])
    samples_new = _load_samples_new_cols(first_mf)
    if samples_new is not None:
        log.info("samples.csv 新規列: %s", [c for c in samples_new.columns if c != "idx"])

    brightness_df = _derive_brightness_from_parquet(cache_path)

    # images.db は parquet で補完できる列を除外
    parquet_provided = {"min_eye_brightness"} if brightness_df is not None else set()
    img_db = _load_image_db_features(exclude_cols=parquet_provided)

    # OI class features
    oi_class_df = _load_oi_class_features()

    # --- 各マニフェストを更新 ---
    for mf_path in manifest_files:
        log.info("処理中: %s", mf_path.name)
        update_manifest(mf_path, samples_new, brightness_df, img_db, oi_class_df, dry_run=args.dry_run)

    action = "[dry-run] 確認完了" if args.dry_run else "完了"
    log.info("%s: %d マニフェスト処理", action, len(manifest_files))


if __name__ == "__main__":
    main()

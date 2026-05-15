"""
DB の skip_flag / t (true_t_raw) を CSV に反映する。

通常運用: python tools/update_csv_from_db.py
  → skip_flag / true_t_raw のみ更新（高速）

初回のみ (--add-image-features):
  - images.db → min_eye_brightness を image_path 単位でマージ
  - Open Images ラベル → is_boy/girl/man/woman/person/sunglasses/glasses/goggles
  - niqe_score が NaN の行について NIQE を再計算（pyiqa/torch 必要）
"""

import argparse
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/samples.db"
DEFAULT_CSV = "data/samples.csv"
DEFAULT_IMAGES_DB = "data/images.db"
OI_LABELS_PATH = Path("data") / ".open_images_cache" / "oi_image_labels.csv"

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


def _load_min_eye_brightness(images_db_path: str) -> pd.DataFrame | None:
    if not Path(images_db_path).exists():
        logger.warning("images.db が見つかりません: %s", images_db_path)
        return None
    conn = sqlite3.connect(images_db_path)
    available = {row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()}
    if "min_eye_brightness" not in available:
        conn.close()
        return None
    df = pd.read_sql("SELECT image_path, min_eye_brightness FROM images", conn)
    conn.close()
    # images.db はファイル名のみ、samples.csv はフルパス → basename で正規化
    df["image_path"] = df["image_path"].apply(lambda p: Path(p).name)
    logger.info("images.db: min_eye_brightness %d 画像", len(df))
    return df


def _load_oi_class_features() -> pd.DataFrame | None:
    if not OI_LABELS_PATH.exists():
        logger.warning("oi_image_labels.csv が見つかりません: %s", OI_LABELS_PATH)
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
    logger.info("OI class features: %d 画像カバー", len(pivot))
    return pivot


def _compute_niqe_for_nulls(csv_df: pd.DataFrame) -> pd.DataFrame:
    """niqe_score が NaN の行のみ NIQE を計算して埋める。"""
    if "niqe_score" not in csv_df.columns:
        logger.info("niqe_score 列が存在しないためスキップ")
        return csv_df

    null_mask = csv_df["niqe_score"].isna()
    n_null = int(null_mask.sum())
    if n_null == 0:
        logger.info("niqe_score: 全行に値あり、スキップ")
        return csv_df

    logger.info("NIQE 計算開始: %d 行 (NaN のみ)...", n_null)

    try:
        import torch
        import pyiqa
        from skimage import io as sk_io
        from skimage.transform import resize as sk_resize
        from skimage.util import img_as_float
    except ImportError as e:
        logger.warning("NIQE に必要なライブラリが見つかりません（スキップ）: %s", e)
        return csv_df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    niqe_metric = pyiqa.create_metric("niqe", device=device)

    null_df = csv_df[null_mask]

    # (image_path, face_index, part_name) 単位で eye bbox を計算
    key_cols = ["image_path", "face_index", "part_name"]
    grp_px = null_df.groupby(key_cols).agg(px_min=("px", "min"), px_max=("px", "max"),
                                             py_min=("py", "min"), py_max=("py", "max"))
    grp_qx = null_df.groupby(key_cols).agg(qx_min=("qx", "min"), qx_max=("qx", "max"),
                                             qy_min=("qy", "min"), qy_max=("qy", "max"))
    bbox_df = grp_px.join(grp_qx)
    bbox_df["x_min"] = bbox_df[["px_min", "qx_min"]].min(axis=1)
    bbox_df["x_max"] = bbox_df[["px_max", "qx_max"]].max(axis=1)
    bbox_df["y_min"] = bbox_df[["py_min", "qy_min"]].min(axis=1)
    bbox_df["y_max"] = bbox_df[["py_max", "qy_max"]].max(axis=1)
    bbox_df["w"] = bbox_df["x_max"] - bbox_df["x_min"]
    bbox_df["h"] = bbox_df["y_max"] - bbox_df["y_min"]
    eye_bbox_map = bbox_df[["x_min", "x_max", "y_min", "y_max", "w", "h"]].to_dict("index")

    def _niqe_from_crop(img_uint8: np.ndarray, bbox: dict, margin: float = 0.20) -> float | None:
        img_h, img_w = img_uint8.shape[:2]
        mx, my = bbox["w"] * margin, bbox["h"] * margin
        x1 = max(int(math.floor(bbox["x_min"] - mx)), 0)
        y1 = max(int(math.floor(bbox["y_min"] - my)), 0)
        x2 = min(int(math.ceil(bbox["x_max"] + mx)), img_w)
        y2 = min(int(math.ceil(bbox["y_max"] + my)), img_h)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img_as_float(img_uint8[y1:y2, x1:x2]).astype("float32")
        if crop.size == 0:
            return None
        if crop.ndim == 2:
            crop = np.stack([crop, crop, crop], axis=-1)
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = crop[..., :3]
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 224:
            scale = 224 / min(ch, cw)
            crop = sk_resize(crop, (max(1, int(ch * scale)), max(1, int(cw * scale))),
                             order=3, preserve_range=True, anti_aliasing=True).astype("float32")
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        with torch.no_grad():
            score = niqe_metric(tensor)
        return float(score.item() if torch.is_tensor(score) else score)

    # 画像ごとにグループ化して 1 枚ずつ読み込み
    grouped: dict = defaultdict(list)
    for key in eye_bbox_map:
        grouped[key[0]].append((key[1], key[2]))

    niqe_scores: dict = {}
    total = len(grouped)
    for i, (image_path, entries) in enumerate(grouped.items()):
        if i % 200 == 0:
            logger.info("  NIQE %d/%d 画像...", i, total)
        try:
            img = sk_io.imread(image_path)
        except Exception:
            for face_index, part_name in entries:
                niqe_scores[(image_path, face_index, part_name)] = None
            continue
        for face_index, part_name in entries:
            bbox = eye_bbox_map[(image_path, face_index, part_name)]
            try:
                niqe_scores[(image_path, face_index, part_name)] = _niqe_from_crop(img, bbox)
            except Exception as e:
                logger.warning("NIQE failed %s: %s", image_path, e)
                niqe_scores[(image_path, face_index, part_name)] = None

    # DataFrame にマージして反映
    score_rows = [
        {"image_path": k[0], "face_index": k[1], "part_name": k[2], "_niqe_new": v}
        for k, v in niqe_scores.items()
    ]
    if not score_rows:
        return csv_df
    score_df = pd.DataFrame(score_rows)
    out = csv_df.merge(score_df, on=key_cols, how="left")
    update_mask = out["_niqe_new"].notna()
    out.loc[update_mask, "niqe_score"] = out.loc[update_mask, "_niqe_new"]
    out = out.drop(columns=["_niqe_new"])
    n_filled = int(update_mask.sum())
    logger.info("NIQE 計算完了: %d 行に値を追加", n_filled)
    return out


def update_csv_from_db(
    csv_path: str,
    db_path: str,
    columns: tuple[str, ...] = ("skip_flag", "t"),
    add_image_features: bool = False,
    images_db_path: str = DEFAULT_IMAGES_DB,
) -> pd.DataFrame:
    csv_df = pd.read_csv(csv_path)
    logger.info("CSV loaded: %s (%d rows)", csv_path, len(csv_df))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT idx, {', '.join(columns)} FROM samples")
    db_rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    db_df = pd.DataFrame(db_rows).set_index("idx")
    logger.info("DB loaded: %s (%d rows)", db_path, len(db_df))

    changes = {}
    col_map = {"t": "true_t_raw"}

    for col in columns:
        csv_col = col_map.get(col, col)
        old = csv_df[csv_col].copy()
        new = csv_df["idx"].map(db_df[col])

        if col == "t":
            old_num = pd.to_numeric(old, errors="coerce")
            new_num = pd.to_numeric(new, errors="coerce")
            old_cmp = old_num.fillna(-999.0)
            new_cmp = new_num.fillna(-999.0)
            new = new.where(new.notna(), "")
        else:
            old_cmp = old.fillna("__NA__")
            new_cmp = new.fillna("__NA__")

        changed = (~np.isclose(old_cmp, new_cmp, rtol=1e-14, atol=1e-14)).sum() if col == "t" else (old_cmp != new_cmp).sum()
        changes[col] = changed
        csv_df[csv_col] = new

    if add_image_features:
        brightness_df = _load_min_eye_brightness(images_db_path)
        already_has = "min_eye_brightness" in csv_df.columns
        if brightness_df is not None and (not already_has or csv_df["min_eye_brightness"].notna().sum() == 0):
            if already_has:
                csv_df = csv_df.drop(columns=["min_eye_brightness"])
            csv_df["_img_basename"] = csv_df["image_path"].apply(lambda p: Path(p).name)
            brightness_df = brightness_df.rename(columns={"image_path": "_img_basename"})
            csv_df = csv_df.merge(brightness_df, on="_img_basename", how="left")
            csv_df = csv_df.drop(columns=["_img_basename"])
            n_filled = csv_df["min_eye_brightness"].notna().sum()
            logger.info("min_eye_brightness 追加: %d/%d 行に値あり", n_filled, len(csv_df))

        oi_df = _load_oi_class_features()
        if oi_df is not None:
            new_oi_cols = [c for c in oi_df.columns if c != "image_id" and c not in csv_df.columns]
            if new_oi_cols:
                csv_df["_image_id"] = csv_df["image_path"].apply(lambda p: Path(p).stem)
                csv_df = csv_df.merge(
                    oi_df[["image_id"] + new_oi_cols].rename(columns={"image_id": "_image_id"}),
                    on="_image_id", how="left",
                )
                csv_df = csv_df.drop(columns=["_image_id"])
                for col in new_oi_cols:
                    csv_df[col] = csv_df[col].fillna(0).astype(int)
                logger.info("OI class features 追加: %s", new_oi_cols)

        csv_df = _compute_niqe_for_nulls(csv_df)

    csv_df.to_csv(csv_path, index=False)
    logger.info("Updated CSV written: %s", csv_path)

    for col, n in changes.items():
        logger.info("  %s: %d rows changed", col, n)

    return csv_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="DB から CSV へ skip_flag / t を反映 + 画像特徴量マージ"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"DB path (default: {DEFAULT_DB})")
    parser.add_argument(
        "--add-image-features", action="store_true",
        help="min_eye_brightness / OI class features を新規マージ（初回のみ必要）",
    )
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.db).exists():
        print(f"[ERROR] DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    update_csv_from_db(
        args.csv, args.db,
        add_image_features=args.add_image_features,
    )


if __name__ == "__main__":
    main()

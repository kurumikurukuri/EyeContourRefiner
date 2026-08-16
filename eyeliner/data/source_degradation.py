"""元画像ピクセル空間で適用する劣化オーグメンテーション（EXP031）。

パッチ空間ではなく **ワープ前の元画像 ROI** に劣化を掛ける。パッチは元画像を
1.4〜3.5 倍に補間拡大したもの（`analyses/aug_design_audit/summary.md` 事実1）なので、
パッチ空間で JPEG や解像度劣化を模擬しても元画像で実際に起きる劣化とは一致しない。

実装する劣化は 2 種:

    JPEG 圧縮    : ROI を quality q で encode→decode する
    解像度ジッタ  : ROI を 1/s に縮小（INTER_AREA）してから元サイズへ拡大（INTER_LINEAR）

両方有効な場合は **縮小 → 縮小サイズで JPEG → 拡大** の順で適用する。実世界では
低解像度で撮影された画像がその解像度のまま JPEG 保存されるため、この順序が物理的に忠実。

ROI の原点は 8 の倍数に切り下げる。元画像の JPEG 8×8 ブロック格子に合わせることで、
切り出し位置によってブロック境界の位置が変わるのを防ぐ。
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ROI 原点のアライン単位（元画像の JPEG 8×8 ブロック格子に合わせる）。
JPEG_BLOCK_PX: int = 8
# ROI に付けるマージン [px]。extract_patch の双線形補間が参照する近傍を含める。
ROI_MARGIN_PX: int = 8
# JPEG に現実的な文脈を与えるための ROI 最小サイズ [px]。
MIN_ROI_PX: int = 64


@dataclass(frozen=True)
class SourceDegradeConfig:
    """元画像空間の劣化 aug 設定。

    jpeg / res_jitter がいずれも False の場合、劣化は一切適用されない
    （`enabled` が False を返し、呼び出し側で dataset に渡されない）。
    """

    jpeg: bool = False
    jpeg_prob: float = 0.5
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 90
    res_jitter: bool = False
    res_jitter_prob: float = 0.5
    res_scale_min: float = 1.0
    res_scale_max: float = 2.5

    @property
    def enabled(self) -> bool:
        return self.jpeg or self.res_jitter

    def validate(self) -> None:
        """設定値の妥当性を検証する。不正なら ValueError を送出する。"""
        if not 0.0 <= self.jpeg_prob <= 1.0:
            raise ValueError(f"jpeg_prob は [0, 1] であること: {self.jpeg_prob}")
        if not 0.0 <= self.res_jitter_prob <= 1.0:
            raise ValueError(f"res_jitter_prob は [0, 1] であること: {self.res_jitter_prob}")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError(
                "jpeg quality は 1 <= min <= max <= 100 であること: "
                f"min={self.jpeg_quality_min} max={self.jpeg_quality_max}"
            )
        if not 1.0 <= self.res_scale_min <= self.res_scale_max:
            raise ValueError(
                "res_scale は 1.0 <= min <= max であること: "
                f"min={self.res_scale_min} max={self.res_scale_max}"
            )


@dataclass(frozen=True)
class SourceDegradeSample:
    """1 サンプル分の劣化パラメータ。枝A・枝B で同一のものを共有する。

    None は「その劣化を適用しない」を意味する。
    """

    jpeg_quality: int | None = None
    res_scale: float | None = None

    @property
    def is_identity(self) -> bool:
        return self.jpeg_quality is None and self.res_scale is None


def sample_source_degradation(
    cfg: SourceDegradeConfig,
    rng: random.Random,
) -> SourceDegradeSample:
    """設定に従って 1 サンプル分の劣化パラメータを引く。

    Args:
        cfg: 劣化設定。
        rng: 乱数生成器（dataset が seed から生成した専用インスタンス）。

    Returns:
        SourceDegradeSample（劣化なしの場合は両フィールドが None）。
    """
    quality: int | None = None
    scale: float | None = None

    if cfg.jpeg and rng.random() < cfg.jpeg_prob:
        quality = rng.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)

    if cfg.res_jitter and rng.random() < cfg.res_jitter_prob:
        drawn = rng.uniform(cfg.res_scale_min, cfg.res_scale_max)
        # s≈1.0 は恒等変換。無駄な resize 往復を避けるため None に潰す。
        scale = drawn if drawn > 1.0 + 1e-6 else None

    return SourceDegradeSample(jpeg_quality=quality, res_scale=scale)


def _expand_to_min(lo: int, hi: int, min_size: int) -> tuple[int, int]:
    """区間 [lo, hi) を中心を保ったまま min_size 以上に広げる。"""
    deficit = min_size - (hi - lo)
    if deficit <= 0:
        return lo, hi
    left = deficit // 2
    return lo - left, hi + (deficit - left)


def compute_patch_source_bbox(
    px: float, py: float,
    qx: float, qy: float,
    t_min: float, t_max: float,
    perp_scale: float,
    margin_px: int = ROI_MARGIN_PX,
) -> tuple[int, int, int, int] | None:
    """extract_patch がサンプルする元画像領域の bbox を返す。

    **画像境界でクリップしない**（座標が負や画像サイズ超過になりうる）。クリップと
    反射パディングは degrade_region() の責務で、そちらが元画像の反射境界を正確に
    再現する。ここでクリップしてしまうと、footprint が画像外へ出るサンプルで
    「全画像での BORDER_REFLECT_101」と「切り出し ROI での BORDER_REFLECT_101」が
    別物になり、パッチが大きく変わってしまう。

    返す bbox は必ずパッチの footprint を完全に含む。したがって劣化を適用しなければ
    bbox 経由でも実質同一のパッチが得られる。「実質」と書いたのは、座標を (x0, y0)
    だけ平行移動すると warpAffine のアフィン行列の並進成分が変わり、cv2 内部の
    固定小数点座標の丸めが変化するため（footprint 不足ではなく、切り出しを伴わない
    純粋な座標シフトでも生じうる差である）。

    なお学習時の dataset 経路では、劣化パラメータが恒等（is_identity）のとき
    呼び出し側が直接抽出パスへフォールバックするため、この丸め差は劣化が実際に
    掛かるサンプルにしか生じない。

    Args:
        px, py, qx, qy: 元画像座標の P, Q。
        t_min, t_max:   パッチが覆う t の範囲（BranchConfig.patch_t_min/max）。
        perp_scale:     縦方向カバレッジ倍率（BranchConfig.patch_perp_scale）。
        margin_px:      footprint に付けるマージン [px]（既定 ROI_MARGIN_PX）。

    Returns:
        (x0, y0, x1, y1) の整数タプル（クリップなし）。P==Q の場合は None。
    """
    dx, dy = qx - px, qy - py
    norm = math.hypot(dx, dy)
    if norm <= 0.0:
        return None

    ux, uy = dx / norm, dy / norm
    perp_x, perp_y = -uy, ux
    half = 0.5 * perp_scale * norm

    xs: list[float] = []
    ys: list[float] = []
    for t in (t_min, t_max):
        sx = t * px + (1.0 - t) * qx
        sy = t * py + (1.0 - t) * qy
        for sign in (-1.0, 1.0):
            xs.append(sx + sign * half * perp_x)
            ys.append(sy + sign * half * perp_y)

    x0 = math.floor(min(xs)) - margin_px
    x1 = math.ceil(max(xs)) + margin_px
    y0 = math.floor(min(ys)) - margin_px
    y1 = math.ceil(max(ys)) + margin_px

    x0, x1 = _expand_to_min(x0, x1, MIN_ROI_PX)
    y0, y1 = _expand_to_min(y0, y1, MIN_ROI_PX)

    # 原点を 8 の倍数へ切り下げ（元画像の JPEG ブロック格子に合わせる）。
    # Python の % は負数でも非負を返すため、負の座標でも floor 方向に揃う。
    x0 -= x0 % JPEG_BLOCK_PX
    y0 -= y0 % JPEG_BLOCK_PX

    return x0, y0, x1, y1


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    """RGB uint8 配列を JPEG で encode→decode して返す。失敗時は入力をそのまま返す。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        logger.warning("JPEG encode 失敗のため劣化をスキップ: quality=%d shape=%s", quality, rgb.shape)
        return rgb
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        logger.warning("JPEG decode 失敗のため劣化をスキップ: quality=%d shape=%s", quality, rgb.shape)
        return rgb
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def degrade_region(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    sample: SourceDegradeSample,
) -> tuple[np.ndarray, int, int] | None:
    """bbox の領域を切り出して劣化を適用し、(劣化後 ROI, x0, y0) を返す。

    bbox が画像外へはみ出す場合は、はみ出した分を `BORDER_REFLECT_101` で
    パディングして補う。パディングが必要になる辺では切り出し境界が画像境界と
    一致しているため、これは extract_patch が全画像に対して行う反射と厳密に等価。
    こうしないと、目が画像端に近いサンプルでパッチが大きく変わってしまう。

    `image` は一切変更しない（cv2 の各関数は新しい配列を返す）。

    Args:
        image:  元画像 RGB uint8 (H, W, 3)。
        bbox:   compute_patch_source_bbox() の戻り値（クリップなし）。
        sample: 適用する劣化パラメータ。

    Returns:
        (ROI 配列, x0, y0)。呼び出し側は座標から (x0, y0) を引いて extract_patch に渡す。
        ROI が画像とまったく重ならない場合や、反射パディングが成立しないほど
        画像外へ出ている場合は None（呼び出し側は劣化なしの直接抽出にフォールバックする）。
    """
    x0, y0, x1, y1 = bbox
    img_h, img_w = int(image.shape[0]), int(image.shape[1])

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(img_w, x1), min(img_h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    crop = image[cy0:cy1, cx0:cx1]
    left, top = cx0 - x0, cy0 - y0
    right, bottom = x1 - cx1, y1 - cy1
    if left or top or right or bottom:
        crop_h, crop_w = crop.shape[:2]
        # BORDER_REFLECT_101 は border < size を要求する。
        if max(left, right) >= crop_w or max(top, bottom) >= crop_h:
            return None
        crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101,
        )

    if sample.is_identity:
        return crop, x0, y0

    roi_h, roi_w = crop.shape[:2]
    work = crop

    if sample.res_scale is not None:
        small_w = max(1, int(round(roi_w / sample.res_scale)))
        small_h = max(1, int(round(roi_h / sample.res_scale)))
        work = cv2.resize(work, (small_w, small_h), interpolation=cv2.INTER_AREA)

    if sample.jpeg_quality is not None:
        work = _jpeg_roundtrip(work, sample.jpeg_quality)

    if work.shape[0] != roi_h or work.shape[1] != roi_w:
        work = cv2.resize(work, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    return work, x0, y0

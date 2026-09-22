"""画像ヘッダからサイズだけを読む共通処理（Pillow）。画像本体は展開しない。

norm_eye_scale の算出（rebuild_samples_db）と、巨大な画像を FaceMesh に渡す前の検査
（restore_face_pose）で共有する。Pillow が無くても他の復元は動くよう、import は任意扱い。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class ImageSizeError(Exception):
    """画像ヘッダからサイズを読めなかった（壊れている・巨大すぎる等）。"""


def pillow_available() -> bool:
    return Image is not None


def read_image_size(path: Path) -> Tuple[int, int]:
    """画像の (幅, 高さ) を返す。

    Raises:
        ImageSizeError: Pillow が無い / 画像として読めない / 展開爆弾と判定された。
    """
    if Image is None:
        raise ImageSizeError("Pillow がありません（pip install pillow）")
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageSizeError(str(exc)) from exc

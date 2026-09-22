"""復元の基準となる実験 manifest の検出・検証。

experiments/EXP{NNN}_*/manifests/manifest_{train,test}.csv のうち最新の実験番号のものを
自動検出し、samples.db を再構築できる列が揃っているかを境界で検証する。

manifest の `image_path` は外部入力として扱う。`data/images/<ID>.jpg`（ID は英数字・`_`・`-`）
以外は拒否し、任意パスへのアクセスや別ディレクトリへの書き込みに使われないようにする。

EXP001〜007 の manifest は `face_index` を持たないため、この復元処理では利用できない。
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

REQUIRED_COLUMNS: tuple[str, ...] = (
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("nose_x", "nose_y")

VALID_PART_NAMES: frozenset[str] = frozenset({"eyeliner_left", "eyeliner_right"})
VALID_SUB_KEYS: frozenset[str] = frozenset({"upper", "lower"})
POINTS_PER_SUB_KEY = 7

MANIFEST_FILENAMES: tuple[str, ...] = ("manifest_train.csv", "manifest_test.csv")

_IMAGE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_IMAGE_PATH_RE = re.compile(r"data/images/([A-Za-z0-9_-]+)\.jpg")
_EXP_DIR_RE = re.compile(r"EXP(\d+)_")
_EXP_ARG_RE = re.compile(r"(?:EXP)?(\d+)")


class ManifestError(ValueError):
    """manifest の検出・検証エラー。"""


@dataclass(frozen=True)
class ManifestSource:
    """検出した実験の manifest 一式。"""

    exp_id: int
    exp_dir: Path
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ManifestRow:
    """検証済みの manifest 1 行（= アノテーション済み 1 点）。"""

    idx: int
    image_id: str
    image_path: str
    face_index: int
    part_name: str
    sub_key: str
    point_index: int
    px: float
    py: float
    qx: float
    qy: float
    t: float
    nose_x: Optional[float]
    nose_y: Optional[float]


def is_valid_image_id(image_id: str) -> bool:
    """Image ID として安全な文字列（英数字・`_`・`-`）か。例外を出さず判定する。"""
    return _IMAGE_ID_RE.fullmatch(image_id) is not None


def validate_image_id(image_id: str) -> str:
    """Open Images の Image ID として安全な文字列であることを検証する。"""
    if not is_valid_image_id(image_id):
        raise ManifestError(
            f"不正な Image ID です（使用可能: 英数字 / _ / -）: {image_id!r}"
        )
    return image_id


def parse_image_path(image_path: str) -> str:
    """`data/images/<ID>.jpg` から Image ID を取り出す。それ以外の形式は拒否する。"""
    normalized = image_path.replace("\\", "/")
    match = _IMAGE_PATH_RE.fullmatch(normalized)
    if match is None:
        raise ManifestError(
            f"image_path は data/images/<ID>.jpg 形式のみ許可されます: {image_path!r}"
        )
    return match.group(1)


def _parse_exp_number(exp: Union[int, str]) -> int:
    if isinstance(exp, int):
        return exp
    match = _EXP_ARG_RE.fullmatch(exp.strip())
    if match is None:
        raise ManifestError(f"実験番号の形式が不正です: {exp!r}（例: 32, 032, EXP032）")
    return int(match.group(1))


def _experiment_manifests(experiments_dir: Path) -> dict[int, ManifestSource]:
    found: dict[int, ManifestSource] = {}
    if not experiments_dir.is_dir():
        return found
    for exp_dir in sorted(experiments_dir.iterdir()):
        match = _EXP_DIR_RE.match(exp_dir.name)
        if match is None or not exp_dir.is_dir():
            continue
        paths = tuple(
            exp_dir / "manifests" / name
            for name in MANIFEST_FILENAMES
            if (exp_dir / "manifests" / name).is_file()
        )
        if paths:
            found[int(match.group(1))] = ManifestSource(int(match.group(1)), exp_dir, paths)
    return found


def find_latest_manifest(
    experiments_dir: Path,
    exp: Union[int, str, None] = None,
) -> ManifestSource:
    """最新（または `exp` 指定）の実験の manifest を返す。

    Raises:
        ManifestError: manifest が 1 つも無い / 指定した実験が無い場合。
    """
    available = _experiment_manifests(experiments_dir)
    if not available:
        raise ManifestError(
            f"manifest が見つかりません: {experiments_dir}/EXP*/manifests/manifest_"
            "{train,test}.csv"
        )
    if exp is None:
        return available[max(available)]

    number = _parse_exp_number(exp)
    if number not in available:
        names = ", ".join(f"EXP{n:03d}" for n in sorted(available))
        raise ManifestError(
            f"EXP{number:03d} の manifest が見つかりません（利用可能: {names}）"
        )
    return available[number]


def _parse_int(value: str, column: str, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ManifestError(f"{where}: {column} が整数ではありません: {value!r}") from None


def _parse_float(value: Optional[str], column: str, where: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ManifestError(f"{where}: {column} が数値ではありません: {value!r}") from None
    if not math.isfinite(number):
        raise ManifestError(f"{where}: {column} が有限値ではありません: {value!r}")
    return number


def _parse_optional_float(value: Optional[str], column: str, where: str) -> Optional[float]:
    """省略可能な列。空欄・NaN・inf（同梱 manifest の鼻先未検出行）は欠損 (None) とみなす。"""
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        raise ManifestError(f"{where}: {column} が数値ではありません: {value!r}") from None
    return number if math.isfinite(number) else None


def _parse_row(record: dict[str, str], where: str) -> ManifestRow:
    missing = [c for c in REQUIRED_COLUMNS if record.get(c) is None]  # 途中で切れた行は None になる
    if missing:
        raise ManifestError(f"{where}: 列が足りません: {', '.join(missing)}")
    try:
        image_id = parse_image_path(record["image_path"])
    except ManifestError as exc:
        raise ManifestError(f"{where}: {exc}") from None

    part_name = record["part_name"]
    if part_name not in VALID_PART_NAMES:
        raise ManifestError(f"{where}: part_name が不正です: {part_name!r}")
    sub_key = record["sub_key"]
    if sub_key not in VALID_SUB_KEYS:
        raise ManifestError(f"{where}: sub_key が不正です: {sub_key!r}")

    face_index = _parse_int(record["face_index"], "face_index", where)
    point_index = _parse_int(record["point_index"], "point_index", where)
    if face_index < 0:
        raise ManifestError(f"{where}: face_index が負です: {face_index}")
    if not 0 <= point_index < POINTS_PER_SUB_KEY:
        raise ManifestError(
            f"{where}: point_index は 0〜{POINTS_PER_SUB_KEY - 1} が必要です: {point_index}"
        )

    return ManifestRow(
        idx=_parse_int(record["idx"], "idx", where),
        image_id=image_id,
        image_path=f"data/images/{image_id}.jpg",
        face_index=face_index,
        part_name=part_name,
        sub_key=sub_key,
        point_index=point_index,
        px=_parse_float(record["px"], "px", where),
        py=_parse_float(record["py"], "py", where),
        qx=_parse_float(record["qx"], "qx", where),
        qy=_parse_float(record["qy"], "qy", where),
        t=_parse_float(record["t"], "t", where),
        nose_x=_parse_optional_float(record.get("nose_x"), "nose_x", where),
        nose_y=_parse_optional_float(record.get("nose_y"), "nose_y", where),
    )


def _read_manifest(path: Path) -> list[ManifestRow]:
    if not path.is_file():
        raise ManifestError(f"manifest が見つかりません: {path}")

    rows: list[ManifestRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            hint = (
                "（EXP001〜007 の manifest は face_index を持たないため復元に使えません）"
                if "face_index" in missing else ""
            )
            raise ManifestError(
                f"{path.name}: 必要な列がありません: {', '.join(missing)}{hint}"
            )
        for record in reader:
            rows.append(_parse_row(record, f"{path.name}:{reader.line_num}"))
    return rows


def load_manifest_rows(paths: Iterable[Path]) -> tuple[ManifestRow, ...]:
    """manifest CSV 群を検証しながら読み込み、idx 昇順の不変タプルで返す。

    Raises:
        ManifestError: 列不足・値不正・idx 重複・空 manifest。
    """
    rows: list[ManifestRow] = []
    seen: dict[int, str] = {}
    for path in paths:
        path = Path(path)
        file_rows = _read_manifest(path)
        if not file_rows:
            raise ManifestError(f"{path.name}: manifest が空です")
        for row in file_rows:
            if row.idx in seen:
                raise ManifestError(
                    f"idx が重複しています: {row.idx}（{seen[row.idx]} と {path.name}）"
                )
            seen[row.idx] = path.name
        rows.extend(file_rows)
    return tuple(sorted(rows, key=lambda r: r.idx))


def load_manifest(source: ManifestSource) -> tuple[ManifestRow, ...]:
    return load_manifest_rows(source.paths)


def image_ids(rows: Sequence[ManifestRow]) -> list[str]:
    """manifest に含まれる Image ID を重複なし・昇順で返す。"""
    return sorted({row.image_id for row in rows})

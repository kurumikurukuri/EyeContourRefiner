"""pyproject.toml の依存宣言が実際の import と噛み合っていることを検証する。

コアライブラリ（eyeliner/）・ONNX エクスポートスクリプト（scripts/）・配布モデルの
参考実装（models/）が import エラーにならないことを保証する。import は base
dependencies だけでなく optional-dependencies（demo 等の extra）も含めた宣言と突き合わせる。
mediapipe は models/demo_visualize.py 専用のため `demo` extra に宣言している。
tests/ 自身の import は dev extra を含めた宣言と突き合わせる。
"""
from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).parent.parent

SCANNED_DIRS = ("eyeliner", "scripts", "models")
FIRST_PARTY_DIRS = ("eyeliner", "scripts", "models", "tests")

# import 名と配布パッケージ名が異なるもの
IMPORT_TO_DISTRIBUTION = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "PIL": "pillow",
}

# 直接 import しているが、宣言済みパッケージの必須依存として必ず入るもの
PROVIDED_BY = {
    "pydantic": "fastapi",
}


def _first_party_module_names() -> set[str]:
    names = set(FIRST_PARTY_DIRS)
    for directory in FIRST_PARTY_DIRS:
        names.update(p.stem for p in (ROOT / directory).rglob("*.py"))
    return names


def _scanned_files() -> list[Path]:
    return [p for d in SCANNED_DIRS for p in (ROOT / d).rglob("*.py")]


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _distribution_for(import_name: str) -> str:
    provider = PROVIDED_BY.get(import_name)
    if provider is not None:
        return canonicalize_name(provider)
    return canonicalize_name(IMPORT_TO_DISTRIBUTION.get(import_name, import_name))


def _requirement_key(line: str) -> tuple[str, tuple[str, ...], str, str]:
    req = Requirement(line)
    marker = str(req.marker) if req.marker else ""
    return canonicalize_name(req.name), tuple(sorted(req.extras)), str(req.specifier), marker


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pyproject_dependencies() -> list[str]:
    return list(_pyproject()["project"]["dependencies"])


def _pyproject_dev_dependencies() -> list[str]:
    return list(_pyproject()["project"]["optional-dependencies"]["dev"])


def _pyproject_all_dependencies() -> list[str]:
    deps = list(_pyproject_dependencies())
    for extra in _pyproject()["project"].get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


def _requirements_txt_lines() -> list[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def test_scan_covers_core_and_release_entry_points():
    files = _scanned_files()
    assert any(p.is_relative_to(ROOT / "eyeliner") for p in files)
    assert any(p.name == "export_dual_onnx.py" for p in files)
    assert any(p.name == "infer_example.py" for p in files)


def test_all_third_party_imports_are_declared_in_pyproject():
    declared = {_requirement_key(r)[0] for r in _pyproject_all_dependencies()}
    ignored = set(sys.stdlib_module_names) | _first_party_module_names() | {"__future__"}

    missing: dict[str, list[str]] = {}
    for path in _scanned_files():
        for import_name in _top_level_imports(path) - ignored:
            distribution = _distribution_for(import_name)
            if distribution not in declared:
                missing.setdefault(distribution, []).append(path.relative_to(ROOT).as_posix())

    report = {dist: sorted(files)[:3] for dist, files in sorted(missing.items())}
    assert not missing, f"pyproject.toml の dependencies に未宣言の import がある: {report}"


def test_all_third_party_imports_in_tests_are_declared():
    """テスト自身が使う第三者パッケージが dependencies か dev extra に宣言されていること。"""
    declared = {_requirement_key(r)[0] for r in _pyproject_dependencies() + _pyproject_dev_dependencies()}
    ignored = set(sys.stdlib_module_names) | _first_party_module_names() | {"__future__"}

    missing: dict[str, list[str]] = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for import_name in _top_level_imports(path) - ignored:
            distribution = _distribution_for(import_name)
            if distribution not in declared:
                missing.setdefault(distribution, []).append(path.relative_to(ROOT).as_posix())

    report = {dist: sorted(files)[:3] for dist, files in sorted(missing.items())}
    assert not missing, f"dependencies にも dev extra にも未宣言の import がある: {report}"


def _pyproject_demo_dependencies() -> list[str]:
    return list(_pyproject()["project"]["optional-dependencies"]["demo"])


def test_mediapipe_is_capped_to_versions_with_legacy_solutions_api():
    """`mp.solutions`（FaceMesh）は 0.10.30 以降の wheel に含まれないため上限を宣言する。

    models/demo_visualize.py は mp.solutions.face_mesh に依存する。face_makeup が固定する
    0.10.21 で動作することを確認済み（0.10.32 では AttributeError になる）。
    """
    (line,) = [r for r in _pyproject_demo_dependencies() if _requirement_key(r)[0] == "mediapipe"]
    specifier = Requirement(line).specifier

    assert Version("0.10.21") in specifier
    assert Version("0.10.30") not in specifier
    assert Version("0.10.32") not in specifier


def test_opencv_is_capped_below_5():
    """OpenCV 5.0 は warpAffine の固定小数点丸めが 4.x と異なり、パッチ画素が 1/255 ずれる。

    学習・配布モデルは 4.x で作られており、tests/test_source_degradation.py の
    bbox 含有性テスト（ビット一致を検定）も 4.x 前提のため 5 未満に固定する。
    """
    (line,) = [r for r in _pyproject_dependencies() if _requirement_key(r)[0] == "opencv-python"]
    specifier = Requirement(line).specifier

    assert Version("4.8.0") in specifier
    assert Version("4.14.0") in specifier
    assert Version("5.0.0") not in specifier


def test_requirements_txt_matches_pyproject_dependencies():
    pyproject = {_requirement_key(r) for r in _pyproject_dependencies()}
    requirements = {_requirement_key(r) for r in _requirements_txt_lines()}

    assert pyproject - requirements == set(), "pyproject.toml にだけある依存"
    assert requirements - pyproject == set(), "requirements.txt にだけある依存"

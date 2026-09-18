"""実験間の独立性（CLAUDE.md「実験間の独立性（CRITICAL）」）のうち、コード依存を検証する。

ある実験の scr/ が、sys.path 操作によって別の実験の scr/ にあるモジュールを import していないこと。
共通処理は eyeliner/ に置き、各実験はそこから利用する。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXPERIMENTS = ROOT / "experiments"
EXP_ID = re.compile(r"EXP(\d{3})")


def _experiment_id(scr_file: Path) -> str:
    match = EXP_ID.match(scr_file.parent.parent.name)
    assert match is not None, f"実験ディレクトリ名が EXP{{NNN}}_ 形式ではない: {scr_file}"
    return match.group(1)


def _scr_modules() -> dict[Path, set[str]]:
    return {scr: {p.stem for p in scr.glob("*.py")} for scr in sorted(EXPERIMENTS.glob("EXP*/scr"))}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_scr_files_are_found():
    assert len(list(EXPERIMENTS.glob("EXP*/scr/*.py"))) > 50


def test_scr_does_not_import_modules_from_other_experiments():
    modules_by_scr = _scr_modules()
    violations: list[str] = []
    for scr, own_modules in modules_by_scr.items():
        other_modules = set().union(*(m for s, m in modules_by_scr.items() if s != scr))
        foreign = other_modules - own_modules - set(sys.stdlib_module_names)
        for path in sorted(scr.glob("*.py")):
            for name in sorted(_top_level_imports(path) & foreign):
                violations.append(f"{path.relative_to(ROOT).as_posix()}: import {name}")

    assert not violations, "別の実験の scr/ のモジュールを import している:\n" + "\n".join(violations)


def test_scr_does_not_add_other_experiment_directories_to_sys_path():
    violations: list[str] = []
    for path in sorted(EXPERIMENTS.glob("EXP*/scr/*.py")):
        own_id = _experiment_id(path)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "sys.path" in line and set(EXP_ID.findall(line)) - {own_id}:
                violations.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: {line.strip()}")

    assert not violations, "sys.path に別の実験のディレクトリを追加している:\n" + "\n".join(violations)

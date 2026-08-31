"""Layer boundaries.

Two different things are checked here, and the second one matters more:

1. the real contracts hold for this repo (a regression guard), and
2. the mechanism actually FAILS on a violation.

Without (2), "import boundaries enforced" would rest on a checker that has never
once said no - which in Phase 1, when there are barely any cross-imports, is
exactly the situation a green result would be hiding.
"""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _lint_imports_cmd() -> list[str]:
    """Locate the console script.

    ``python -m importlinter.cli`` is NOT usable: the module has no __main__
    guard, so it exits 0 having done nothing - which would make every assertion
    below vacuously pass.
    """
    found = shutil.which("lint-imports")
    if found:
        return [found]
    scripts = Path(sys.executable).parent
    for name in ("lint-imports.exe", "lint-imports"):
        candidate = scripts / name
        if candidate.exists():
            return [str(candidate)]
    pytest.skip("lint-imports console script not found")


def _run_lint_imports(cwd: Path, config: str, extra_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(extra_path)
    return subprocess.run(
        [*_lint_imports_cmd(), "--config", config, "--no-cache"],
        cwd=cwd, capture_output=True, text=True, env=env, check=False,
    )


def test_repo_contracts_hold() -> None:
    result = _run_lint_imports(REPO, ".importlinter", REPO)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout


def test_the_three_directions_the_brief_names_are_encoded() -> None:
    """Section 7: data !-> training, training !-> serving, serving !-> data."""
    cfg = configparser.ConfigParser()
    cfg.read(REPO / ".importlinter", encoding="utf-8")

    forbidden: dict[str, set[str]] = {}
    for section in cfg.sections():
        if not section.startswith("importlinter:contract"):
            continue
        sources = cfg[section]["source_modules"].split()
        targets = set(cfg[section]["forbidden_modules"].split())
        for src in sources:
            forbidden.setdefault(src, set()).update(targets)

    assert "reckon.training" in forbidden["reckon.data"]
    assert "reckon.serve" in forbidden["reckon.training"]
    assert "reckon.data" in forbidden["reckon.serve"]


def test_a_violation_is_actually_detected(tmp_path: Path) -> None:
    """Plant a forbidden import in a throwaway package and require a red build."""
    pkg = tmp_path / "layerdemo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lower.py").write_text("VALUE = 1\n", encoding="utf-8")
    # upper -> lower is fine; lower -> upper is the violation we plant.
    (pkg / "upper.py").write_text("from layerdemo import lower\n", encoding="utf-8")

    config = tmp_path / ".importlinter"
    config.write_text(
        "[importlinter]\n"
        "root_package = layerdemo\n\n"
        "[importlinter:contract:1]\n"
        "name = upper must not import lower\n"
        "type = forbidden\n"
        "source_modules =\n"
        "    layerdemo.upper\n"
        "forbidden_modules =\n"
        "    layerdemo.lower\n",
        encoding="utf-8",
    )

    result = _run_lint_imports(tmp_path, ".importlinter", tmp_path)
    assert result.returncode != 0, "import-linter passed a known violation"
    assert "BROKEN" in result.stdout.upper()
    assert "layerdemo.upper" in result.stdout


def test_no_violation_passes_in_the_same_harness(tmp_path: Path) -> None:
    """Control for the test above: identical setup, violating import removed."""
    pkg = tmp_path / "layerdemo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lower.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "upper.py").write_text("VALUE = 2\n", encoding="utf-8")

    config = tmp_path / ".importlinter"
    config.write_text(
        "[importlinter]\n"
        "root_package = layerdemo\n\n"
        "[importlinter:contract:1]\n"
        "name = upper must not import lower\n"
        "type = forbidden\n"
        "source_modules =\n"
        "    layerdemo.upper\n"
        "forbidden_modules =\n"
        "    layerdemo.lower\n",
        encoding="utf-8",
    )

    result = _run_lint_imports(tmp_path, ".importlinter", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_finds_lint_imports() -> None:
    """The pre-commit launcher must resolve the console script."""
    from tools.hooks.run_lint_imports import find_lint_imports

    command = find_lint_imports()
    assert command is not None
    assert Path(command[0]).is_file()


def test_launcher_runs_without_the_venv_on_path() -> None:
    """Git invokes hooks with the ambient PATH, not the project environment.

    A hook that only passes when you happen to commit from `uv run` is worse
    than no hook, because it is trusted.
    """
    stripped = dict(os.environ)
    stripped.pop("VIRTUAL_ENV", None)
    venv_scripts = str(Path(sys.executable).parent)
    stripped["PATH"] = os.pathsep.join(
        p for p in stripped.get("PATH", "").split(os.pathsep)
        if p and Path(p) != Path(venv_scripts)
    )

    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "hooks" / "run_lint_imports.py")],
        cwd=REPO, capture_output=True, text=True, env=stripped, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout

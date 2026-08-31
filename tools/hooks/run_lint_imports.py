"""Pre-commit launcher for import-linter.

Git runs hooks with the ambient PATH, which normally does not contain the
project's virtualenv. A `language: system` entry of plain `lint-imports`
therefore works when you commit from `uv run` and fails from a bare shell or an
IDE - a hook that only sometimes runs is worse than no hook, because it is
trusted.

The alternative, pre-commit's `additional_dependencies`, would pin import-linter
a second time outside `uv.lock` and let the two drift. This launcher instead
locates the project venv and keeps `uv.lock` the single source of truth.

Stdlib only: it must run under whatever interpreter git happens to hand it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_lint_imports() -> list[str] | None:
    """Locate the console script: project venv first, then PATH."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "lint-imports.exe",  # Windows
        REPO_ROOT / ".venv" / "Scripts" / "lint-imports",
        REPO_ROOT / ".venv" / "bin" / "lint-imports",          # POSIX
    ]
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]

    found = shutil.which("lint-imports")
    return [found] if found else None


def main(argv: list[str] | None = None) -> int:
    command = find_lint_imports()
    if command is None:
        print(
            "lint-imports not found. Create the environment first:\n"
            "  uv sync --extra dev",
            file=sys.stderr,
        )
        return 1
    return subprocess.run(
        [*command, *(argv or [])], cwd=REPO_ROOT, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

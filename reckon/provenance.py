"""Run provenance and seeding.

Section 1 of the brief: every run must be reproducible from a config plus a seed,
and every run records its config hash, git SHA and library versions. A result
whose provenance block cannot be regenerated does not go in the results table, so
this module is deliberately dependency-free and importable from anywhere.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "TRACKED_PACKAGES",
    "file_sha256",
    "git_sha",
    "git_is_dirty",
    "package_versions",
    "run_metadata",
    "set_global_seed",
]

#: Libraries whose version can change a numeric result. Recorded on every run.
TRACKED_PACKAGES: tuple[str, ...] = (
    "pydantic", "numpy", "scipy", "rapidfuzz", "torch", "transformers", "datasets",
    "pillow", "augraphy", "onnxruntime", "paddleocr",
)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes. Used for config hashes and dataset manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_sha(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Current commit SHA, or None outside a repo / before the first commit."""
    return _git("rev-parse", "HEAD", cwd=cwd)


def git_is_dirty(cwd: str | os.PathLike[str] | None = None) -> bool | None:
    """True when the working tree has uncommitted changes.

    A dirty tree means the git SHA does not fully describe the code that ran, so
    the flag is recorded alongside it rather than being quietly ignored.
    """
    out = _git("status", "--porcelain", cwd=cwd)
    return None if out is None else bool(out)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of *names*; absent packages are omitted."""
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def set_global_seed(seed: int) -> int:
    """Seed every RNG that is actually installed, and return the seed.

    numpy and torch are seeded only if importable, so this stays usable in the
    Phase 1 environment where neither is a dependency yet.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def run_metadata(
    config_path: str | os.PathLike[str] | None = None,
    *,
    seed: int | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Provenance block written into every run, report and checkpoint."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(root),
        "git_dirty": git_is_dirty(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "seed": seed,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": file_sha256(config_path) if config_path else None,
    }
    return meta

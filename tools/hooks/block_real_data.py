"""Pre-commit hook: refuse any staged file under ``data/real/``.

`.gitignore` already excludes that directory, but `git add -f` overrides an
ignore file. This hook is the second, independent barrier, and unlike the ignore
file it also fires for a path that was staged before the ignore rule existed.

Exit codes: 0 = clean, 1 = at least one forbidden path staged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import PurePosixPath

#: Any staged path containing this directory segment sequence is refused.
FORBIDDEN_SEGMENTS: tuple[tuple[str, ...], ...] = (
    ("data", "real"),
)


def is_forbidden(path: str) -> bool:
    """True when *path* sits under a forbidden directory, at any depth."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    for segments in FORBIDDEN_SEGMENTS:
        n = len(segments)
        for i in range(len(parts) - n + 1):
            if tuple(parts[i : i + n]) == segments:
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    offenders = [f for f in args.filenames if is_forbidden(f)]
    if not offenders:
        return 0

    print("BLOCKED: real patient documents must never be committed.", file=sys.stderr)
    for f in offenders:
        print(f"  {f}", file=sys.stderr)
    print(
        "\ndata/real/ is gitignored and additionally blocked by this hook.\n"
        "Unstage with:  git restore --staged <path>\n"
        "If you believe a file here is genuinely synthetic, move it out of\n"
        "data/real/ rather than weakening the hook.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

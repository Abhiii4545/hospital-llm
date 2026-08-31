"""Pre-commit hook: block Indian PII patterns in staged text files.

Complements ``block_real_data.py``. That hook guards a path; this one guards
content, because leaked PII usually arrives pasted into a notebook, a fixture or
a debug log rather than as a file under ``data/real/``.

The synthetic corpus legitimately contains fake PANs, GSTINs and phone numbers,
so the hook supports two escape hatches, both of which leave an audit trail:

* path prefixes listed in ``tools/pii_allowlist.txt``
* an inline ``pii-allow`` marker on the offending line

Exit codes: 0 = clean, 1 = at least one match.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: (name, compiled pattern). Ordered most-specific first for clearer reporting.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Aadhaar: 12 digits, never starting 0 or 1, optionally spaced 4-4-4.
    ("aadhaar", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    # PAN: AAAAA9999A pii-allow
    ("pan", re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])")),
    # GSTIN: 15 chars, 14th is Z
    ("gstin", re.compile(r"(?<![A-Z0-9])\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9](?![A-Z0-9])")),
    # Indian mobile, optionally +91 prefixed
    ("phone_in", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
)

INLINE_ALLOW = "pii-allow"

#: Binary or generated files that are never worth scanning.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".tif", ".tiff", ".webp",
    ".zip", ".gz", ".tar", ".whl", ".so", ".dll", ".pyd", ".bin",
    ".safetensors", ".onnx", ".pt", ".ckpt", ".lock",
}

MAX_BYTES = 2_000_000


def load_allowlist(path: Path) -> list[str]:
    """Path prefixes exempt from scanning. Missing file means no exemptions."""
    if not path.is_file():
        return []
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            prefixes.append(line.rstrip("/"))
    return prefixes


def is_allowlisted(filename: str, prefixes: list[str]) -> bool:
    norm = filename.replace("\\", "/").lstrip("./")
    return any(norm == p or norm.startswith(p + "/") for p in prefixes)


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return (line number, pattern name, matched text) for each hit."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if INLINE_ALLOW in line:
            continue
        for name, pattern in PATTERNS:
            for m in pattern.finditer(line):
                hits.append((lineno, name, m.group(0)))
    return hits


def _redact(value: str) -> str:
    """Never echo the matched PII in full - the hook output is itself a leak."""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    parser.add_argument(
        "--allowlist",
        default="tools/pii_allowlist.txt",
        help="file of path prefixes exempt from scanning",
    )
    args = parser.parse_args(argv)

    prefixes = load_allowlist(Path(args.allowlist))
    failed = False

    for filename in args.filenames:
        path = Path(filename)
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if is_allowlisted(filename, prefixes):
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: block_real_data covers the paths

        for lineno, name, value in scan_text(text):
            failed = True
            print(
                f"{filename}:{lineno}: possible {name}: {_redact(value)}",
                file=sys.stderr,
            )

    if failed:
        print(
            "\nBLOCKED: possible PII in staged content.\n"
            f"If this is synthetic, add a '{INLINE_ALLOW}' comment on the line, or\n"
            "add the path prefix to tools/pii_allowlist.txt. Do not disable the hook.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The privacy machinery must be proven to fire, not merely to exist.

These tests are the reason the hooks are trustworthy. If any of them starts
failing, real patient data can reach version control.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.hooks.block_real_data import is_forbidden, main as block_main  # noqa: E402
from tools.hooks.pii_scan import (  # noqa: E402
    is_allowlisted,
    main as pii_main,
    scan_text,
)

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# .gitignore
# --------------------------------------------------------------------------

def test_gitignore_blocks_real_data_on_line_one() -> None:
    """Section 1: data/real/ is on line one, before anything else is written."""
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines[0].strip() == "data/real/"


def test_git_actually_ignores_the_real_data_directory() -> None:
    """Assert on git's behaviour, not on the text of the ignore file."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/real/bill.pdf"],
        cwd=REPO, capture_output=True, check=False,
    )
    assert result.returncode == 0, "git does not ignore data/real/"


def test_nothing_under_real_data_is_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    assert [p for p in tracked if "data/real/" in p.replace("\\", "/")] == []


# --------------------------------------------------------------------------
# block_real_data hook
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "data/real/bill.pdf",
        "data/real/CONSENT.md",
        "data/real/nested/deep/scan_001.png",
        "./data/real/x.json",
        r"data\real\windows_path.pdf",
        "some/prefix/data/real/leaked.pdf",
    ],
)
def test_forbidden_paths_are_detected(path: str) -> None:
    assert is_forbidden(path)


@pytest.mark.parametrize(
    "path",
    [
        "data/synthetic/bill.pdf",
        "reckon/data/build_corpus.py",
        "data/realistic/x.png",       # substring, not a path segment
        "docs/real-world-notes.md",
        "tests/test_privacy_hooks.py",
    ],
)
def test_permitted_paths_are_not_flagged(path: str) -> None:
    assert not is_forbidden(path)


def test_block_hook_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert block_main(["reckon/schema.py"]) == 0
    assert block_main(["data/real/bill.pdf"]) == 1
    assert "BLOCKED" in capsys.readouterr().err


def test_block_hook_survives_force_add(tmp_path: Path) -> None:
    """`git add -f` defeats .gitignore. The hook is the barrier that does not.

    A throwaway repository is used so the project's own history is untouched.
    """
    repo = tmp_path / "repo"
    (repo / "data" / "real").mkdir(parents=True)
    (repo / "data" / "real" / "bill.txt").write_text("patient record", encoding="utf-8")
    (repo / ".gitignore").write_text("data/real/\n", encoding="utf-8")

    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, capture_output=True, text=True, check=False
    )
    run("init", "-q")
    assert run("check-ignore", "-q", "data/real/bill.txt").returncode == 0
    run("add", "-f", "data/real/bill.txt")
    staged = run("diff", "--cached", "--name-only").stdout.split()
    assert "data/real/bill.txt" in staged, "force-add did bypass .gitignore"

    # ...and the hook catches exactly this case.
    assert block_main(staged) == 1


# --------------------------------------------------------------------------
# pii_scan hook
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("aadhaar 234512345678 here", "aadhaar"),        # pii-allow
        ("aadhaar 2345 1234 5678 here", "aadhaar"),      # pii-allow
        ("pan ABCDE1234F end", "pan"),                   # pii-allow
        ("gstin 27AAPFU0939F1ZV end", "gstin"),          # pii-allow
        ("call 9876543210 now", "phone_in"),             # pii-allow
        ("call +91 9876543210 now", "phone_in"),         # pii-allow
        ("mail patient@example.com ok", "email"),        # pii-allow
    ],
)
def test_pii_patterns_are_detected(text: str, expected: str) -> None:
    names = {name for _, name, _ in scan_text(text)}
    assert expected in names


@pytest.mark.parametrize(
    "text",
    [
        "amount 1,23,456.00",
        "uhid UH202500123",
        "invoice 30049099",
        "date 2025-01-05",
        "room rent 4500",
    ],
)
def test_benign_content_is_not_flagged(text: str) -> None:
    assert scan_text(text) == []


def test_inline_marker_suppresses_a_line() -> None:
    marker = "pii" + "-allow"  # built at runtime so this line is not itself exempt
    assert scan_text("pan ABCDE1234F") != []          # pii-allow
    assert scan_text(f"pan ABCDE1234F  # {marker}") == []  # pii-allow


def test_reported_matches_are_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook's own output must not become the leak."""
    secret = "9876543210"  # pii-allow
    target = tmp_path / "leak.txt"
    target.write_text(f"phone {secret}\n", encoding="utf-8")

    assert pii_main([str(target), "--allowlist", "does-not-exist.txt"]) == 1
    err = capsys.readouterr().err
    assert secret not in err
    assert "phone_in" in err
    assert "**" in err


def test_pii_hook_passes_on_clean_file(tmp_path: Path) -> None:
    target = tmp_path / "clean.py"
    target.write_text("total = 1234\n", encoding="utf-8")
    assert pii_main([str(target), "--allowlist", "does-not-exist.txt"]) == 0


def test_pii_hook_skips_binary_suffixes(tmp_path: Path) -> None:
    target = tmp_path / "scan.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n 9876543210")  # pii-allow
    assert pii_main([str(target), "--allowlist", "does-not-exist.txt"]) == 0


def test_allowlist_matches_path_prefixes_not_substrings() -> None:
    prefixes = ["data/synthetic", "reckon/data/generators"]
    assert is_allowlisted("data/synthetic/bill.json", prefixes)
    assert is_allowlisted("reckon/data/generators/names.py", prefixes)
    assert not is_allowlisted("data/synthetic_backup/bill.json", prefixes)
    assert not is_allowlisted("reckon/schema.py", prefixes)


def test_allowlist_is_empty_by_default() -> None:
    """Nothing is blanket-exempt; exemptions are inline and visible per line."""
    from tools.hooks.pii_scan import load_allowlist

    assert load_allowlist(REPO / "tools" / "pii_allowlist.txt") == []


def test_repository_content_is_pii_clean() -> None:
    """The hook run over every tracked file in this repo, as CI does."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout.split()
    if not tracked:
        pytest.skip("nothing committed yet")
    assert pii_main([*tracked, "--allowlist", "tools/pii_allowlist.txt"]) == 0

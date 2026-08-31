"""Run provenance: a result that cannot be reproduced does not get reported."""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path

from reckon.provenance import (
    TRACKED_PACKAGES,
    file_sha256,
    git_is_dirty,
    git_sha,
    package_versions,
    run_metadata,
    set_global_seed,
)

REPO = Path(__file__).resolve().parent.parent


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    payload = b"seed: 1337\nlr: 3e-5\n"
    target.write_bytes(payload)
    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_config_hash_changes_when_the_config_changes(tmp_path: Path) -> None:
    """This is the mechanism that marks old results stale."""
    target = tmp_path / "config.yaml"
    target.write_text("seed: 1337\n", encoding="utf-8")
    before = file_sha256(target)
    target.write_text("seed: 1338\n", encoding="utf-8")
    assert file_sha256(target) != before


def test_git_sha_is_available_or_explicitly_absent() -> None:
    sha = git_sha(REPO)
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_git_dirty_flag_is_a_bool_in_a_repo() -> None:
    assert isinstance(git_is_dirty(REPO), bool)


def test_git_sha_is_none_before_the_first_commit(tmp_path: Path) -> None:
    """A fresh repo has no HEAD, and that must be None rather than a crash.

    Note this deliberately does NOT assert that a temp directory is outside any
    repository. On a machine where the user's home directory is itself a git
    repo - which is the case on the box this was developed on - pytest's
    tmp_path is inside it, and git happily reports that outer repo's SHA. A
    fresh `git init` is the portable way to reach the no-HEAD branch.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert git_sha(tmp_path) is None
    assert git_is_dirty(tmp_path) is False


def test_git_helpers_do_not_raise_on_a_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert git_sha(missing) is None
    assert git_is_dirty(missing) is None


def test_package_versions_reports_installed_and_omits_absent() -> None:
    versions = package_versions()
    assert "pydantic" in versions
    assert set(versions).issubset(set(TRACKED_PACKAGES))
    assert package_versions(("definitely-not-installed-xyz",)) == {}


def test_run_metadata_has_every_required_key(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("seed: 1337\n", encoding="utf-8")

    meta = run_metadata(config, seed=1337, repo_root=REPO)
    for key in (
        "timestamp_utc", "git_sha", "git_dirty", "python", "platform",
        "packages", "seed", "config_path", "config_sha256",
    ):
        assert key in meta, key

    assert meta["seed"] == 1337
    assert meta["config_sha256"] == file_sha256(config)
    assert meta["packages"]["pydantic"]


def test_run_metadata_without_a_config() -> None:
    meta = run_metadata(repo_root=REPO)
    assert meta["config_path"] is None
    assert meta["config_sha256"] is None


def test_seeding_is_reproducible() -> None:
    set_global_seed(1337)
    first = [random.random() for _ in range(5)]
    set_global_seed(1337)
    assert [random.random() for _ in range(5)] == first


def test_different_seeds_diverge() -> None:
    set_global_seed(1)
    first = [random.random() for _ in range(5)]
    set_global_seed(2)
    assert [random.random() for _ in range(5)] != first

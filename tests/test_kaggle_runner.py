"""Kaggle runner: credential detection and kernel metadata.

The credential check is the part that must not be wrong. A false negative wastes
someone's time; a false positive sends a push at a server with no auth and gets a
confusing error back. Neither secret file is ever read for its key here - only
its presence, and the legacy file's username.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import kaggle_run  # noqa: E402


@pytest.fixture
def kaggle_dir(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / ".kaggle"
    directory.mkdir()
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(directory))
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return directory


def test_no_credentials_fails_with_instructions(kaggle_dir, capsys) -> None:
    with pytest.raises(SystemExit):
        kaggle_run.check_credentials()
    err = capsys.readouterr().err
    assert "access_token" in err and "kaggle.json" in err
    assert "Do not paste a token into a chat" in err


def test_legacy_kaggle_json_yields_its_username(kaggle_dir) -> None:
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "legacyuser", "key": "NOT_A_REAL_KEY"}),
        encoding="utf-8",
    )
    assert kaggle_run.check_credentials() == "legacyuser"


def test_access_token_requires_an_explicit_username(kaggle_dir, capsys) -> None:
    """The newer token file contains only a key, so the username must be given."""
    (kaggle_dir / "access_token").write_text("NOT_A_REAL_TOKEN", encoding="utf-8")

    with pytest.raises(SystemExit):
        kaggle_run.check_credentials()
    assert "--username" in capsys.readouterr().err

    assert kaggle_run.check_credentials("someuser") == "someuser"


def test_explicit_username_wins_over_the_file(kaggle_dir) -> None:
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "fromfile", "key": "NOT_A_REAL_KEY"}), encoding="utf-8"
    )
    assert kaggle_run.check_credentials("override") == "override"


def test_environment_credentials_are_accepted(kaggle_dir, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
    monkeypatch.setenv("KAGGLE_KEY", "NOT_A_REAL_KEY")
    assert kaggle_run.check_credentials() == "envuser"


def test_malformed_legacy_file_fails_clearly(kaggle_dir, capsys) -> None:
    (kaggle_dir / "kaggle.json").write_text("not json at all", encoding="utf-8")
    with pytest.raises(SystemExit):
        kaggle_run.check_credentials()
    assert "not valid JSON" in capsys.readouterr().err


def test_kernel_metadata_enables_gpu_and_internet(tmp_path, monkeypatch) -> None:
    """The two settings that silently ruin a run when forgotten."""
    monkeypatch.setattr(kaggle_run, "KERNEL_DIR", tmp_path / "kernel")
    monkeypatch.setattr(kaggle_run, "NOTEBOOK",
                        Path("notebooks/train_kaggle.ipynb"))

    path = kaggle_run.write_kernel_metadata("someuser", "someuser/corpus", "slug")
    meta = json.loads(path.read_text(encoding="utf-8"))

    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True          # a bill corpus is not public by default
    assert meta["dataset_sources"] == ["someuser/corpus"]
    assert meta["id"] == "someuser/slug"
    assert (tmp_path / "kernel" / "slug.ipynb").exists()


def test_the_notebook_it_ships_is_kaggle_shaped() -> None:
    """A Colab notebook pushed to Kaggle fails on the Drive mount."""
    source = Path("notebooks/train_kaggle.ipynb").read_text(encoding="utf-8")
    assert "google.colab" not in source
    assert "/kaggle/input" in source
    assert "0.372" in source                   # the gate is stated in the notebook

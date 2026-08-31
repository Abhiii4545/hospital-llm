"""Push the training notebook to Kaggle as a GPU kernel, run it, and fetch output.

Removes every manual step except one: the Kaggle API token has to be placed by
you. This script never reads, prints or transmits the token itself - it only
checks that the file exists and lets the official CLI use it, the same way
`git push` used the Windows credential manager.

    # one-time, done by YOU - never paste a token into a chat:
    #   Kaggle -> Settings -> API -> Create New Token, then either
    #     echo <TOKEN> > ~/.kaggle/access_token   (newer KGAT_ style)
    #     mv kaggle.json ~/.kaggle/kaggle.json     (legacy style)

    python scripts/kaggle_run.py --dataset <owner>/<slug> --username <you>
    python scripts/kaggle_run.py --status
    python scripts/kaggle_run.py --fetch

Kaggle runs a pushed kernel in BATCH mode: it executes top to bottom with no
interaction, then saves output. That is the right mode here - a training run
should not need a human watching it - but it also means a cell that waits for
input hangs the whole thing, which is why the notebook has none.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KERNEL_DIR = Path("build/kaggle_kernel")
NOTEBOOK = Path("notebooks/train_kaggle.ipynb")

#: Kaggle has two auth styles in the wild and the CLI accepts either.
#: - legacy: ~/.kaggle/kaggle.json, a JSON object with username + key
#: - current: ~/.kaggle/access_token, a bare KGAT_... string (no username in it)
#:
#: Resolved by FUNCTION, not as a module constant. `Path.home()` at import time
#: cannot be pointed elsewhere for a test, and a credential check that cannot be
#: tested is a credential check nobody has verified.
def _kaggle_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle"))


def legacy_token_path() -> Path:
    return _kaggle_dir() / "kaggle.json"


def access_token_path() -> Path:
    return _kaggle_dir() / "access_token"


def _fail(message: str) -> None:
    print(f"\n{message}\n", file=sys.stderr)
    raise SystemExit(1)


def check_credentials(username_arg: str | None = None) -> str:
    """Confirm credentials exist and determine the username.

    The secret itself is never read, printed or transmitted by this script - only
    its presence is checked, and the official CLI reads the file. That is also why
    the newer `access_token` style needs --username: the username is simply not in
    that file, and the alternative would be parsing a file whose entire contents
    are a live key.
    """
    has_legacy = legacy_token_path().exists()
    has_access = access_token_path().exists()
    has_env = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )

    if not (has_legacy or has_access or has_env):
        _fail(
            "No Kaggle credentials found.\n\n"
            "Run ONE of these YOURSELF. Do not paste a token into a chat - if you\n"
            "already have, expire it on the Kaggle API page and make a new one.\n\n"
            "  Newer token (KGAT_...):\n"
            "    mkdir -p ~/.kaggle\n"
            "    echo <YOUR_TOKEN> > ~/.kaggle/access_token\n"
            "    chmod 600 ~/.kaggle/access_token\n"
            "    then re-run with --username <your-kaggle-username>\n\n"
            "  Legacy kaggle.json:\n"
            "    move the downloaded kaggle.json to ~/.kaggle/kaggle.json\n\n"
            "This script never reads the key; the CLI does."
        )

    if username_arg:
        return username_arg
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if has_legacy:
        try:
            username = json.loads(
                legacy_token_path().read_text(encoding="utf-8")
            ).get("username")
        except Exception:                                  # noqa: BLE001
            _fail("~/.kaggle/kaggle.json is not valid JSON.")
        else:
            if username:
                return username

    _fail(
        "Credentials found, but an access_token file does not carry a username.\n"
        "Re-run with --username <your-kaggle-username>."
    )
    return ""                                              # unreachable


def ensure_cli() -> None:
    if shutil.which("kaggle"):
        return
    print("installing the kaggle CLI …")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
                   check=True)
    if not shutil.which("kaggle"):
        _fail("kaggle CLI installed but not on PATH; open a new shell and retry.")


def write_kernel_metadata(username: str, dataset: str, slug: str) -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK, KERNEL_DIR / f"{slug}.ipynb")

    metadata = {
        "id": f"{username}/{slug}",
        "title": "RECKON v2 - train Head B",
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # Both are required and are the two settings people forget in the UI:
        # without internet, pip and the base-model download fail; without the
        # GPU it silently trains on CPU and burns the session.
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
    }
    path = KERNEL_DIR / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def run(command: list[str]) -> int:
    print("$", " ".join(command))
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="<owner>/<dataset-slug> to attach")
    parser.add_argument("--slug", default="reckon-v2-train-head-b")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--username", default=None,
                        help="required with the newer access_token style")
    args = parser.parse_args()

    username = check_credentials(args.username)
    ensure_cli()
    kernel = f"{username}/{args.slug}"

    if args.status:
        return run(["kaggle", "kernels", "status", kernel])

    if args.fetch:
        out = Path("reports/kaggle_output")
        out.mkdir(parents=True, exist_ok=True)
        return run(["kaggle", "kernels", "output", kernel, "-p", str(out)])

    if not args.dataset:
        _fail("--dataset <owner>/<slug> is required to push. "
              "Find it in the URL of your Kaggle dataset page.")
    if not NOTEBOOK.exists():
        _fail(f"{NOTEBOOK} not found. Run scripts/write_kaggle_notebook.py first.")

    write_kernel_metadata(username, args.dataset, args.slug)
    code = run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])
    if code == 0:
        print(
            f"\nPushed. Kaggle is now running it on a GPU.\n"
            f"  watch:  python scripts/kaggle_run.py --status\n"
            f"  fetch:  python scripts/kaggle_run.py --fetch\n"
            f"  web:    https://www.kaggle.com/code/{kernel}\n"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

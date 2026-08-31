"""Generate notebooks/train_colab.ipynb.

Written as a generator rather than a hand-edited .ipynb so the notebook stays
reviewable in diffs. Notebook JSON with embedded outputs is unreadable in code
review, and a training notebook is exactly the artefact where a silent change
costs someone a free-tier session.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/train_colab.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# RECKON v2 - train Head B on a free-tier GPU

**Read this before running.** The session will die. Colab disconnects and Kaggle
enforces a limit, so this notebook is built around resuming, not around finishing
in one go.

Order of work, from the project brief:

1. **Head B first** (per-page line items). It is the harder task and the one that
   decides whether the project works.
2. Only if Head B beats the B1 baseline on line-item F1 do you move to Head A.
   If it does not, stop and diagnose - that is a gate, not a suggestion.

Set `DRIVE_DIR` or `HUB_REPO` below **before** starting. Local Colab disk is
ephemeral; a checkpoint written only there is a checkpoint you will lose.
"""),
    code("""
# Runtime check. If this says CPU, change Runtime > Change runtime type > T4 GPU.
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "NO GPU")
"""),
    md("""
## 1. Get the code and dependencies

`uv` is used locally; on Colab plain pip is less trouble and resolves to the same
pinned versions from `pyproject.toml`.
"""),
    code("""
REPO_URL = "https://github.com/Abhiii4545/hospital-llm.git"
BRANCH   = "main"

import os, subprocess, sys
if not os.path.exists("hospital-llm"):
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, "hospital-llm"], check=True)
%cd hospital-llm

!pip -q install -e .
!pip -q install "torch" "transformers>=4.44,<5" "accelerate" "bitsandbytes" "sentencepiece"
print("done")
"""),
    md("""
## 2. Mount Drive for checkpoints

Skip only if you are pushing to the Hugging Face Hub instead. Do not skip both.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/reckon/head_b"   # checkpoints land here
HUB_REPO  = None                                      # or "you/reckon-head-b"
import os; os.makedirs(DRIVE_DIR, exist_ok=True)
print("checkpoint target:", DRIVE_DIR)
"""),
    md("""
## 3. Get the corpus

The corpus is ~10k page PNGs and is **not** in git. Either rebuild it here
(slow: roughly 0.8 pages/sec/worker) or copy a prebuilt one from Drive.

Rebuilding on Colab is usually the wrong call - it spends GPU-session wall-clock
on a CPU-bound task. Build it locally, zip it, upload once.
"""),
    code("""
CORPUS_ZIP = "/content/drive/MyDrive/reckon/synthetic.zip"   # prebuilt, preferred

import os, subprocess
if os.path.exists(CORPUS_ZIP):
    !mkdir -p data && unzip -q -o "$CORPUS_ZIP" -d data/
else:
    # Fallback: build a small corpus in-session. Enough to prove the pipeline,
    # NOT enough to train a model worth reporting.
    !playwright install chromium
    !python -m reckon.data.build_corpus --documents 400 --out data/synthetic --workers 2

!wc -l data/synthetic/manifest.jsonl
"""),
    md("""
## 4. Check the target-length distribution BEFORE training

`max_length` is the single setting most likely to silently ruin line-item recall.
If targets are truncated, the model learns to stop early and no error is ever
raised. Measure, then set it.
"""),
    code("""
from transformers import DonutProcessor
from reckon.schema import donut_special_tokens
from reckon.training.dataset import PageDataset

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.tokenizer.add_special_tokens({"additional_special_tokens": donut_special_tokens()})

ds = PageDataset("data/synthetic/manifest.jsonl", head="b", split="train",
                 processor=processor)
lengths = ds.target_lengths(processor.tokenizer)

import numpy as np
lengths = np.array(lengths)
for q in (50, 90, 95, 99, 100):
    print(f"p{q:<3} {np.percentile(lengths, q):7.0f} tokens")
print("\\nSet max_length at or above p99. Truncation is silent.")
"""),
    md("""
## 5. Train Head B

Checkpoints every 200 steps to Drive. If the session dies, **re-run this same
cell** - it resumes from the last checkpoint including optimiser, scheduler and
RNG state, not just the weights.
"""),
    code("""
import yaml, pathlib
cfg = yaml.safe_load(open("reckon/training/configs/head_b.yaml"))
cfg["drive_dir"]  = DRIVE_DIR
cfg["hub_repo"]   = HUB_REPO
cfg["push_to_hub"] = HUB_REPO is not None
cfg["max_length"] = 1024          # <- set from the p99 above
pathlib.Path("configs_run.yaml").write_text(yaml.safe_dump(cfg))

!python -m reckon.training.train --config configs_run.yaml
"""),
    md("""
## 6. Evaluate against the baselines

The number that matters is line-item F1 against **B1**, not against B0. Beating a
regex engine is not the claim; beating a competent week of engineering is.
"""),
    code("""
!python -m reckon.eval.run
!ls -la reports/
"""),
    md("""
## 7. The gate

**Head B must beat B1 on line-item F1 on synthetic-test.** If it does not, stop.
Do not start Head A. Diagnose instead:

- Is `max_length` truncating targets? (step 4)
- Is the loss still falling at the end of the schedule? (more epochs, not more layers)
- Are the special tokens actually registered? Without that the target encoding is
  worse than plain JSON.
- Is one layout collapsing the average? Check the per-layout slice in the report.

Write the number down whichever way it goes.
"""),
    md("""
## 8. Head A, and only then

Same procedure with `reckon/training/configs/head_a.yaml`. Head A is the easier
task; a bounded header block is far shorter than a line-item list.

After both heads exist, the assembly layer joins per-page outputs, deduplicates
rows reprinted across page breaks and reconciles the totals - all plain Python,
already tested, nothing to train.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline: without it the end-of-file-fixer hook rewrites the
    # notebook on every commit, so a regenerated file never matches the
    # committed one.
    OUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

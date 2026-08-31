"""Generate notebooks/train_kaggle.ipynb.

Kaggle differs from Colab in ways that break a Colab notebook outright, so this
is a separate file rather than a branch inside one:

* there is no Drive to mount; the dataset is read-only at /kaggle/input/<slug>/
* /kaggle/working persists during a session and is saved as notebook output, but
  a session killed mid-run can lose it - so checkpoints go to the Hugging Face
  Hub by default, with /kaggle/working as the fallback
* internet is OFF by default and must be enabled in the sidebar, or pip and the
  base-model download both fail
* sessions cap at 12h and the weekly GPU budget is 30h
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/train_kaggle.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# RECKON v2 — train Head B on Kaggle

**Before you run anything**, set these in the right-hand sidebar:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (or P100) |
| Internet | **On** — pip and the base-model download both need it |
| Environment | Latest / Always use latest |

Then **+ Add Input → Datasets** and attach your synthetic corpus.

Kaggle gives 30 GPU-hours a week and a 12-hour session cap. That is comfortable
for Head B; you will not finish and restart the way Colab forces you to. But the
session can still die, so this notebook checkpoints every 200 steps and resumes.
"""),
    code("""
# Confirm the GPU and the attached dataset before spending any time.
import subprocess, glob, os
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
      or "NO GPU — set Accelerator in the sidebar")

inputs = glob.glob("/kaggle/input/*")
print("\\nAttached datasets:", inputs or "NONE — use + Add Input → Datasets")
"""),
    md("""
## 1. Locate the corpus

The dataset is read-only at `/kaggle/input/<slug>/`. If you uploaded the zip
rather than the unpacked folder, it is extracted to `/kaggle/working` first —
`/kaggle/input` cannot be written to.
"""),
    code("""
import glob, os, zipfile, subprocess

CORPUS = None
for root in glob.glob("/kaggle/input/*"):
    if os.path.exists(f"{root}/manifest.jsonl"):
        CORPUS = root; break
    inner = glob.glob(f"{root}/**/manifest.jsonl", recursive=True)
    if inner:
        CORPUS = os.path.dirname(inner[0]); break
    zips = glob.glob(f"{root}/*.zip")
    if zips:
        print("extracting", zips[0], "…")
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall("/kaggle/working/corpus")
        found = glob.glob("/kaggle/working/corpus/**/manifest.jsonl", recursive=True)
        if found:
            CORPUS = os.path.dirname(found[0]); break

assert CORPUS, "No manifest.jsonl found. Is the dataset attached?"
print("corpus:", CORPUS)
print("pages :", sum(1 for _ in open(f"{CORPUS}/manifest.jsonl")))
"""),
    md("""
## 2. Clone the code and install
"""),
    code("""
REPO_URL = "https://github.com/Abhiii4545/hospital-llm.git"

import os, subprocess
os.chdir("/kaggle/working")
if not os.path.exists("hospital-llm"):
    subprocess.run(["git","clone","--depth","1",REPO_URL,"hospital-llm"], check=True)
os.chdir("/kaggle/working/hospital-llm")

!pip -q install -e . 2>&1 | tail -2
!pip -q install "transformers>=4.44,<5" accelerate sentencepiece 2>&1 | tail -2

# bitsandbytes gives 8-bit AdamW, which is part of what makes this fit.
# If it fails to install the trainer falls back to fp32 AdamW and SAYS SO -
# it is not a silent downgrade, because that would resurface later as an OOM.
!pip -q install bitsandbytes 2>&1 | tail -2

import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
"""),
    md("""
## 3. Where checkpoints go

`/kaggle/working` survives the session and is saved as notebook output, but a
session killed mid-run can lose it. Pushing to the Hugging Face Hub is the safe
option: add your token under **Add-ons → Secrets** as `HF_TOKEN`.

Leave `HUB_REPO = None` to keep checkpoints local only — acceptable for a short
run, risky for a long one.
"""),
    code("""
HUB_REPO = None          # e.g. "Abhiii4545/reckon-head-b"
OUT_DIR  = "/kaggle/working/checkpoints/head_b"

import os
os.makedirs(OUT_DIR, exist_ok=True)

if HUB_REPO:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import login; login(os.environ["HF_TOKEN"])
    print("checkpoints ->", HUB_REPO, "and", OUT_DIR)
else:
    print("checkpoints -> %s ONLY. A killed session can lose these." % OUT_DIR)
"""),
    md("""
## 4. Measure the target length BEFORE training

`max_length` is the single setting most likely to silently ruin line-item recall.
Truncated targets teach the model to stop early and **nothing raises**. Measure,
then set it — do not use the default.
"""),
    code("""
from transformers import DonutProcessor
from reckon.schema import donut_special_tokens
from reckon.training.dataset import PageDataset
import numpy as np

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": donut_special_tokens()})

ds = PageDataset(f"{CORPUS}/manifest.jsonl", head="b", split="train",
                 processor=processor)
lengths = np.array(ds.target_lengths(processor.tokenizer))

for q in (50, 90, 95, 99, 100):
    print(f"p{q:<4} {np.percentile(lengths, q):7.0f} tokens")

MAX_LENGTH = int(np.percentile(lengths, 99)) + 16   # p99 with headroom
print(f"\\nMAX_LENGTH = {MAX_LENGTH}")
print(f"would truncate {100*(lengths > MAX_LENGTH).mean():.2f}% of targets")
"""),
    md("""
## 5. Train

Re-run this cell after a disconnect — it resumes from the last checkpoint,
restoring optimiser, scheduler, scaler and RNG state, not just the weights.
Weights alone would silently restart the epoch.
"""),
    code("""
import yaml, pathlib

cfg = yaml.safe_load(open("reckon/training/configs/head_b.yaml"))
cfg["manifest"]    = f"{CORPUS}/manifest.jsonl"
cfg["out_dir"]     = OUT_DIR
cfg["max_length"]  = MAX_LENGTH
cfg["hub_repo"]    = HUB_REPO
cfg["push_to_hub"] = HUB_REPO is not None
pathlib.Path("run.yaml").write_text(yaml.safe_dump(cfg))
print(yaml.safe_dump(cfg))

!python -m reckon.training.train --config run.yaml
"""),
    md("""
## 6. Evaluate, and check the gate

**Head B must beat 0.372 line-item F1.** That is B1 — OCR plus heuristics — on
real OCR'd images, measured in `docs/RESULTS.md`. Not B0's number, and
emphatically not the phase-2 mini-set's 0.987, which came from noise-free text.
"""),
    code("""
import os
os.environ["RECKON_CHECKPOINT"] = OUT_DIR
!python -m reckon.eval.run
!ls -la reports/
"""),
    md("""
## 7. If it does not clear 0.372, stop here

Do **not** start Head A. Diagnose:

- **Truncation?** Re-check step 4. It is the most common cause.
- **Still improving at the end of the schedule?** More epochs, not more layers.
- **Special tokens registered?** Without them the target encoding is *worse* than
  plain JSON — the format is longer in characters and only wins in tokens.
- **One layout collapsing the average?** Check the per-layout slice in the report
  rather than the headline.
- **Precision high, recall low?** That is B1's failure signature (0.82 / 0.24) —
  it means rows are being found on some layouts and not at all on others.

Write the number down whichever way it goes. A result that does not clear the
gate is a finding, not a failure to hide.
"""),
    md("""
## 8. Save the outputs

Kaggle keeps `/kaggle/working` as notebook output only when the notebook is
**saved with "Save & Run All"**. An interactive session that is simply closed
loses it — which is why HUB_REPO is worth setting.
"""),
    code("""
!du -sh /kaggle/working/checkpoints/* 2>/dev/null
!cat reports/*.md 2>/dev/null | head -60
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

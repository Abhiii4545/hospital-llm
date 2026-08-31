# ============================================================================
# RECKON v2 - Kaggle bootstrap. Paste this as ONE cell and run it.
#
# Does setup + verification only. It deliberately STOPS before training so you
# can read MAX_LENGTH first: that value is the single setting most likely to
# silently destroy line-item recall, and a wrong one raises no error at all.
#
# Before running: Session options -> GPU T4 x2, Internet On, and attach the
# dataset via + Add Input.
# ============================================================================
import glob, json, os, subprocess, sys

# --- 1. GPU ---------------------------------------------------------------
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE - set Session options -> Accelerator -> GPU T4 x2")
assert gpu, "No GPU. Training on CPU would take weeks; stop and enable it."

# --- 2. Find the corpus ---------------------------------------------------
CORPUS = None
for root in glob.glob("/kaggle/input/*"):
    hits = glob.glob(f"{root}/**/manifest.jsonl", recursive=True)
    if hits:
        CORPUS = os.path.dirname(hits[0]); break
assert CORPUS, "No manifest.jsonl in /kaggle/input. Attach the dataset."
print("corpus:", CORPUS)

# --- 3. Verify it is COMPLETE ---------------------------------------------
# A partial upload leaves the manifest intact and the images missing, so the
# page count looks right and training dies hours later on a missing file.
rows = [json.loads(l) for l in open(f"{CORPUS}/manifest.jsonl") if l.strip()]
missing = [r for r in rows if not os.path.exists(os.path.join(CORPUS, r["image"]))]
print(f"pages in manifest : {len(rows):,}")
print(f"images present    : {len(rows) - len(missing):,}")
assert not missing, (
    f"{len(missing):,} pages missing, e.g. {[m['image'] for m in missing[:3]]}. "
    "The dataset upload is incomplete - re-upload before training."
)
print("corpus verified complete\n")

# --- 4. Code and dependencies ---------------------------------------------
os.chdir("/kaggle/working")
if not os.path.exists("hospital-llm"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Abhiii4545/hospital-llm.git",
                    "hospital-llm"], check=True)
os.chdir("/kaggle/working/hospital-llm")
sys.path.insert(0, os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.44,<5", "accelerate", "sentencepiece"], check=False)
# 8-bit AdamW is part of what makes this fit in 16GB. If it will not install the
# trainer falls back to fp32 AdamW and SAYS SO rather than failing later as OOM.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
               check=False)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# --- 5. Measure the target length -----------------------------------------
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

print(f"\ntrain pages: {len(lengths):,}")
for q in (50, 90, 95, 99, 100):
    print(f"  p{q:<4} {np.percentile(lengths, q):7.0f} tokens")

MAX_LENGTH = int(np.percentile(lengths, 99)) + 16
print(f"\nMAX_LENGTH = {MAX_LENGTH}")
print(f"would truncate {100 * (lengths > MAX_LENGTH).mean():.2f}% of targets")
print("\nSetup complete. Paste MAX_LENGTH into the training cell.")

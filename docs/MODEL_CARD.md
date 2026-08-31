# Model card — RECKON v2 extraction heads

**Status: not yet trained.** The training code, corpus and evaluation harness are
complete and tested; no weights exist. Every performance field below reads
`not measured`, and will keep reading that until a run produces a number. A model
card with placeholder metrics is worse than one with blanks, because blanks are
obviously incomplete and placeholders get quoted.

---

## Model details

| | |
|---|---|
| Developed by | Aadithya (portfolio project) |
| Model type | OCR-free document understanding, image → structured text |
| Base model | [`naver-clova-ix/donut-base`](https://huggingface.co/naver-clova-ix/donut-base) (MIT) |
| Architecture | Swin encoder + BART decoder, two separately fine-tuned heads |
| Input | Single page image, 960×1280 portrait |
| Output | Donut-style tagged token sequence, parsed to the `reckon.schema` document model |
| Licence | MIT (this project); base model MIT |
| Version | 2.0.0.dev0 — untrained |

### Two heads, on purpose

| Head | Input | Output | Runs on |
|---|---|---|---|
| A | page image | hospital, patient, insurance, totals blocks | page 1, and any page carrying totals |
| B | page image | `line_items[]` for that page only | every page |

**Whole-document single-pass extraction was rejected deliberately.** A multi-page
itemised bill with 40+ line items produces a target sequence far beyond Donut's
practical decoder length, and long-sequence degradation destroys line-item recall
*silently* — the model simply stops early and nothing raises. Splitting by page
and by head keeps every target inside a budget that can be measured (see
`PageDataset.target_lengths`).

Assembly across pages — deduplicating rows reprinted at a page break, reconciling
line items against printed totals — is **plain Python, not a model**. Those
operations have correct answers, and a model would only make them unauditable.

## Intended use

Assisting a human adjudicator with Indian hospital bills and discharge summaries:
extract structured fields, then apply deterministic IRDAI rules.

**Out of scope.** Not for autonomous claim decisions without human review. Not
for clinical decisions. Not validated outside Indian hospital billing. Not
validated on handwritten-only documents.

## Training data

Synthetic corpus: 21 structurally distinct layouts across 7 archetypes, rendered
with Jinja2 + headless Chromium and degraded with Augraphy across four scan-
quality buckets. See [DATASHEET.md](DATASHEET.md).

No real patient documents are used for training. The real corpus exists only for
evaluation, and `real-test` is read exactly twice.

## Evaluation

The harness (`reckon/eval/`) was built **before** the model, deliberately. It
reports, per system and per field:

- exact match, normalized exact match, and CER — **per field, never as a bare
  average**;
- line-item precision/recall/F1 by Hungarian assignment on description
  similarity, with insertions and deletions counted separately;
- TED-based document accuracy and strict full-document exact match;
- **rupee-denominated business error**: median and p95 absolute error on net
  payable, share of documents wrong by more than ₹100, and rupees misadjudicated
  per 1,000 claims;
- coverage-vs-accuracy under abstention, and the auto-processing rate.

### Results

| System | Line-item F1 | Field exact (norm) | Median ₹ error | synthetic-test | real-test |
|---|---|---|---|---|---|
| B0 — v1 rules | not measured | not measured | not measured | — | — |
| B1 — OCR + heuristics | not measured | not measured | not measured | — | — |
| B2 — LiLT | not trained | not trained | not trained | — | — |
| **RECKON v2 (Donut)** | **not trained** | **not trained** | **not trained** | — | — |

Mini-set smoke-test numbers exist in `reports/` but are explicitly **not**
results: that set is noise-free, five layouts, and its renderer and parsers were
written by the same person on the same day.

The headline number for this project, when it exists, is the **gap between
synthetic-test and real-test** — not the synthetic figure.

## Limitations and known risks

- **Untrained.** Everything below is a predicted risk, not an observed one.
- **Synthetic-to-real gap is unmeasured and expected to be large.** The corpus is
  rendered HTML with simulated degradation, not photographs of real paper.
- **Bilingual coverage is narrow.** Only Telugu and Devanagari headers, and only
  in states where those scripts are actually used. Tamil, Kannada, Malayalam,
  Bengali and Gujarati bills are unrepresented.
- **No handwriting recognition.** Handwritten corrections appear in the corpus as
  a visual condition; the model is not trained to read them.
- **Category assignment is partly a normalizer heuristic**, so a prediction of
  "Medicines" scores correct against a truth of `pharmacy`. Defensible, but it
  flatters the normalized metric; the raw-string metric is the control.
- **Room-rent errors are amplified.** Proportionate deduction means a 1% error in
  the room rate moves the payable by more than 1% of the room rent
  (`reckon/eval/sensitivity.py`). Accuracy on `room_rent` matters more than its
  share of the bill suggests.
- **No fairness evaluation has been run.** Names are sampled region-weighted and
  the model may perform unevenly across naming conventions — particularly
  initials-first and mononym patients. This should be a slice in the first real
  evaluation.

## Ethical considerations

The output influences whether a person's medical claim is paid. Two design
consequences:

1. **Adjudication is deterministic and traceable.** Every deduction carries a
   rule id, a clause citation and a human-readable reason; a `Deduction` cannot
   be constructed without them.
2. **Abstention is a first-class output.** Low-confidence fields route to a human
   rather than being guessed, and the coverage curve makes the trade-off explicit
   instead of implicit.

Real patient documents never enter version control, never appear in demos, and
are redacted before storage. See [PRIVACY.md](PRIVACY.md).

## How to reproduce

```bash
uv sync --extra dev
make corpus
```

Then `notebooks/train_colab.ipynb` on a free T4. Every run records its config
SHA-256, git SHA, dirty-tree flag, seed and library versions; a result whose
provenance cannot be regenerated does not go in the table above.

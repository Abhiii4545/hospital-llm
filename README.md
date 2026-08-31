# RECKON v2

Document understanding for Indian hospital bills and discharge summaries, with a
deterministic IRDAI adjudication layer on top.

**RECKON v1 was a rules engine** — OCR, regex, positional heuristics. It worked on
the layouts it had been written against and broke on everything else. Every new
hospital format meant new rules, and the rules interfered with each other as they
accumulated.

**v2 replaces the extraction layer with a trained model and keeps the rules engine
for adjudication**, where deterministic, auditable logic is the correct answer
rather than a compromise: an insurer must be able to point at the clause that
produced a deduction.

> ### The state of this project, stated plainly
>
> **The model is not trained.** The corpus, the evaluation harness, the training
> code, the adjudication engine and the serving layer are complete and tested —
> 500+ tests, 8 enforced import contracts. What does not exist is a set of
> weights, because training needs a GPU session that has not been run.
>
> **There is therefore no results table, and the synthetic-to-real gap — the
> headline number this project is built to produce — is unmeasured.** Every
> performance cell in the docs reads `not measured`. Placeholder numbers would be
> worse than blanks: blanks are obviously incomplete, placeholders get quoted.
>
> What runs today with no GPU: the whole adjudication path, the evaluation
> harness against two baselines, and the corpus generator.

---

## What is actually here

| | |
|---|---|
| **Evaluation harness** | Built *before* the model, deliberately. Per-field / line-item / document / **rupee-denominated** metrics, slices, coverage-vs-accuracy, sensitivity analysis. |
| **Baselines** | B0 (v1 rules, faithfully reimplemented) and B1 (OCR + positional heuristics), both running. |
| **Synthetic corpus** | ~10,000 pages, 21 structurally distinct layouts, 4 scan-quality buckets, controlled messiness. |
| **Training code** | Two Donut heads, resumable to ≤200 steps, sized for a free T4. Written and tested; **not run**. |
| **Adjudication** | IRDAI List I as YAML with clause citations; room-rent capping with proportionate deduction; every deduction traceable. **Runs today.** |
| **Serving** | FastAPI. `/adjudicate` works with no model. ONNX export variants and a latency table that refuses to omit its accuracy column. |
| **Not built** | B2 (LiLT baseline), the real-document corpus, a trained model — and therefore every number. |

## Try the part that works

```bash
uv sync --extra dev --extra serve
```

```bash
uv run uvicorn reckon.serve.api:app --reload
```

Open `http://127.0.0.1:8000/review`. Adjudication runs with no model, no
checkpoint and no GPU, and returns a full audit trail:

```
Gross considered: 62800.00
  -800.00  LIST_I_ATTENDANT (List I, item 13-18 (attendant and ayah charges)) [row 3: attendant charges]
      Attendant / ayah charges are non-payable (IRDAI List I)
  -5000.00  ROOM_RENT_CAP (Policy schedule - room rent capping with proportionate deduction) [row 1: room rent - deluxe]
      Room rate 10000.00/day exceeds the eligible 5000.00/day; this charge is reduced proportionately by factor 0.5000
  ...
Net payable: 31000.00
```

## Commands

```bash
make test
```

`make imports` checks the 8 layer contracts, `make eval` writes an evaluation
report for B0 and B1, `make corpus` builds the synthetic corpus (~50 min at 6
workers), and `make contact-sheet` renders 100 sampled pages for human review.
On Windows use `./make.ps1 <target>` — GNU make is usually absent.

## Design decisions worth defending

Full reasoning in [docs/DECISIONS.md](docs/DECISIONS.md). The ones that shaped
everything:

**Two heads, page-level — not whole-document single-pass.** A 40-item bill
produces a target sequence beyond Donut's practical decoder length, and
long-sequence degradation destroys line-item recall *silently* — the decoder just
stops early and nothing raises. Head A takes header/totals, Head B takes one
page's line items, and assembly is plain Python.

**The evaluation harness was built before the model.** A harness written
afterwards measures what the model happens to be good at.

**Ground truth is the string that is printed.** Formatting and truth come out of
one code path in the generator, so a label cannot disagree with the pixels.

**The business metric is in rupees.** Median and p95 error on net payable, share
of documents wrong by more than ₹100, rupees misadjudicated per 1,000 claims.
Per-field accuracy treats `totals.net_amount` and `hospital.city` as equally
important. They are not.

**Room-rent errors are amplified, and that is measured.** Proportionate deduction
means a 1% misread of the room rate moves the payable by more than 1% of the room
rent — with a discontinuity at the cap that a linear error analysis would miss.
See [reckon/eval/sensitivity.py](reckon/eval/sensitivity.py).

**Deductions cannot exist without a citation.** A `Deduction` raises unless it
carries a rule id, clause and human-readable reason. "An adjudication with no
traceable reason is a bug" is made unrepresentable rather than merely reviewed for.

## Things that went wrong, and what they cost

Kept because they are the honest part of the record.

- **The first `heavy` augmentation bucket produced pages no human could read.**
  That is label noise, not difficulty: 15% of the corpus would have been teaching
  the model to hallucinate text that was not there.
- **The legibility metric written to catch that was itself wrong.** `p50 − p5` on
  a 95%-white page measures histogram spread, and ranked the clean render *below*
  a degraded one. Now `p90 − p2` plus Laplacian variance, calibrated against a
  deliberately destroyed control that a test requires them to reject.
- **B0 does not fail silently on unfamiliar layouts — it fails silently *wrong*.**
  On pipe-delimited tables its fallback regex still matches, dragging `|` into the
  description and picking up an unrelated number as the amount. The bill comes
  back looking plausible. That is a sharper indictment of v1 than "it returned
  nothing".
- **A table header reading `median |err|` silently split the business table.**
  Caught by a test that walks every generated markdown table and checks cell counts.
- **Two corpus bugs were found by looking at images, not by reading code**:
  honorifics that did not match sex, and a Telugu header on a Tamil Nadu hospital.
  Neither would have appeared in a class-balance table.

## Privacy

Real patient documents never enter version control. `data/real/` is line one of
`.gitignore`; a pre-commit hook independently blocks that path *and* Indian PII
patterns in content (`git add -f` defeats an ignore file — it does not defeat a
hook); and CI asserts nothing under it is tracked. Redaction happens *before*
storage, with surrogates of the same length and character class so page geometry —
which is signal — survives. See [docs/PRIVACY.md](docs/PRIVACY.md).

`real-test` is read exactly twice, and both numbers get written down whatever they
say.

## Documentation

- [docs/DECISIONS.md](docs/DECISIONS.md) — every non-obvious decision, with reasons
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — intended use, limitations, blank results
- [docs/DATASHEET.md](docs/DATASHEET.md) — corpus composition, and its errata
- [docs/PRIVACY.md](docs/PRIVACY.md) — the four barriers
- [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) — free-tier training

## Licensing

MIT / Apache-2.0 / BSD only, licence stated at every addition. Base model
`naver-clova-ix/donut-base` is MIT. LayoutLMv3 is CC-BY-NC-SA and is excluded; a
test asserts it appears in no config.

## What comes next

1. Train Head B and check it against **B1**, not B0. Beating a regex engine is not
   the claim; beating a competent week of engineering is.
2. Collect the real corpus and take the first `real-test` read.
3. Build B2 (LiLT). If Donut cannot beat it, that is a finding worth reporting.

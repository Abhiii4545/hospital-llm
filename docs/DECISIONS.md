# Decision log

Architecture decisions fixed by the brief are recorded first, so that later
phases can be checked against them. After that, every decision made *during*
implementation that was not spelled out in the brief is logged with its reason,
so it can be argued with rather than discovered later in a diff.

## Fixed by the brief (not open for re-litigation)

| # | Decision | Note |
|---|---|---|
| A1 | Donut (`naver-clova-ix/donut-base`, MIT), page-level, two heads | Head A header/totals, Head B line items |
| A2 | No whole-document single-pass extraction | 40+ line items exceeds practical decoder length; long-sequence degradation would silently destroy line-item recall |
| A3 | Assembly is plain Python, not a model | Concatenate per page, dedup across page breaks, reconcile totals |
| A4 | Train Head B first | Harder task, and it decides whether the project works |
| A5 | Input resolution starts at 960x1280 | Donut's 2560x1920 default will not fit in 16GB |
| A6 | Schema keys added as tokenizer special tokens before training | |
| A7 | fp16, gradient checkpointing, batch 1, grad accum 8-16, 8-bit AdamW | |
| A8 | Three baselines built before the model: B0 v1 rules, B1 OCR+heuristics, B2 LiLT (MIT) | |
| A9 | LayoutLMv3 is forbidden | CC-BY-NC-SA |
| A10 | Evaluation harness is built before the model | |
| A11 | Adjudication stays a deterministic rules engine | Insurers need auditable logic |
| A12 | OCR for the review UI is presentation-only | Never feeds the model or the metrics |

## Made during Phase 1

### D1 — A dedicated git repository was initialised for the project

The project folder sat inside a git repository rooted at the user's **home
directory**. Every privacy guarantee in this project is defined relative to "the
repo": `.gitignore` line one, the pre-commit hooks, the CI check. All of them
would have been installed into the home repository and would have governed the
wrong tree. `git init` was run inside the project folder. Nothing outside it was
touched.

This also surfaced a real testing hazard, recorded in
`tests/test_provenance.py`: pytest's `tmp_path` is *inside* a repository on this
machine, so a test asserting "a temp directory has no git SHA" passes on CI and
fails here. The test now creates a fresh `git init` instead of assuming.

### D2 — Raw and typed model families, generated from one definition

Section 5 requires every metric twice, on raw strings and normalized values. That
needs two representations. The `Raw*` mirrors are generated from the typed models
rather than hand-written, so a field added in one place appears in both and the
two cannot drift. A test asserts field-name parity.

### D3 — The schema refuses `float`, and refuses nothing else

Money and quantity fields raise on a `float` input. Everything else is permissive
and optional. **An extractor is allowed to be wrong, and a wrong prediction must
stay representable or it cannot be scored.** Business invariants
(`net == gross - discount + tax`) belong to reconciliation and adjudication, not
to the schema. A GSTIN with a bad checksum therefore validates fine as a
*string*; `is_valid_gstin` is a separate, explicit check.

### D4 — Normalization policies that the brief left open

* **Numeric dates are always day-first.** `05/13/2025` returns `None` rather than
  13 May. A miss is more honest than a confidently wrong value.
* **Two-digit years pivot at 70.** 00–69 → 2000s, 70–99 → 1900s.
* **`NIL` and a lone dash in an amount column are zero, not missing.** `N/A` is
  missing. This is billing convention, and it is applied symmetrically to ground
  truth and prediction.
* **Sub-year ages floor to 0 years.** Lossy, but whole years are what an
  adjudication rule keys off. The raw string is retained for the raw metric.
* **Identifiers drop separators** (`POL-1234 5678` → `POL12345678`). Separator
  placement in an ID is presentation.
* **Person names lose honorifics; organisation names keep them.** "Dr." is part
  of "Dr. Rao's Nursing Home". Two different functions, deliberately.
* **An initial's dot becomes a space, not nothing** — otherwise `A.B.C. Naidu`
  collapses to `abc naidu` instead of `a b c naidu`.

Normalization is kept *principled* rather than maximal on purpose. Aggressive
normalization inflates the normalized metric and shrinks the raw-vs-normalized
gap, which is precisely the signal that gap is supposed to carry.

### D5 — `ward_type` is a canonical string, not an enum

The brief lists `ward_type` without a type. Room-rent capping keys off it, so it
needs a closed vocabulary, but promoting it to a schema enum would be a schema
change. It is a `str | None` normalized against `normalize.WARD_TYPES`, with
unrecognised wards mapping to `"other"`. Same reasoning for `hospital_type`
(free string) and `sex` (canonical `male`/`female`/`other`).

**Open question for review:** should `ward_type` and `sex` become proper enums in
the schema? That is a schema change and needs sign-off.

### D6 — Closed-vocabulary *values* also become special tokens

The brief says to add schema keys. `Category` values are emitted verbatim into
the Donut target string and come from a closed set of 12, so each is worth one
decoder step too. They are added as bare tokens, not wrapped in `<s_...>` tags,
because they are values and not keys.

### D7 — Import contracts go beyond the three the brief names

Section 7 names three forbidden directions. Four more were added: `schema` and
`normalize` are leaves, `adjudicate` is model-free, and `eval` cannot import
`training` or `serving` — the harness should not depend on the things it judges.
A test plants a deliberate violation in a throwaway package and requires
import-linter to fail on it, because a contract that has never said no is not
evidence.

### D8 — The PII allowlist is empty; exemptions are inline

Synthetic PII-shaped literals in tests carry a `pii-allow` marker on their own
line rather than the directory being blanket-exempted, so the exemption is
visible exactly where it applies.

### D9 — A PowerShell shim stands in for GNU make

GNU make is not installed on this Windows machine. The canonical `Makefile` is
kept for CI and Colab; `make.ps1` exposes the same target names locally.

## Dependencies so far

| Package | Licence | Why |
|---|---|---|
| pydantic | MIT | Schema |
| pytest | MIT | Tests |
| import-linter | BSD-2-Clause | Layer boundaries |
| pre-commit | MIT | Privacy hooks |
| hatchling | MIT | Build backend |

No dependency is added without its licence stated. MIT / Apache-2.0 / BSD only.

# Datasheet — RECKON v2 synthetic corpus

Following Gebru et al., *Datasheets for Datasets*.

---

## Motivation

**Why was this dataset created?** RECKON v1 was a rules engine that broke on
layout variance. Testing whether a trained model does better needs a corpus with
*structural* variety, and no public corpus of Indian hospital bills exists —
they are medical records.

**Who funded it?** Nobody. Personal portfolio project.

## Composition

**What do instances represent?** One rendered page of a hospital bill or
discharge summary, plus per-page extraction targets for both model heads.

| | |
|---|---|
| Unit | page image (PNG, 794×1123 before augmentation) |
| Target | JSON with `head_a` (header/totals) and `head_b` (line items) |
| Documents | ~6,300 |
| Pages | ~10,000 |
| Layouts | 21, across 7 archetypes |
| Splits | train 80% / val 10% / synth-test 10%, stratified by layout |

**Archetypes.** Large corporate (dense itemised, multi-page); mid-size grouped by
category with subtotals; small nursing home (sparse, stamped); government
(minimal, bilingual); standalone diagnostic centre (test panel); pharmacy
sub-bill (batch and expiry columns); discharge summary with billing annexure.

**What varies, and why.** Column *order*, header placement, totals placement,
table style, rows per page, fonts, date formats, currency decoration, digit
grouping. Column order in particular is varied because it is precisely what a
fixed-order regex cannot survive — cosmetic-only variation would leave v1's
failure mode untested. A test asserts both `rate,quantity` and `quantity,rate`
orderings exist.

**Deliberate messiness**, at the brief's rates:

| Condition | Rate | Why |
|---|---|---|
| Missing UHID | 8% | Real bills omit it |
| Misaligned totals | 5% | Printed totals genuinely disagree with the rows |
| Row duplicated across a page break | 12% | Continuation pages reprint the last row |
| Handwritten correction | 6% | Pen amendments are common |

**Scan quality**, four buckets, non-uniform because real claim intake is:

| Bucket | Share | Simulates |
|---|---|---|
| clean | 20% | good flatbed scan |
| light | 35% | office scanner |
| medium | 30% | photocopy of a fax |
| heavy | 15% | phone capture |

**Is anything missing?** Yes, and it matters:

- No true handwriting — handwritten corrections are rendered in a script font.
- Only Telugu and Devanagari bilingual headers. Tamil, Kannada, Malayalam,
  Bengali and Gujarati are absent.
- No genuinely adversarial layouts (rotated tables, multi-column bills,
  stapled-together bills from different hospitals).
- No real paper. Every degradation is simulated.

**Does it contain confidential or personal data?** **No.** Every value is
generated. Hospital names are invented and a test asserts no real hospital brand
can appear. No real logo, trademark or letterhead is reproduced. GSTINs are
structurally valid with correct check digits but correspond to no real entity.

## Collection

**How was it acquired?** Generated. Values are sampled from distributions chosen
to resemble Indian hospital billing rather than library defaults: mononyms and
initials-first Telugu names, room rent correlated with ward class, long-tailed
pharmacy rates, insurer-specific policy-number formats.

**The critical property:** ground truth is *the string that gets printed*.
Formatting and truth come out of one code path, so a label cannot disagree with
the pixels.

**Was anyone paid?** No human labelling was involved, and that is a limitation:
there is no human-verified ground truth, so a systematic generation bug would be
invisible to the labels. The contact-sheet review exists to catch exactly that,
and did — see below.

## Preprocessing

Rendered with Jinja2 → headless Chromium (Playwright, Apache-2.0), then degraded
with Augraphy (MIT). Both the clean render and the degraded page pass two
legibility guards (ink/paper contrast, Laplacian edge energy); a page failing
either falls back to the clean render and the fallback is counted.

**Raw data is not kept** — the corpus is regenerated deterministically from a
seed. The manifest carries a SHA-256 per page and an order-independent corpus
hash, so changed data invalidates old results automatically.

## Uses

**Used for:** training the two Donut heads; evaluating all systems on
synthetic-test.

**Should not be used for:** claiming real-world performance. The synthetic-test
number is the optimistic bound. The honest number is the gap between it and
`real-test`, and that gap is this project's headline result.

**Known bias risks.** Region weighting is a guess, not a census. Ward-class
distribution is invented. Amounts are plausible but not calibrated against real
tariffs. A model tuned hard on this corpus will be tuned on those guesses.

## Distribution and maintenance

Not distributed. Regenerated from `reckon/data/build_corpus.py` at a given seed
and git SHA. `data/synthetic/` is gitignored — ~10k PNGs do not belong in a
repository, and the generator is the artefact worth versioning.

## Public insurer samples — investigated, and they do not work

The brief lists "publicly posted sample bills from insurer websites" as a source
for the real corpus. That was searched and one candidate fetched and read, so the
conclusion is measured rather than assumed:

**What is publicly posted by Indian insurers and TPAs is claim FORMS, not
itemised hospital bills.** The most promising candidate - a TPA's "Sample filled
Claim Form" - is a *Claim Acknowledgment Sheet* whose fields are filled with
placeholders (`Insured Name: XYZ`, `Patient Name: PQR`, `Policy No: 12345678`,
`Mobile: XXXXXXXXXX`). Text extraction finds no `particulars`, `s.no`,
`quantity`, `rate`, `room rent` or `gstin` - there is **no itemised billing table
in it at all**.

That matters because the line-item table is the core of what this system
extracts, and the part Head B exists for. A claim form overlaps our schema only
on a few header fields, and its values are dummies.

**Conclusion: public insurer material cannot serve as `real-dev` or
`real-test`.** The real corpus has to come from genuine documents with recorded
consent, per section 4.2. Nothing downloadable substitutes for it, and this is
recorded so the search is not repeated.

## Errata — bugs found by looking at the images

Recorded because they are the argument for the contact-sheet review step, and
none would have shown up in a class-balance table:

1. **Honorifics did not match sex** — `Ms. Rohit Kumar / Sex: F`. Fixed; sex is
   sampled first and the name follows.
2. **A Telugu bilingual header appeared on a Tamil Nadu hospital.** Fixed;
   bilingual layouts are restricted to states whose script the project has.
3. **The `heavy` bucket produced pages no human could read.** That is label
   noise, not difficulty — 15% of the corpus would have been teaching the model
   to hallucinate. Tuned down, and two legibility guards added.
4. **The legibility metric written to catch (3) was itself wrong** — `p50 − p5`
   measures histogram spread on a mostly-white page and ranked the clean render
   *below* a degraded one. Replaced with `p90 − p2` plus Laplacian variance, both
   calibrated against a deliberately destroyed control that a test requires them
   to reject.

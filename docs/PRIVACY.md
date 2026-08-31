# Privacy policy for RECKON v2

Real patient documents exist in this project's workflow. They are the reason the
project has any credibility, and they are also the thing most likely to cause
real harm if mishandled. This document is the operational rule, not an aspiration.

## The four barriers

Each barrier is independent, so no single mistake is sufficient to leak data.

1. **`.gitignore`, line one.** `data/real/` is the first line of the file,
   written before any other file in the repository existed.
2. **`tools/hooks/block_real_data.py`.** A pre-commit hook that refuses any
   staged path containing a `data/real` segment, at any depth. This exists
   because `git add -f` overrides `.gitignore`; the hook does not care.
   Its behaviour under a forced add is covered by a test.
3. **`tools/hooks/pii_scan.py`.** A pre-commit hook that scans staged *content*
   for Aadhaar, PAN, GSTIN, Indian mobile numbers and email addresses. Leaked
   PII usually arrives pasted into a notebook, a fixture or a debug log rather
   than as a file in the obvious directory.
4. **CI.** `git ls-files | grep data/real/` must be empty on every push, which
   catches anything force-added in a commit that predates the hooks.

The hook's own error output redacts what it matched. A hook that prints the
Aadhaar number it found has simply moved the leak into the terminal scrollback
and the CI log.

## Redaction happens before storage, not after

A document is redacted **before** it is written into `data/real/`, never
afterwards. Names, UHIDs, policy numbers, phone numbers and addresses are
replaced with realistic surrogates **of the same length and character class**, so
that page geometry — line wrapping, column widths, where a field overflows — is
preserved. Geometry is signal for the model; destroying it would make the real
corpus unrepresentative of the documents the system will actually see.

The mapping from surrogate back to original is one-way and is kept outside the
repository. Each redaction is recorded in a manifest.

## Consent

Every real document requires explicit, recorded permission from the person whose
data it is. Permissions live in `data/real/CONSENT.md` (gitignored), one entry
per source, with the date and the scope of what was agreed.

## Splits, and the rule about looking

| Split | Size | May I look at it? |
|---|---|---|
| `real-dev` | 50 docs | Yes. Error analysis and threshold tuning happen here. |
| `real-test` | 100+ docs | **No.** Evaluated exactly twice: once at Phase 6, once at the end. |

`real-test` is never trained on, never tuned against, and individual failures in
it are not inspected until the project is finished. Both reads are written down
whatever the numbers are. The gap between synthetic-test and real-test is the
headline result of this project, and it is only meaningful if the lock holds.

## What never leaves the repository

No real document — redacted or otherwise — goes into the demo, the README, a
Weights & Biases artifact, a screenshot, or a slide. Demos use synthetic
documents exclusively.

## If a leak happens

Rotate nothing, panic about nothing, and do this in order: stop, remove the file
from the working tree, rewrite history for every commit that contains it, force
push, and notify the person whose data it was. A leaked commit that is merely
reverted is still public.

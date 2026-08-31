## Results — untrained systems, synthetic-test

The first real numbers in this project. **No trained model exists**, so this is
what the system does today with no learning at all.

| System | Line-item F1 | P / R | TED acc | Strict | Median Rs error | % err > Rs 100 |
|---|---|---|---|---|---|---|
| B0 — v1 rules | not run on images | — | — | — | — | — |
| **B1 — OCR + heuristics** | **0.372** | 0.82 / 0.24 | 0.221 | 0.00 | **Rs 45,367** | **90%** |
| **B3 — Donut-CORD zero-shot** | **0.276** | 0.34 / 0.23 | 0.121 | 0.00 | not scored | not scored |
| B2 — LiLT | not trained | — | — | — | — | — |
| **RECKON v2 (Donut)** | **not trained** | — | — | — | — | — |

B1: 60 pages, real OCR (RapidOCR / PaddleOCR models), 18.53s per page on CPU.
B3: 20 pages, 5 of which collapsed into decoder repetition loops.

### The number that matters

**B1 misadjudicates the median claim by Rs 45,367, and gets
90% of documents wrong by more than Rs 100.** Not usable at any
abstention threshold.

Its precision is 0.82 but its recall is 0.24: it finds rows on
layouts its header detection understands and finds *nothing* on the rest. That is
the RECKON v1 failure mode, reproduced and measured.

### Header matching, before and after

Driving the review UI exposed first-match-wins in B1's header matching (`age`
came out as the string "Patient Name"). Fixing it to best-score-wins was then
**measured rather than assumed**:

| metric | before | after |
|---|---|---|
| line-item F1 | 0.3719 | 0.3719 |
| line-item precision / recall | 0.82 / 0.24 | 0.82 / 0.24 |
| TED accuracy | 0.2211 | **0.2265** |

Line-item numbers are byte-identical, which is the right answer: the change
touched header fields, not row parsing. Document accuracy improved slightly. A
single page had looked like a regression by eye; the benchmark says it was not.

### Why the phase 2 mini-set number was misleading

B1 scored **0.987** line-item F1 on the mini-set and **0.372** here. The
mini-set's text was noise-free, so its OCR stage never ran. A 62-point collapse
is the cost of that shortcut, and it is why the mini-set is labelled a smoke-test
everywhere it appears rather than a benchmark.

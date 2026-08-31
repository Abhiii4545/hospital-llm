# RECKON v2

Document understanding for Indian hospital bills and discharge summaries, with a
deterministic IRDAI adjudication layer on top.

**RECKON v1 was a rules engine** — OCR plus regex plus positional heuristics. It
worked on the layouts it had been written against and broke on everything else;
each new hospital format meant new rules. **v2 replaces the extraction layer with
a trained model** and keeps the rules engine only for adjudication, where
deterministic, auditable logic is the correct answer rather than a compromise.

The primary deliverable of this project is not the model. It is an honest,
measured comparison between v1 and v2 — **including where v2 is worse**. The
evaluation harness is built before the model, and the headline number is the gap
between synthetic-test and real-test performance, not the synthetic number alone.

---

## Status

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Skeleton, schema, normalizers, privacy hooks, CI | complete |
| 2 | Eval harness + B0 (v1 rules) + B1 (OCR heuristics) | **complete** |
| 3 | Synthetic corpus, 8–12k pages | **complete** |
| 4 | Head B — line items | not started |
| 5 | Head A + assembly + ablations | not started |
| 6 | B2 (LiLT) + first locked real-test read | not started |
| 7 | Error taxonomy on real-dev | not started |
| 8 | Serving, ONNX variants, demo, docs | not started |

No results table appears here yet. Phase 2 produces numbers, but only on a
20-document **smoke-test** whose text is noise-free and whose renderer and
parsers were written by the same person on the same day. Those numbers measure
the harness, not the system, and are deliberately not promoted to this page.
Real numbers begin at Phase 3 (synthetic corpus) and Phase 6 (locked real-test).

## Quick start

```bash
uv sync --extra dev
```

Then, on Linux/macOS:

```bash
make test
```

On Windows (GNU make is usually absent):

```bash
./make.ps1 test
```

## Layout

```
reckon/
  schema.py       single source of truth for the document schema
  normalize.py    pure normalization functions (dates, money, names, IDs)
  provenance.py   git SHA / config hash / library versions / seeding
  data/           synthetic corpus generation          (phase 3)
  models/         Donut heads, assembly, baselines     (phases 2, 4, 5, 6)
  training/       resumable training loop              (phase 4)
  eval/           metrics, slices, sensitivity, report (phase 2)
  adjudicate/     IRDAI rules engine                   (phase 6)
  serve/          FastAPI, ONNX export, latency bench  (phase 8)
```

Layer boundaries are enforced mechanically by `import-linter` in CI, not by
convention: the data layer cannot import training, training cannot import
serving, and serving cannot import data generation.

## Privacy

Real patient documents exist in this project's workflow and **never** enter
version control. `data/real/` is the first line of `.gitignore`, and a
pre-commit hook independently blocks both that path and a set of Indian PII
patterns so that `git add -f` cannot bypass the ignore file. See
[docs/PRIVACY.md](docs/PRIVACY.md).

## Licensing

Only MIT / Apache-2.0 / BSD models and datasets are used. Every dependency is
introduced with its licence stated. LayoutLMv3 (CC-BY-NC-SA) is excluded.

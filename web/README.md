# RECKON review UI

Next.js 16 + React 19 front end for the RECKON v2 FastAPI backend.

## What it does

- Loads a bill page (drag, file picker, or **Try a sample** from the synthetic corpus)
- Runs extraction — heuristics today, the trained Donut heads once a checkpoint exists
- Shows every extracted field, editable, with the OCR box highlighted on the page
- Runs IRDAI adjudication and shows the full audit trail, with each deduction's
  rule id and clause pinned to the line it came from
- Logs every correction as future training data

## Run it

The backend must be running first:

```bash
uv run uvicorn reckon.serve.api:app --port 8000
```

```bash
npm install && npm run dev
```

Point it elsewhere with `NEXT_PUBLIC_RECKON_API` (see `.env.example`). The
`NEXT_PUBLIC_` prefix is required — Next only inlines those into the browser
bundle, and a bare `RECKON_API` is `undefined` client-side.

## Two things this UI is deliberate about

**It shows how unreliable the extraction is.** The banner carries the measured
line-item F1 and median rupee error for whichever engine ran. A review tool that
presents heuristic output as though it were trustworthy is worse than no tool:
the reviewer stops reviewing.

**Highlight boxes are presentation only.** Donut emits no coordinates; the boxes
come from a separate OCR pass. They are never fed back into the model and never
used to compute a metric.

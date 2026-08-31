"""Score off-the-shelf Donut-CORD zero-shot against the synthetic corpus.

Answers "can we skip fine-tuning and just use a HuggingFace model?" with a
measurement. Runs on CPU at roughly 50s/page, so it samples rather than sweeps.
"""

from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path

from PIL import Image

from reckon.eval.metrics import DocumentPair, score_fields, score_line_items, score_documents
from reckon.models.baselines.b3_donut_cord import B3DonutCord, has_repetition_loop
from reckon.schema import RawDocument, RawLineItem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--out", default="reports/b3_zeroshot.json")
    args = ap.parse_args()

    paths = sorted(glob.glob("data/synthetic/pages/*_p00.png"))[: args.pages]
    if not paths:
        print("no corpus pages", file=sys.stderr)
        return 1

    engine = B3DonutCord()
    pairs, loops, raw_samples = [], 0, []

    for i, path in enumerate(paths, 1):
        target_path = Path(path).with_suffix(".json")
        if not target_path.exists():
            continue
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        head_a = payload.get("head_a", {})

        truth = RawDocument()
        for block in ("hospital", "patient", "insurance", "totals"):
            if block in head_a:
                model = type(getattr(truth, block))
                setattr(truth, block, model(**head_a[block]))
        truth.line_items = [RawLineItem(**r) for r in payload["head_b"]["line_items"]]

        t0 = time.perf_counter()
        text = engine.run_raw(Image.open(path).convert("RGB"))
        dt = time.perf_counter() - t0
        if has_repetition_loop(text):
            loops += 1
        if len(raw_samples) < 2:
            raw_samples.append(text[:400])

        from reckon.models.baselines.b3_donut_cord import parse_cord_sequence
        pairs.append(DocumentPair(Path(path).stem, parse_cord_sequence(text), truth, {}))
        print(f"[{i}/{len(paths)}] {Path(path).stem} {dt:.0f}s loop={has_repetition_loop(text)}",
              flush=True)

    li = score_line_items(pairs)
    docs = score_documents(pairs)
    fields = score_fields(pairs)
    result = {
        "system": "B3 (Donut-CORD zero-shot)",
        "model": engine.model_name,
        "pages": len(pairs),
        "line_item_precision": round(li.precision, 4),
        "line_item_recall": round(li.recall, 4),
        "line_item_f1": round(li.f1, 4),
        "line_items_true": li.n_true,
        "line_items_pred": li.n_pred,
        "line_items_matched": li.n_matched,
        "ted_accuracy": round(docs.ted_accuracy, 4),
        "strict_exact": round(docs.strict_exact_match, 4),
        "repetition_loops": loops,
        # Both numbers, always. The normalized column credits a correct
        # absence, so a system extracting nothing scores the absence rate;
        # present-only cannot be earned by abstaining.
        "field_normalized_flattered": {
            k: round(v.normalized_exact, 4) for k, v in fields.items()
        },
        "field_present_only": {
            k: round(v.accuracy_when_present, 4) for k, v in fields.items()
        },
        "field_support": {k: v.truth_present for k, v in fields.items()},
        "raw_samples": raw_samples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # trailing newline so the end-of-file hook does not rewrite it
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("field_exact_normalized", "raw_samples")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score B1 (OCR + heuristics) on real page IMAGES, not on perfect text.

B1's only previous number came from the phase 2 mini-set, whose text was
noise-free. That was an upper bound it cannot reach in practice, and quoting it
would make the comparison against a trained model too flattering to the baseline.
This runs the real OCR stack over corpus images and scores with the same harness.

It also answers a question worth answering before spending GPU hours: how far can
you get with NO trained model at all?
"""

from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path

from reckon.eval.metrics import (DocumentPair, score_business, score_documents,
                                 score_fields, score_line_items)
from reckon.models.baselines.b1_ocr_heuristic import B1OcrHeuristic
from reckon.models.baselines.ocr_rapid import RapidOcrBackend
from reckon.schema import RawDocument, RawLineItem


def load_truth(page_png: str) -> RawDocument | None:
    target = Path(page_png).with_suffix(".json")
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    truth = RawDocument()
    for block, values in payload.get("head_a", {}).items():
        model = type(getattr(truth, block))
        setattr(truth, block, model(**values))
    truth.line_items = [RawLineItem(**r) for r in payload["head_b"]["line_items"]]
    return truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--out", default="reports/b1_on_images.json")
    args = ap.parse_args()

    manifest = Path("data/synthetic/manifest.jsonl")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == "synth_test"][: args.pages]
    if not rows:
        print("no synth_test pages", file=sys.stderr)
        return 1

    engine = B1OcrHeuristic(backend=RapidOcrBackend())
    pairs, times = [], []
    for i, row in enumerate(rows, 1):
        png = "data/synthetic/" + row["image"]
        truth = load_truth(png)
        if truth is None:
            continue
        t0 = time.perf_counter()
        pred = engine.extract(png)
        times.append(time.perf_counter() - t0)
        pairs.append(DocumentPair(row["page_id"], pred, truth, {
            "quality": row.get("quality_effective", row.get("quality")),
            "layout": row.get("layout"), "language": row.get("language"),
        }))
        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] {times[-1]:.1f}s/page", flush=True)

    li = score_line_items(pairs)
    docs = score_documents(pairs)
    fields = score_fields(pairs)
    biz = score_business(pairs)

    present = {k: v.accuracy_when_present for k, v in fields.items() if v.truth_present}
    result = {
        "system": "B1 (OCR + heuristics) on IMAGES",
        "ocr": "RapidOCR (PaddleOCR models via ONNXRuntime, Apache-2.0)",
        "pages": len(pairs),
        "sec_per_page": round(sum(times) / max(1, len(times)), 2),
        "line_item_precision": round(li.precision, 4),
        "line_item_recall": round(li.recall, 4),
        "line_item_f1": round(li.f1, 4),
        "line_items_true": li.n_true, "line_items_pred": li.n_pred,
        "line_items_matched": li.n_matched,
        "insertions": li.insertions, "deletions": li.deletions,
        "ted_accuracy": round(docs.ted_accuracy, 4),
        "strict_exact": round(docs.strict_exact_match, 4),
        "median_rupee_error": str(biz.median_error),
        "p95_rupee_error": str(biz.p95_error),
        "pct_error_over_100": round(biz.pct_error_over_threshold, 4),
        "field_present_only": {k: round(v, 4) for k, v in sorted(
            present.items(), key=lambda x: -x[1])},
        "attribute_accuracy": {k: round(v, 4) for k, v in li.attribute_accuracy.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("field_present_only", "attribute_accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

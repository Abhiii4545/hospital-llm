"""Error taxonomy: classify failures so they can be fixed rather than admired.

Phase 7 of the brief takes 30 failures from `real-dev` (never `real-test`) and
builds a taxonomy with a hypothesised cause and a proposed fix for each. This
module does the mechanical half - grouping failures into categories that suggest
different fixes - so the human half is spent on the cases that are genuinely
ambiguous instead of on sorting.

The categories are chosen so that each one implies a DIFFERENT action. A taxonomy
whose buckets all lead to "train longer" is not a taxonomy.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from rapidfuzz.distance import Levenshtein

from reckon.eval.metrics import DocumentPair, EvalCache, match_line_items
from reckon.normalize import normalize_amount, normalize_date
from reckon.schema import FIELD_PATHS

__all__ = ["ErrorCase", "CATEGORIES", "classify_field_error", "collect",
           "summarise", "write_taxonomy"]

#: category -> (what it means, what to do about it). The second half is the
#: point: a bucket that does not imply an action is not worth having.
CATEGORIES: dict[str, tuple[str, str]] = {
    "formatting_only": (
        "raw strings differ, normalized values agree",
        "not a model error - check the normalizer is not over-reaching",
    ),
    "missed_field": (
        "truth had a value, prediction had none",
        "recall problem: check the field is visible at the input resolution, "
        "and that the target was not truncated",
    ),
    "hallucinated_field": (
        "truth had no value, prediction invented one",
        "precision problem: the model has learned the field is always present; "
        "confirm absent-field examples are in the training split",
    ),
    "digit_confusion": (
        "numeric value off by a small character edit (5/6, 1/7, 0/8)",
        "resolution or augmentation: raise input resolution, or reduce the "
        "degradation intensity that destroys thin strokes",
    ),
    "magnitude_error": (
        "numeric value wrong by an order of magnitude",
        "digit grouping misread - likely Indian lakh grouping; add targeted "
        "examples and check the normalizer",
    ),
    "date_unparsed": (
        "predicted date string does not parse",
        "add the observed date format to normalize.normalize_date, or the layout "
        "to the corpus",
    ),
    "wrong_field": (
        "predicted value belongs to a different field of the same document",
        "layout confusion: the two fields are adjacent; add layouts that "
        "separate them differently",
    ),
    "value_confusion": (
        "both present, genuinely different values",
        "the hard bucket - inspect individually",
    ),
}

_DIGIT_LOOKALIKES = (("5", "6"), ("1", "7"), ("0", "8"), ("3", "8"), ("9", "4"))


@dataclass
class ErrorCase:
    doc_id: str
    field: str
    truth: str | None
    predicted: str | None
    category: str
    note: str = ""
    meta: dict = field(default_factory=dict)


def _is_digit_confusion(truth: str, predicted: str) -> bool:
    if len(truth) != len(predicted):
        return False
    if Levenshtein.distance(truth, predicted) > 2:
        return False
    differing = [(a, b) for a, b in zip(truth, predicted) if a != b]
    if not differing:
        return False
    return all(
        (a, b) in _DIGIT_LOOKALIKES or (b, a) in _DIGIT_LOOKALIKES
        or (a.isdigit() and b.isdigit())
        for a, b in differing
    )


def classify_field_error(
    field_path: str,
    raw_truth: str | None,
    raw_pred: str | None,
    norm_truth: object,
    norm_pred: object,
    other_truth_values: dict[str, str] | None = None,
) -> str | None:
    """Category for one field disagreement, or None when it is not an error."""
    if raw_truth == raw_pred:
        return None
    if norm_truth == norm_pred:
        return "formatting_only"

    if raw_truth is not None and raw_pred is None:
        return "missed_field"
    if raw_truth is None and raw_pred is not None:
        return "hallucinated_field"

    truth, pred = str(raw_truth), str(raw_pred)

    # Did the model put another field's value here?
    for other_path, other_value in (other_truth_values or {}).items():
        if other_path != field_path and other_value and other_value == pred:
            return "wrong_field"

    if "date" in field_path:
        if normalize_date(pred) is None:
            return "date_unparsed"

    truth_amount, pred_amount = normalize_amount(truth), normalize_amount(pred)
    if truth_amount is not None and pred_amount is not None and truth_amount != 0:
        ratio = abs(pred_amount / truth_amount)
        if ratio >= Decimal("9") or ratio <= Decimal("0.12"):
            return "magnitude_error"

    if _is_digit_confusion(truth, pred):
        return "digit_confusion"

    return "value_confusion"


def collect(pairs: Sequence[DocumentPair], limit: int | None = None) -> list[ErrorCase]:
    """Every field-level failure across *pairs*, classified."""
    cache = EvalCache()
    cases: list[ErrorCase] = []

    for pair in pairs:
        npred, ntruth = cache.normalized(pair)
        truth_values = {
            path: getattr(getattr(pair.truth, path.split(".")[0]), path.split(".")[1])
            for path in FIELD_PATHS
        }
        for path in FIELD_PATHS:
            block, name = path.split(".", 1)
            raw_truth = getattr(getattr(pair.truth, block), name)
            raw_pred = getattr(getattr(pair.pred, block), name)
            category = classify_field_error(
                path, raw_truth, raw_pred,
                getattr(getattr(ntruth, block), name),
                getattr(getattr(npred, block), name),
                truth_values,
            )
            if category:
                cases.append(ErrorCase(
                    doc_id=pair.doc_id, field=path,
                    truth=raw_truth, predicted=raw_pred,
                    category=category, meta=dict(pair.meta),
                ))

        # Line items: recall failures are the ones that matter operationally.
        matches = match_line_items(npred.line_items, ntruth.line_items)
        matched_truth = {j for _, j, _ in matches}
        for index, item in enumerate(ntruth.line_items):
            if index not in matched_truth:
                cases.append(ErrorCase(
                    doc_id=pair.doc_id, field="line_items[]",
                    truth=item.description, predicted=None,
                    category="missed_field",
                    note="line item never produced",
                    meta=dict(pair.meta),
                ))

    if limit:
        cases = cases[:limit]
    return cases


#: Metadata keys that identify a page rather than describe a population. Slicing
#: by these produces one bucket per document, which is noise, not a breakdown.
_NOT_A_SLICE = frozenset({
    "page_id", "doc_id", "image", "targets", "sha256", "page", "total_pages",
    "n_line_items", "ink_contrast", "edge_energy",
})

#: A "slice" with more distinct values than this is an identifier in disguise.
_MAX_SLICE_CARDINALITY = 30


def summarise(cases: Sequence[ErrorCase]) -> dict[str, Counter]:
    """Failure counts by category, by field, and by every usable metadata slice.

    Slice keys are DISCOVERED rather than hard-coded. They were hard-coded once
    and silently produced no slice tables at all, because the corpus calls the
    key `template` and this file was looking for `layout`.
    """
    by_category = Counter(c.category for c in cases)
    by_field = Counter(c.field for c in cases)

    candidates: dict[str, Counter] = defaultdict(Counter)
    for case in cases:
        for key, value in case.meta.items():
            if key in _NOT_A_SLICE or isinstance(value, (dict, list)):
                continue
            candidates[key][str(value)] += 1

    by_slice = {
        key: counter for key, counter in candidates.items()
        if 1 < len(counter) <= _MAX_SLICE_CARDINALITY
    }
    return {"category": by_category, "field": by_field, **by_slice}


def write_taxonomy(
    cases: Sequence[ErrorCase],
    out_path: Path | str = "reports/error_taxonomy.md",
    source: str = "unspecified",
    sample: int = 30,
) -> Path:
    """Write the taxonomy, with a slot per case for a human hypothesis and fix."""
    counts = summarise(cases)
    lines: list[str] = [
        "# Error taxonomy\n",
        f"_Source: **{source}**. {len(cases)} failures collected._\n",
        "> Failures are drawn from `real-dev` only. `real-test` is read exactly",
        "> twice and its individual failures are never inspected until the",
        "> project is finished.\n",
        "## By category\n",
        "| category | n | share | what it means | what to do |",
        "|---|---|---|---|---|",
    ]
    total = max(1, len(cases))
    for category, n in counts["category"].most_common():
        meaning, action = CATEGORIES.get(category, ("?", "?"))
        lines.append(f"| `{category}` | {n} | {n / total:.1%} | {meaning} | {action} |")

    lines.append("\n## By field\n")
    lines.append("| field | failures |")
    lines.append("|---|---|")
    for field_path, n in counts["field"].most_common(15):
        lines.append(f"| `{field_path}` | {n} |")

    for key in ("layout", "quality", "language", "messiness"):
        if key in counts and counts[key]:
            lines.append(f"\n## By `{key}`\n")
            lines.append(f"| {key} | failures |")
            lines.append("|---|---|")
            for value, n in counts[key].most_common(12):
                lines.append(f"| {value} | {n} |")

    lines.append(f"\n## Individual cases (first {sample})\n")
    lines.append("Hypothesis and fix are for a human to fill in. The classifier")
    lines.append("groups; it does not diagnose.\n")
    for index, case in enumerate(cases[:sample], start=1):
        lines.append(f"### {index}. `{case.field}` — {case.category}\n")
        lines.append(f"- document: `{case.doc_id}`")
        lines.append(f"- truth: `{case.truth!r}`")
        lines.append(f"- predicted: `{case.predicted!r}`")
        if case.note:
            lines.append(f"- note: {case.note}")
        if case.meta.get("layout"):
            lines.append(
                f"- slice: layout={case.meta.get('layout')}, "
                f"quality={case.meta.get('quality')}, "
                f"messiness={case.meta.get('messiness')}"
            )
        lines.append("- **hypothesised cause:** _TODO_")
        lines.append("- **proposed fix:** _TODO_\n")

    lines.append("## Fixes implemented\n")
    lines.append("The brief allows exactly TWO fixes, chosen by value, then a")
    lines.append("re-measure. Implementing every fix at once makes it impossible")
    lines.append("to attribute the change.\n")
    lines.append("| # | fix | rationale | measured effect |")
    lines.append("|---|---|---|---|")
    lines.append("| 1 | _TODO_ | _TODO_ | _not measured_ |")
    lines.append("| 2 | _TODO_ | _TODO_ | _not measured_ |")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

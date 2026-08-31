"""Evaluation metrics for RECKON v2.

Built before any model exists, which is the point: a harness written after the
fact tends to measure what the model happens to be good at.

Four families, per section 5 of the brief:

* field level      - exact, normalized exact, CER. Reported PER FIELD, never as a
  bare average, because an average hides that ``totals.net_amount`` sits at 0.71
  while ``hospital.name`` sits at 0.99.
* line-item level  - Hungarian assignment on description similarity, then
  precision/recall/F1 and per-attribute accuracy within matched rows.
  Insertions and deletions are reported separately: a hallucinated row and a
  dropped row cost an insurer very different things.
* document level   - TED-based accuracy and strict full-document exact match.
  The strict number is brutal. It is reported anyway.
* business level   - rupees. This is the metric that means anything to the
  person paying for the system.

Every metric is computed twice, on raw strings and on normalized values. The gap
is how much of the error is mere formatting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from reckon.eval.ted import json_to_tree, ted_accuracy
from reckon.normalize import normalize_document, normalize_text
from reckon.schema import (
    FIELD_PATHS,
    LINE_ITEM_FIELDS,
    Document,
    LineItem,
    RawDocument,
)

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "MATERIAL_ERROR_RUPEES",
    "DocumentPair",
    "FieldScore",
    "LineItemScore",
    "DocumentScore",
    "BusinessScore",
    "CoveragePoint",
    "score_fields",
    "score_line_items",
    "score_documents",
    "score_business",
    "coverage_curve",
    "auto_processing_rate",
    "deployment_sentence",
    "default_net_payable",
    "deduction_total",
    "median",
    "percentile",
]

#: Description similarity below which a Hungarian pairing is NOT counted as a
#: match. Hungarian assignment always returns a full pairing, so without a floor
#: two totally unrelated rows would be scored as a matched pair.
DEFAULT_MATCH_THRESHOLD = 0.60

#: Net-payable error above which a document is "materially wrong" (section 5).
MATERIAL_ERROR_RUPEES = Decimal("100")


class DocumentPair(NamedTuple):
    """One evaluation unit: a prediction and its ground truth, both verbatim.

    Both sides are ``RawDocument``. Normalized values are DERIVED here, using the
    same functions the pipeline uses, so a metric can never be computed under a
    different rule than the one the pipeline applied.
    """

    doc_id: str
    pred: RawDocument
    truth: RawDocument
    meta: Mapping[str, object] = {}


# --------------------------------------------------------------------------
# small numeric helpers - Decimal-safe, no float money anywhere
# --------------------------------------------------------------------------

def median(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile(values: Sequence[Decimal], q: float) -> Decimal:
    """Nearest-rank percentile. Stated explicitly because p95 definitions differ."""
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _get(doc: RawDocument | Document, path: str) -> object:
    block, name = path.split(".", 1)
    return getattr(getattr(doc, block), name)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _cer(pred: object, truth: object) -> float:
    """Character error rate against the truth string.

    An empty truth cannot have a rate expressed per truth-character, so it is
    defined as 0 when the prediction is also empty and 1 otherwise, and capped
    at 1 so a long hallucination cannot dominate an average.
    """
    p, t = _as_text(pred), _as_text(truth)
    if not t:
        return 0.0 if not p else 1.0
    return min(1.0, Levenshtein.distance(p, t) / len(t))


# --------------------------------------------------------------------------
# field level
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldScore:
    path: str
    n: int
    exact: float             # raw string match
    normalized_exact: float  # match after normalization
    cer: float
    truth_present: int
    pred_present: int
    misses: int              # truth had a value, prediction did not
    hallucinations: int      # truth had none, prediction invented one

    @property
    def formatting_gap(self) -> float:
        """How much of the raw error was only formatting."""
        return self.normalized_exact - self.exact


def score_fields(pairs: Sequence[DocumentPair]) -> dict[str, FieldScore]:
    """Per-field scores. Returns a dict keyed by field path - never an average."""
    normalized = [
        (normalize_document(p.pred), normalize_document(p.truth)) for p in pairs
    ]

    scores: dict[str, FieldScore] = {}
    for path in FIELD_PATHS:
        exact = norm_exact = cer_total = 0.0
        truth_present = pred_present = misses = hallucinations = 0

        for pair, (npred, ntruth) in zip(pairs, normalized):
            raw_p, raw_t = _get(pair.pred, path), _get(pair.truth, path)
            exact += 1.0 if raw_p == raw_t else 0.0
            cer_total += _cer(raw_p, raw_t)

            norm_p, norm_t = _get(npred, path), _get(ntruth, path)
            norm_exact += 1.0 if norm_p == norm_t else 0.0

            if raw_t is not None:
                truth_present += 1
                if raw_p is None:
                    misses += 1
            elif raw_p is not None:
                hallucinations += 1
            if raw_p is not None:
                pred_present += 1

        n = len(pairs) or 1
        scores[path] = FieldScore(
            path=path,
            n=len(pairs),
            exact=exact / n,
            normalized_exact=norm_exact / n,
            cer=cer_total / n,
            truth_present=truth_present,
            pred_present=pred_present,
            misses=misses,
            hallucinations=hallucinations,
        )
    return scores


# --------------------------------------------------------------------------
# line-item level
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LineItemScore:
    n_true: int
    n_pred: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    insertions: int          # predicted rows with no ground-truth counterpart
    deletions: int           # ground-truth rows the model never produced
    attribute_accuracy: dict[str, float] = field(default_factory=dict)
    attribute_support: dict[str, int] = field(default_factory=dict)


def _similarity(a: LineItem, b: LineItem) -> float:
    """Description similarity in [0, 1]. Descriptions only, per the brief."""
    left = normalize_text(a.description) or ""
    right = normalize_text(b.description) or ""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return fuzz.ratio(left, right) / 100.0


def match_line_items(
    pred: Sequence[LineItem],
    truth: Sequence[LineItem],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Hungarian assignment on description similarity.

    Returns accepted ``(pred_index, truth_index, similarity)`` triples. Pairings
    below *threshold* are dropped: the assignment is always complete, so without
    a floor two unrelated rows would be recorded as a match.
    """
    if not pred or not truth:
        return []

    cost = np.zeros((len(pred), len(truth)), dtype=float)
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            cost[i, j] = 1.0 - _similarity(p, t)

    rows, cols = linear_sum_assignment(cost)
    accepted = []
    for i, j in zip(rows, cols):
        similarity = 1.0 - cost[i, j]
        if similarity >= threshold:
            accepted.append((int(i), int(j), float(similarity)))
    return accepted


def score_line_items(
    pairs: Sequence[DocumentPair],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> LineItemScore:
    n_true = n_pred = n_matched = 0
    attribute_hits: dict[str, int] = {f: 0 for f in LINE_ITEM_FIELDS}
    attribute_support: dict[str, int] = {f: 0 for f in LINE_ITEM_FIELDS}

    for pair in pairs:
        pred_doc = normalize_document(pair.pred)
        truth_doc = normalize_document(pair.truth)
        pred_items, truth_items = pred_doc.line_items, truth_doc.line_items

        n_pred += len(pred_items)
        n_true += len(truth_items)

        matches = match_line_items(pred_items, truth_items, threshold)
        n_matched += len(matches)

        for i, j, _ in matches:
            for attribute in LINE_ITEM_FIELDS:
                truth_value = getattr(truth_items[j], attribute)
                if truth_value is None:
                    continue
                attribute_support[attribute] += 1
                if getattr(pred_items[i], attribute) == truth_value:
                    attribute_hits[attribute] += 1

    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return LineItemScore(
        n_true=n_true,
        n_pred=n_pred,
        n_matched=n_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        insertions=n_pred - n_matched,
        deletions=n_true - n_matched,
        attribute_accuracy={
            a: (attribute_hits[a] / attribute_support[a] if attribute_support[a] else 0.0)
            for a in LINE_ITEM_FIELDS
        },
        attribute_support=attribute_support,
    )


# --------------------------------------------------------------------------
# document level
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentScore:
    n: int
    strict_exact_match: float
    ted_accuracy: float
    ted_accuracy_raw: float


def _payload(doc: Document | RawDocument) -> dict:
    return doc.model_dump(mode="json")


def score_documents(pairs: Sequence[DocumentPair]) -> DocumentScore:
    strict = 0.0
    ted_norm = 0.0
    ted_raw = 0.0

    for pair in pairs:
        npred, ntruth = normalize_document(pair.pred), normalize_document(pair.truth)
        if npred == ntruth:
            strict += 1.0
        ted_norm += ted_accuracy(
            json_to_tree(_payload(npred)), json_to_tree(_payload(ntruth))
        )
        ted_raw += ted_accuracy(
            json_to_tree(_payload(pair.pred)), json_to_tree(_payload(pair.truth))
        )

    n = len(pairs) or 1
    return DocumentScore(
        n=len(pairs),
        strict_exact_match=strict / n,
        ted_accuracy=ted_norm / n,
        ted_accuracy_raw=ted_raw / n,
    )


# --------------------------------------------------------------------------
# business level - the differentiator
# --------------------------------------------------------------------------

def default_net_payable(doc: Document) -> Decimal | None:
    """Payable amount implied by an extraction.

    Line items win when present, because that is what an adjudicator actually
    sums; otherwise the stated net falls back in. This is a PLACEHOLDER for the
    phase 6 rules engine, which replaces it - the business metric exists now so
    the baselines can be scored on rupees before any model is trained.
    """
    if doc.line_items:
        amounts = [
            item.amount
            for item in doc.line_items
            if item.amount is not None and item.is_payable is not False
        ]
        if amounts:
            return sum(amounts, Decimal(0))
    return doc.totals.net_amount


def deduction_total(doc: Document) -> Decimal:
    """Total value of line items explicitly marked non-payable."""
    return sum(
        (
            item.amount
            for item in doc.line_items
            if item.amount is not None and item.is_payable is False
        ),
        Decimal(0),
    )


@dataclass(frozen=True)
class BusinessScore:
    n: int
    n_scored: int                     # pairs where truth had a payable at all
    median_error: Decimal
    p95_error: Decimal
    mean_error: Decimal
    pct_error_over_threshold: float
    pct_deduction_total_exact: float
    rupees_per_1000_claims: Decimal
    threshold: Decimal = MATERIAL_ERROR_RUPEES


def score_business(
    pairs: Sequence[DocumentPair],
    payable_fn: Callable[[Document], Decimal | None] = default_net_payable,
    threshold: Decimal = MATERIAL_ERROR_RUPEES,
) -> BusinessScore:
    """Rupee-denominated error in the number the insurer actually pays."""
    errors: list[Decimal] = []
    deduction_exact = 0
    scored = 0

    for pair in pairs:
        npred, ntruth = normalize_document(pair.pred), normalize_document(pair.truth)
        true_payable = payable_fn(ntruth)
        if true_payable is None:
            continue

        scored += 1
        pred_payable = payable_fn(npred)
        # A missing prediction is not a free pass: the whole amount is the error.
        errors.append(
            abs(true_payable) if pred_payable is None
            else abs(pred_payable - true_payable)
        )
        if deduction_total(npred) == deduction_total(ntruth):
            deduction_exact += 1

    denominator = scored or 1
    over = sum(1 for e in errors if e > threshold)
    mean_error = (sum(errors, Decimal(0)) / denominator) if errors else Decimal(0)

    return BusinessScore(
        n=len(pairs),
        n_scored=scored,
        median_error=median(errors),
        p95_error=percentile(errors, 0.95),
        mean_error=mean_error,
        pct_error_over_threshold=over / denominator,
        pct_deduction_total_exact=deduction_exact / denominator,
        rupees_per_1000_claims=mean_error * 1000,
        threshold=threshold,
    )


# --------------------------------------------------------------------------
# abstention and coverage
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    field_coverage: float       # share of fields the system answers
    field_accuracy: float       # normalized exact match among answered fields
    document_rate: float        # share of docs where EVERY field clears the bar


Confidences = Mapping[str, Mapping[str, float]]


def coverage_curve(
    pairs: Sequence[DocumentPair],
    confidences: Confidences,
    thresholds: Iterable[float] | None = None,
) -> list[CoveragePoint]:
    """Sweep an abstention threshold and trace coverage against accuracy.

    *confidences* maps ``doc_id -> field_path -> score in [0, 1]``. A field with
    no recorded confidence is treated as confidence 0, i.e. always abstained on,
    so a missing score can never inflate coverage.
    """
    if thresholds is None:
        thresholds = [i / 20 for i in range(21)]

    normalized = {
        p.doc_id: (normalize_document(p.pred), normalize_document(p.truth))
        for p in pairs
    }

    points: list[CoveragePoint] = []
    for threshold in thresholds:
        answered = correct = 0
        total = 0
        full_documents = 0

        for pair in pairs:
            npred, ntruth = normalized[pair.doc_id]
            per_field = confidences.get(pair.doc_id, {})
            document_complete = True

            for path in FIELD_PATHS:
                total += 1
                if per_field.get(path, 0.0) >= threshold:
                    answered += 1
                    if _get(npred, path) == _get(ntruth, path):
                        correct += 1
                else:
                    document_complete = False

            if document_complete:
                full_documents += 1

        points.append(
            CoveragePoint(
                threshold=threshold,
                field_coverage=answered / total if total else 0.0,
                field_accuracy=correct / answered if answered else 1.0,
                document_rate=full_documents / len(pairs) if pairs else 0.0,
            )
        )
    return points


def auto_processing_rate(
    curve: Sequence[CoveragePoint], target_accuracy: float = 0.95
) -> CoveragePoint | None:
    """Highest-coverage point that still clears *target_accuracy*.

    None when no threshold reaches the target - which is itself a reportable
    result, and must not be silently rendered as 0%.
    """
    eligible = [p for p in curve if p.field_accuracy >= target_accuracy and p.field_coverage > 0]
    return max(eligible, key=lambda p: p.field_coverage) if eligible else None


def deployment_sentence(
    curve: Sequence[CoveragePoint], target_accuracy: float = 0.95
) -> str:
    """The one sentence worth more in an interview than any F1."""
    point = auto_processing_rate(curve, target_accuracy)
    if point is None:
        return (
            f"No abstention threshold reaches {target_accuracy:.0%} field accuracy; "
            "every document requires human review."
        )
    return (
        f"At a confidence threshold of {point.threshold:.2f}, giving "
        f"{point.field_accuracy:.1%} field accuracy, the system answers "
        f"{point.field_coverage:.1%} of fields and fully auto-processes "
        f"{point.document_rate:.1%} of documents, routing the rest to human review."
    )

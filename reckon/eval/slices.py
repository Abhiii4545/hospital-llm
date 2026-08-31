"""Slice breakdowns.

Section 5: every metric broken down by template, page count, scan quality,
bilingual vs monolingual, and synthetic vs real - and *automated*, so it
regenerates on every eval run with no manual work.

Slice keys are therefore DISCOVERED from the pair metadata rather than
hard-coded. Adding a new metadata key to the corpus adds a new slice table to the
report automatically; nobody has to remember to update this file.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from reckon.eval.metrics import (
    DocumentPair,
    EvalCache,
    score_business,
    score_documents,
    score_fields,
    score_line_items,
)

__all__ = [
    "SliceSummary",
    "discover_slice_keys",
    "group_by",
    "summarise",
    "slice_report",
]

#: Preferred display order. Keys not listed still appear, sorted, after these.
PREFERRED_ORDER = ("source", "template", "pages", "scan_quality", "language")

#: A slice smaller than this is reported but flagged: the numbers are noise.
MIN_RELIABLE_SLICE = 5


@dataclass(frozen=True)
class SliceSummary:
    key: str
    value: str
    n: int
    field_exact_macro: float
    field_normalized_macro: float
    line_item_f1: float
    ted_accuracy: float
    strict_exact: float
    median_business_error: Decimal

    @property
    def underpowered(self) -> bool:
        return self.n < MIN_RELIABLE_SLICE


def discover_slice_keys(pairs: Sequence[DocumentPair]) -> list[str]:
    """Every metadata key present on any pair, in a stable display order."""
    found: set[str] = set()
    for pair in pairs:
        found.update(pair.meta.keys())
    ordered = [k for k in PREFERRED_ORDER if k in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def group_by(pairs: Sequence[DocumentPair], key: str) -> dict[str, list[DocumentPair]]:
    groups: dict[str, list[DocumentPair]] = defaultdict(list)
    for pair in pairs:
        if key in pair.meta:
            groups[str(pair.meta[key])].append(pair)
    return dict(groups)


def summarise(
    pairs: Sequence[DocumentPair], key: str, value: str,
    cache: EvalCache | None = None,
) -> SliceSummary:
    """Headline numbers for one slice.

    The macro averages here are a summary only. The per-field breakdown that the
    brief insists on lives in the main report section; a slice table with 27
    columns would be unreadable and would get skipped, which is worse.
    """
    cache = cache or EvalCache()
    fields = score_fields(pairs, cache)
    n_fields = len(fields) or 1
    line_items = score_line_items(pairs, cache=cache)
    documents = score_documents(pairs, cache)
    business = score_business(pairs, cache=cache)

    return SliceSummary(
        key=key,
        value=value,
        n=len(pairs),
        field_exact_macro=sum(f.exact for f in fields.values()) / n_fields,
        field_normalized_macro=sum(f.normalized_exact for f in fields.values()) / n_fields,
        line_item_f1=line_items.f1,
        ted_accuracy=documents.ted_accuracy,
        strict_exact=documents.strict_exact_match,
        median_business_error=business.median_error,
    )


def slice_report(
    pairs: Sequence[DocumentPair], cache: EvalCache | None = None
) -> dict[str, list[SliceSummary]]:
    """All slice breakdowns, keyed by metadata field.

    One cache spans every slice: the same documents appear in every breakdown, so
    without it each document is normalized and tree-diffed once per slice key.
    """
    cache = cache or EvalCache()
    report: dict[str, list[SliceSummary]] = {}
    for key in discover_slice_keys(pairs):
        summaries = [
            summarise(group, key, value, cache)
            for value, group in sorted(group_by(pairs, key).items())
        ]
        if summaries:
            report[key] = summaries
    return report

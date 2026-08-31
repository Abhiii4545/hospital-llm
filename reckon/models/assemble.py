"""Page fragments -> one document, with a reconciliation report.

Plain Python, deliberately. Section 2.1 of the brief: the assembly layer is not a
model. Concatenating pages, spotting a row reprinted across a page break, and
checking that the line items add up to the printed total are all deterministic
operations with a correct answer, and a model would only make them
non-auditable.

Three jobs:

1. **Concatenate** per-page Head A and Head B outputs in page order.
2. **Deduplicate** rows repeated across a page break. The corpus prints these at
   a 12% rate because real multi-page bills do it, and the model is deliberately
   NOT taught to drop them - that is this layer's job.
3. **Reconcile** the summed line items against the printed totals, and report
   every disagreement rather than silently preferring one source.

The reconciliation report is a deliverable in its own right. An adjudicator needs
to know that a bill does not add up; quietly patching the number would hide
exactly the thing a human should look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from reckon.normalize import (
    normalize_amount,
    normalize_document,
    normalize_quantity,
    normalize_text,
)
from reckon.schema import (
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = [
    "PageFragment",
    "ReconciliationReport",
    "AssembledDocument",
    "assemble",
    "TOLERANCE",
]

#: Rupee tolerance below which a totals mismatch is treated as rounding rather
#: than a real disagreement. Paise-level drift is normal on a printed bill; a
#: rupee or more is not.
TOLERANCE = Decimal("1.00")


@dataclass
class PageFragment:
    """What the two heads produced for a single page."""

    page_index: int
    hospital: RawHospital | None = None
    patient: RawPatient | None = None
    insurance: RawInsurance | None = None
    totals: RawTotals | None = None
    line_items: list[RawLineItem] = field(default_factory=list)


@dataclass
class ReconciliationReport:
    """Everything that did not line up. Empty is the happy path."""

    n_pages: int = 0
    n_rows_raw: int = 0
    n_rows_after_dedup: int = 0
    duplicates_removed: list[str] = field(default_factory=list)

    line_item_sum: Decimal | None = None
    stated_gross: Decimal | None = None
    gross_delta: Decimal | None = None

    stated_net: Decimal | None = None
    computed_net: Decimal | None = None
    net_delta: Decimal | None = None

    row_arithmetic_failures: list[str] = field(default_factory=list)
    conflicting_header_fields: list[str] = field(default_factory=list)
    missing_blocks: list[str] = field(default_factory=list)

    @property
    def balanced(self) -> bool:
        """True when the ARITHMETIC reconciles.

        Deliberately separate from completeness: a bill whose numbers add up is
        balanced even if some header block never arrived. Conflating the two
        made a perfectly reconciled document look broken because no page carried
        an insurance block.
        """
        return not self.arithmetic_flags

    @property
    def complete(self) -> bool:
        """True when every block was produced by some page and pages agree."""
        return not self.missing_blocks and not self.conflicting_header_fields

    @property
    def arithmetic_flags(self) -> list[str]:
        out: list[str] = []
        if self.gross_delta is not None and abs(self.gross_delta) > TOLERANCE:
            out.append(
                f"line items sum to {self.line_item_sum} but gross is "
                f"{self.stated_gross} (delta {self.gross_delta})"
            )
        if self.net_delta is not None and abs(self.net_delta) > TOLERANCE:
            out.append(
                f"stated net {self.stated_net} does not equal computed "
                f"{self.computed_net} (delta {self.net_delta})"
            )
        if self.row_arithmetic_failures:
            out.append(
                f"{len(self.row_arithmetic_failures)} row(s) where "
                "quantity x rate != amount"
            )
        return out

    @property
    def flags(self) -> list[str]:
        """Everything a human should look at: arithmetic and completeness."""
        out = list(self.arithmetic_flags)
        if self.conflicting_header_fields:
            out.append(
                "conflicting values across pages for: "
                + ", ".join(self.conflicting_header_fields)
            )
        if self.missing_blocks:
            out.append("no page carried: " + ", ".join(self.missing_blocks))
        return out


@dataclass
class AssembledDocument:
    document: RawDocument
    report: ReconciliationReport


def _row_key(item: RawLineItem) -> tuple:
    """Identity of a printed row, for duplicate detection.

    Normalized, because the same row reprinted on the next page can differ in
    whitespace or currency decoration after OCR. Serial number is deliberately
    EXCLUDED: a continuation page often renumbers, and requiring it to match
    would miss most real duplicates.
    """
    return (
        normalize_text(item.description),
        normalize_amount(item.amount),
        normalize_quantity(item.quantity),
        normalize_amount(item.unit_rate),
    )


def _dedupe_across_page_breaks(
    fragments: Sequence[PageFragment], report: ReconciliationReport
) -> list[RawLineItem]:
    """Concatenate pages, dropping a row reprinted at a page boundary.

    Only the boundary is checked - the last rows of page N against the first
    rows of page N+1. A global uniqueness filter would be wrong: a bill can
    legitimately bill the same item twice on the same page (two doses of the
    same drug on the same day), and collapsing those would lose real money.
    """
    merged: list[RawLineItem] = []
    for position, fragment in enumerate(fragments):
        rows = list(fragment.line_items)
        if position and merged and rows:
            # Real reprints repeat a short run, almost always exactly one row.
            window = min(3, len(merged), len(rows))
            overlap = 0
            for size in range(window, 0, -1):
                tail = [_row_key(i) for i in merged[-size:]]
                head = [_row_key(i) for i in rows[:size]]
                if tail == head:
                    overlap = size
                    break
            if overlap:
                for dropped in rows[:overlap]:
                    report.duplicates_removed.append(
                        f"page {fragment.page_index}: {dropped.description}"
                    )
                rows = rows[overlap:]
        merged.extend(rows)
    return merged


def _pick_header(
    fragments: Sequence[PageFragment], attribute: str, report: ReconciliationReport
):
    """First non-empty block wins; disagreement between pages is REPORTED.

    Silently taking page 1 would hide the case where two pages of the same
    document carry different patient names - which means the pages were mis-
    grouped, and that is a serious error worth surfacing.
    """
    seen = [getattr(f, attribute) for f in fragments if getattr(f, attribute) is not None]
    if not seen:
        report.missing_blocks.append(attribute)
        return None

    chosen = seen[0]
    for other in seen[1:]:
        for name in type(chosen).model_fields:
            left, right = getattr(chosen, name), getattr(other, name)
            if left and right and normalize_text(left) != normalize_text(right):
                path = f"{attribute}.{name}"
                if path not in report.conflicting_header_fields:
                    report.conflicting_header_fields.append(path)
    return chosen


def assemble(fragments: Sequence[PageFragment]) -> AssembledDocument:
    """Merge per-page fragments into one document plus a reconciliation report."""
    ordered = sorted(fragments, key=lambda f: f.page_index)
    report = ReconciliationReport(n_pages=len(ordered))
    report.n_rows_raw = sum(len(f.line_items) for f in ordered)

    items = _dedupe_across_page_breaks(ordered, report)
    report.n_rows_after_dedup = len(items)

    document = RawDocument(
        hospital=_pick_header(ordered, "hospital", report) or RawHospital(),
        patient=_pick_header(ordered, "patient", report) or RawPatient(),
        insurance=_pick_header(ordered, "insurance", report) or RawInsurance(),
        totals=_pick_header(ordered, "totals", report) or RawTotals(),
        line_items=items,
    )

    _reconcile(document, report)
    return AssembledDocument(document=document, report=report)


def _reconcile(document: RawDocument, report: ReconciliationReport) -> None:
    """Check the arithmetic. Report disagreements; never patch the numbers."""
    typed = normalize_document(document)

    amounts = [i.amount for i in typed.line_items if i.amount is not None]
    if amounts:
        report.line_item_sum = sum(amounts, Decimal(0))

    for position, item in enumerate(typed.line_items):
        if item.quantity is None or item.unit_rate is None or item.amount is None:
            continue
        if item.quantity * item.unit_rate != item.amount:
            report.row_arithmetic_failures.append(
                f"row {position + 1}: {item.quantity} x {item.unit_rate} "
                f"!= {item.amount}"
            )

    totals = typed.totals
    report.stated_gross = totals.gross_amount
    if report.line_item_sum is not None and totals.gross_amount is not None:
        report.gross_delta = report.line_item_sum - totals.gross_amount

    report.stated_net = totals.net_amount
    if totals.gross_amount is not None:
        computed = (
            totals.gross_amount
            - (totals.discount or Decimal(0))
            + (totals.cgst or Decimal(0))
            + (totals.sgst or Decimal(0))
        )
        report.computed_net = computed
        if totals.net_amount is not None:
            report.net_delta = totals.net_amount - computed

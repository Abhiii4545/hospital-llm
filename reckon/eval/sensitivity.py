"""How extraction error propagates into the amount actually paid.

Section 6 of the brief. Room-rent capping applies a PROPORTIONATE deduction
across associated charges, so an error in `room_rent` does not cost the insurer
that error - it costs a fraction of every proportionately-reduced line on the
bill. A 1% misread of a room rate can move the payable by far more than 1% of
the room rent.

This is worth building because it changes what the project optimises. Per-field
accuracy treats `room_rent` and `hospital.city` as equally important. This says
they are not, and by how much, in rupees.

It also has a discontinuity that a linear error analysis would miss entirely:
below the cap, a room-rate error costs only itself; the moment the misread rate
crosses the cap, the proportionate factor engages and the cost jumps.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Sequence

from reckon.adjudicate.engine import (
    Adjudication,
    PolicyTerms,
    RuleSet,
    adjudicate,
    load_policy,
    load_rules,
)
from reckon.schema import Category, Document

__all__ = [
    "Perturbation",
    "SensitivityPoint",
    "sweep_field",
    "amplification",
    "DEFAULT_PERTURBATIONS",
    "perturb_room_rate",
    "perturb_amounts_in_category",
]

#: Relative errors to sweep. Small ones matter most: a model that is 1% off on
#: every room rate is a plausible model, and this says what that costs.
DEFAULT_PERTURBATIONS: tuple[Decimal, ...] = tuple(
    Decimal(v) for v in
    ("-0.10", "-0.05", "-0.02", "-0.01", "0", "0.01", "0.02", "0.05", "0.10")
)


@dataclass(frozen=True)
class SensitivityPoint:
    relative_error: Decimal
    payable: Decimal
    payable_delta: Decimal          # vs the unperturbed payable
    deducted: Decimal

    @property
    def absolute_error(self) -> Decimal:
        return abs(self.payable_delta)


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[Document, Decimal], Document]


def perturb_room_rate(document: Document, relative: Decimal) -> Document:
    """Scale every room-rent unit rate (and its amount) by ``1 + relative``."""
    out = deepcopy(document)
    factor = Decimal(1) + relative
    for item in out.line_items:
        if item.category is not Category.ROOM_RENT:
            continue
        if item.unit_rate is not None:
            item.unit_rate = item.unit_rate * factor
        if item.amount is not None:
            item.amount = item.amount * factor
    return out


def perturb_amounts_in_category(category: Category) -> Callable[[Document, Decimal], Document]:
    def apply(document: Document, relative: Decimal) -> Document:
        out = deepcopy(document)
        factor = Decimal(1) + relative
        for item in out.line_items:
            if item.category is category and item.amount is not None:
                item.amount = item.amount * factor
        return out
    return apply


def perturb_gross(document: Document, relative: Decimal) -> Document:
    out = deepcopy(document)
    if out.totals.gross_amount is not None:
        out.totals.gross_amount = out.totals.gross_amount * (Decimal(1) + relative)
    return out


def sweep_field(
    document: Document,
    perturbation: Perturbation,
    relatives: Sequence[Decimal] = DEFAULT_PERTURBATIONS,
    rules: RuleSet | None = None,
    policy: PolicyTerms | None = None,
) -> list[SensitivityPoint]:
    """Adjudicate the document at each perturbation level."""
    rules = rules or load_rules()
    policy = policy or load_policy()

    baseline: Adjudication = adjudicate(document, rules, policy)
    points: list[SensitivityPoint] = []
    for relative in relatives:
        perturbed = perturbation.apply(document, relative)
        result = adjudicate(perturbed, rules, policy)
        points.append(SensitivityPoint(
            relative_error=relative,
            payable=result.payable,
            payable_delta=result.payable - baseline.payable,
            deducted=result.total_deducted,
        ))
    return points


def amplification(points: Sequence[SensitivityPoint], relative: Decimal) -> Decimal | None:
    """Rupees of payable error per rupee of input error, at one perturbation.

    Greater than 1 means the rules engine AMPLIFIES the extraction error. That is
    the headline number: it is what justifies spending accuracy budget on
    room_rent rather than on hospital.city.
    """
    for point in points:
        if point.relative_error == relative:
            if relative == 0:
                return Decimal(0)
            base = next((p.payable for p in points if p.relative_error == 0), None)
            if base is None or base == 0:
                return None
            return abs(point.payable_delta) / (abs(relative) * base)
    return None


DEFAULT_PERTURBATION_SET: tuple[Perturbation, ...] = (
    Perturbation("room_rent unit rate", perturb_room_rate),
    Perturbation("pharmacy amounts", perturb_amounts_in_category(Category.PHARMACY)),
    Perturbation("surgery amounts", perturb_amounts_in_category(Category.SURGERY)),
    Perturbation("stated gross", perturb_gross),
)


def report(
    document: Document,
    perturbations: Sequence[Perturbation] = DEFAULT_PERTURBATION_SET,
    relatives: Sequence[Decimal] = DEFAULT_PERTURBATIONS,
) -> str:
    """Markdown table of payable error against input error."""
    rules, policy = load_rules(), load_policy()
    lines = ["| perturbed field | " + " | ".join(
        f"{r:+.0%}" for r in relatives) + " |"]
    lines.append("|" + "---|" * (len(relatives) + 1))
    for perturbation in perturbations:
        points = sweep_field(document, perturbation, relatives, rules, policy)
        cells = " | ".join(f"{p.payable_delta:+,.0f}" for p in points)
        lines.append(f"| {perturbation.name} | {cells} |")
    lines.append("")
    lines.append("_Cells are the change in net payable, in rupees._")
    return "\n".join(lines)

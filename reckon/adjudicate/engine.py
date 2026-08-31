"""Deterministic adjudication engine.

Section 6 of the brief: this stays a rules engine, and that is the correct
design. An insurer must be able to point at the clause that produced a
deduction; a model's opinion is not auditable and would not survive an ombudsman
complaint.

Every deduction carries a rule id, a clause citation and a human-readable
reason. **An adjudication with no traceable reason is a bug**, and a test
asserts that no deduction can be produced without one.

Order of operations is fixed and comes from the policy file, because it changes
the answer: applying co-pay before the deductible pays out a different amount
than applying it after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

import yaml

from reckon.schema import Category, Document, LineItem

__all__ = [
    "Deduction",
    "Adjudication",
    "PolicyTerms",
    "RuleSet",
    "load_rules",
    "load_policy",
    "adjudicate",
    "RULES_DIR",
]

RULES_DIR = Path(__file__).parent / "rules"


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Deduction:
    """One deduction, fully traceable back to the clause that caused it."""

    rule_id: str
    clause: str
    reason: str
    amount: Decimal
    line_index: int | None = None
    line_description: str | None = None

    def __post_init__(self) -> None:
        # Enforced here rather than in review: a deduction that cannot be
        # explained to a policyholder must not be constructible at all.
        if not self.rule_id or not self.clause or not self.reason:
            raise ValueError("a deduction requires rule_id, clause and reason")


@dataclass
class Adjudication:
    gross: Decimal
    deductions: list[Deduction] = field(default_factory=list)
    payable: Decimal = Decimal(0)
    notes: list[str] = field(default_factory=list)

    @property
    def total_deducted(self) -> Decimal:
        return _money(sum((d.amount for d in self.deductions), Decimal(0)))

    def by_rule(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for deduction in self.deductions:
            out[deduction.rule_id] = _money(
                out.get(deduction.rule_id, Decimal(0)) + deduction.amount
            )
        return out

    def explain(self) -> str:
        """Plain-text audit trail."""
        lines = [f"Gross considered: {self.gross}"]
        for deduction in self.deductions:
            where = (
                f" [row {deduction.line_index + 1}: {deduction.line_description}]"
                if deduction.line_index is not None else ""
            )
            lines.append(
                f"  -{deduction.amount}  {deduction.rule_id} "
                f"({deduction.clause}){where}\n      {deduction.reason}"
            )
        lines.append(f"Net payable: {self.payable}")
        return "\n".join(lines)


@dataclass
class RuleSet:
    version: int
    rules: list[dict[str, Any]]

    def match(self, description: str | None, category: Category | None) -> dict | None:
        """Most SPECIFIC matching rule, by longest matching needle.

        First-match-wins made attribution depend on rule order in the YAML file:
        "Food & Beverages (attendant)" matched the attendant rule purely because
        it appears earlier. The deduction was right either way, but the cited
        clause was not, and the clause is the part a policyholder gets shown.
        Longest needle wins, so the more specific rule is credited.
        """
        text = (description or "").casefold()
        best: tuple[int, dict] | None = None
        for rule in self.rules:
            for needle in rule.get("match", ()):
                folded = needle.casefold()
                if folded in text and (best is None or len(folded) > best[0]):
                    best = (len(folded), rule)
        if best is not None:
            return best[1]
        if text:
            return None
        for rule in self.rules:
            hint = rule.get("category_hint")
            if hint and category is not None and category.value == hint:
                return rule
        return None


@dataclass
class PolicyTerms:
    sum_insured: Decimal
    room_rent_cap_percent: Decimal
    icu_cap_percent: Decimal
    proportionate_categories: frozenset[str]
    co_pay_percent: Decimal
    deductible: Decimal
    sub_limits: list[dict[str, Any]]
    order: tuple[str, ...]

    @property
    def room_cap_per_day(self) -> Decimal:
        return _money(self.sum_insured * self.room_rent_cap_percent / Decimal(100))

    @property
    def icu_cap_per_day(self) -> Decimal:
        return _money(self.sum_insured * self.icu_cap_percent / Decimal(100))


def load_rules(path: Path | str | None = None) -> RuleSet:
    data = yaml.safe_load(
        Path(path or RULES_DIR / "irdai_list_i.yaml").read_text(encoding="utf-8")
    )
    return RuleSet(version=data["version"], rules=data["rules"])


def load_policy(path: Path | str | None = None, **overrides: Any) -> PolicyTerms:
    data = yaml.safe_load(
        Path(path or RULES_DIR / "policy_default.yaml").read_text(encoding="utf-8")
    )["policy"]
    data.update(overrides)
    room = data["room_rent"]
    return PolicyTerms(
        sum_insured=Decimal(str(data["sum_insured"])),
        room_rent_cap_percent=Decimal(str(room["cap_percent_of_sum_insured_per_day"])),
        icu_cap_percent=Decimal(str(room["icu_cap_percent_of_sum_insured_per_day"])),
        proportionate_categories=frozenset(room["proportionate_categories"]),
        co_pay_percent=Decimal(str(data["co_pay_percent"])),
        deductible=Decimal(str(data["deductible"])),
        sub_limits=list(data.get("sub_limits", [])),
        order=tuple(data["order"]),
    )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _stage_non_payable(
    items: Sequence[LineItem], rules: RuleSet, live: dict[int, Decimal]
) -> list[Deduction]:
    out: list[Deduction] = []
    for index, item in enumerate(items):
        if live.get(index, Decimal(0)) <= 0:
            continue
        rule = rules.match(item.description, item.category)
        if rule is None:
            continue
        amount = live[index]
        live[index] = Decimal(0)
        out.append(Deduction(
            rule_id=rule["id"], clause=rule["clause"], reason=rule["reason"],
            amount=_money(amount), line_index=index,
            line_description=item.description,
        ))
    return out


def _stage_sub_limits(
    items: Sequence[LineItem], policy: PolicyTerms, live: dict[int, Decimal]
) -> list[Deduction]:
    out: list[Deduction] = []
    for index, item in enumerate(items):
        amount = live.get(index, Decimal(0))
        if amount <= 0:
            continue
        text = (item.description or "").casefold()
        for limit in policy.sub_limits:
            if str(limit["match"]).casefold() in text:
                cap = Decimal(str(limit["cap"]))
                if amount > cap:
                    excess = amount - cap
                    live[index] = cap
                    out.append(Deduction(
                        rule_id="SUB_LIMIT",
                        clause=str(limit.get("clause", "policy sub-limit")),
                        reason=(
                            f"Charge of {amount} exceeds the policy sub-limit of "
                            f"{cap} for '{limit['match']}'"
                        ),
                        amount=_money(excess), line_index=index,
                        line_description=item.description,
                    ))
                break
    return out


def _stage_room_rent_cap(
    items: Sequence[LineItem], doc: Document, policy: PolicyTerms,
    live: dict[int, Decimal], notes: list[str],
) -> list[Deduction]:
    """Room-rent capping, with the proportionate deduction it triggers.

    This is the stage that compounds an extraction error. If the eligible room
    rate is R and the billed rate is B > R, the policy pays the associated
    categories only in the ratio R/B - so a 1% error in the billed room rate
    moves the final payable by roughly 1% of every proportionately-reduced
    charge, not just of the room rent. eval/sensitivity.py measures it.
    """
    out: list[Deduction] = []
    room_rows = [
        (index, item) for index, item in enumerate(items)
        if item.category is Category.ROOM_RENT
    ]
    if not room_rows:
        return out

    is_icu = (doc.patient.ward_type or "") in {"icu", "iccu", "nicu", "picu"}
    cap = policy.icu_cap_per_day if is_icu else policy.room_cap_per_day

    # Billed daily rate: prefer the stated unit rate, else amount / quantity.
    index, item = room_rows[0]
    if item.unit_rate is not None:
        billed = item.unit_rate
    elif item.amount is not None and item.quantity:
        billed = item.amount / item.quantity
    else:
        notes.append("room rent present but no daily rate could be derived")
        return out

    if billed <= cap:
        return out

    ratio = cap / billed
    notes.append(
        f"room rate {billed}/day exceeds the {cap}/day cap; proportionate "
        f"factor {ratio:.4f} applied to "
        f"{', '.join(sorted(policy.proportionate_categories))}"
    )

    for position, entry in enumerate(items):
        amount = live.get(position, Decimal(0))
        if amount <= 0 or entry.category is None:
            continue
        if entry.category.value not in policy.proportionate_categories:
            continue
        reduced = _money(amount * ratio)
        excess = _money(amount - reduced)
        if excess <= 0:
            continue
        live[position] = reduced
        out.append(Deduction(
            rule_id="ROOM_RENT_CAP",
            clause="Policy schedule - room rent capping with proportionate deduction",
            reason=(
                f"Room rate {billed}/day exceeds the eligible {cap}/day; this "
                f"charge is reduced proportionately by factor {ratio:.4f}"
            ),
            amount=excess, line_index=position,
            line_description=entry.description,
        ))
    return out


def adjudicate(
    document: Document,
    rules: RuleSet | None = None,
    policy: PolicyTerms | None = None,
) -> Adjudication:
    """Apply the rule set and policy terms to an extracted document."""
    rules = rules or load_rules()
    policy = policy or load_policy()
    items = document.line_items

    live: dict[int, Decimal] = {
        index: item.amount for index, item in enumerate(items)
        if item.amount is not None
    }
    gross = _money(sum(live.values(), Decimal(0)))
    result = Adjudication(gross=gross)

    for stage in policy.order:
        if stage == "non_payable":
            result.deductions += _stage_non_payable(items, rules, live)
        elif stage == "sub_limits":
            result.deductions += _stage_sub_limits(items, policy, live)
        elif stage == "room_rent_cap":
            result.deductions += _stage_room_rent_cap(
                items, document, policy, live, result.notes
            )
        elif stage == "deductible":
            remaining = _money(sum(live.values(), Decimal(0)))
            if policy.deductible > 0 and remaining > 0:
                taken = min(policy.deductible, remaining)
                _scale(live, (remaining - taken) / remaining if remaining else Decimal(0))
                result.deductions.append(Deduction(
                    rule_id="DEDUCTIBLE",
                    clause="Policy schedule - deductible",
                    reason=f"Policy deductible of {policy.deductible} applied",
                    amount=_money(taken),
                ))
        elif stage == "co_pay":
            remaining = _money(sum(live.values(), Decimal(0)))
            if policy.co_pay_percent > 0 and remaining > 0:
                taken = _money(remaining * policy.co_pay_percent / Decimal(100))
                _scale(live, (remaining - taken) / remaining if remaining else Decimal(0))
                result.deductions.append(Deduction(
                    rule_id="CO_PAY",
                    clause="Policy schedule - co-payment",
                    reason=(
                        f"Co-payment of {policy.co_pay_percent}% applied to the "
                        f"admissible amount of {remaining}"
                    ),
                    amount=taken,
                ))
        else:  # pragma: no cover - guarded by a test over the policy file
            raise ValueError(f"unknown adjudication stage: {stage}")

    result.payable = _money(sum(live.values(), Decimal(0)))
    return result


def _scale(live: dict[int, Decimal], factor: Decimal) -> None:
    for key in live:
        live[key] = live[key] * factor

"""Adjudication engine, assembly layer and sensitivity analysis."""

from __future__ import annotations

from decimal import Decimal

import pytest

from reckon.adjudicate.engine import (
    RULES_DIR,
    Deduction,
    adjudicate,
    load_policy,
    load_rules,
)
from reckon.eval.sensitivity import (
    Perturbation,
    amplification,
    perturb_room_rate,
    sweep_field,
)
from reckon.models.assemble import (
    PageFragment,
    assemble,
)
from reckon.schema import (
    Category,
    Document,
    LineItem,
    Patient,
    RawHospital,
    RawLineItem,
    RawPatient,
    RawTotals,
    Totals,
)


def item(description, amount, category, *, quantity="1", rate=None) -> LineItem:
    return LineItem(
        description=description,
        amount=Decimal(amount),
        category=category,
        quantity=Decimal(quantity),
        unit_rate=Decimal(rate) if rate is not None else None,
    )


def doc(items, ward="private") -> Document:
    return Document(patient=Patient(ward_type=ward), line_items=items)


# --------------------------------------------------------------------------
# traceability - the core requirement
# --------------------------------------------------------------------------

def test_a_deduction_cannot_exist_without_a_traceable_reason() -> None:
    """"An adjudication with no traceable reason is a bug" - enforced, not reviewed."""
    with pytest.raises(ValueError):
        Deduction(rule_id="", clause="c", reason="r", amount=Decimal(1))
    with pytest.raises(ValueError):
        Deduction(rule_id="X", clause="", reason="r", amount=Decimal(1))
    with pytest.raises(ValueError):
        Deduction(rule_id="X", clause="c", reason="", amount=Decimal(1))


def test_every_deduction_carries_a_rule_id_and_clause() -> None:
    result = adjudicate(doc([
        item("Room Rent - Private", "4000", Category.ROOM_RENT, rate="4000"),
        item("Attendant Charges", "500", Category.NON_MEDICAL),
        item("Telephone Charges", "120", Category.NON_MEDICAL),
        item("Registration Fee", "250", Category.ADMINISTRATIVE),
    ]))
    assert result.deductions
    for deduction in result.deductions:
        assert deduction.rule_id and deduction.clause and deduction.reason


def test_explain_produces_an_audit_trail() -> None:
    result = adjudicate(doc([
        item("Attendant Charges", "500", Category.NON_MEDICAL),
        item("Consultant Visit", "900", Category.PROFESSIONAL_FEES),
    ]))
    text = result.explain()
    assert "LIST_I_ATTENDANT" in text
    assert "IRDAI List I" in text
    assert "Net payable" in text


# --------------------------------------------------------------------------
# IRDAI List I
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("description", "rule_id"),
    [
        ("Toiletries Kit", "LIST_I_TOILETRIES"),
        ("Attendant Charges", "LIST_I_ATTENDANT"),
        ("Food & Beverages (attendant)", "LIST_I_FOOD_BEVERAGE"),
        ("Telephone Charges", "LIST_I_COMMUNICATION"),
        ("Laundry Charges", "LIST_I_LAUNDRY"),
        ("Registration Fee", "LIST_I_ADMIN_RECORDS"),
        ("Medical Records Charge", "LIST_I_ADMIN_RECORDS"),
    ],
)
def test_list_one_items_are_fully_deducted(description: str, rule_id: str) -> None:
    result = adjudicate(doc([item(description, "1000", Category.NON_MEDICAL)]),
                        policy=load_policy(co_pay_percent=0))
    assert result.payable == Decimal("0.00")
    assert result.by_rule()[rule_id] == Decimal("1000.00")


@pytest.mark.parametrize(
    "description",
    ["Paracetamol 500mg", "Complete Blood Count", "Surgeon Fee", "Nebuliser Kit",
     "Oxygen Charges", "MRI Brain Plain"],
)
def test_clinical_items_are_not_deducted_as_non_payable(description: str) -> None:
    """A rule that eats real treatment would be worse than no rule."""
    result = adjudicate(doc([item(description, "1000", Category.PHARMACY)]))
    assert not any(d.rule_id.startswith("LIST_I") for d in result.deductions)


def test_rules_file_is_data_and_every_rule_is_complete() -> None:
    rules = load_rules()
    assert rules.version >= 1
    ids = [r["id"] for r in rules.rules]
    assert len(set(ids)) == len(ids)
    for rule in rules.rules:
        assert rule["clause"], rule["id"]
        assert rule["reason"], rule["id"]
        assert rule["match"], rule["id"]


# --------------------------------------------------------------------------
# room rent capping and proportionate deduction
# --------------------------------------------------------------------------

def test_room_rate_within_cap_triggers_no_proportionate_deduction() -> None:
    policy = load_policy(sum_insured=500000, co_pay_percent=0)   # cap 5000/day
    result = adjudicate(
        doc([item("Room Rent - Private", "4000", Category.ROOM_RENT, rate="4000"),
             item("Nursing Charges", "2000", Category.NURSING)]),
        policy=policy,
    )
    assert not any(d.rule_id == "ROOM_RENT_CAP" for d in result.deductions)
    assert result.payable == Decimal("6000.00")


def test_exceeding_the_room_cap_reduces_associated_charges_proportionately() -> None:
    """The compounding case: the deduction is not limited to the room rent."""
    policy = load_policy(sum_insured=500000, co_pay_percent=0)   # cap 5000/day
    result = adjudicate(
        doc([item("Room Rent - Deluxe", "10000", Category.ROOM_RENT, rate="10000"),
             item("Nursing Charges", "2000", Category.NURSING),
             item("Surgeon Fee", "50000", Category.SURGERY),
             item("Paracetamol 500mg", "500", Category.PHARMACY)]),
        policy=policy,
    )
    capped = [d for d in result.deductions if d.rule_id == "ROOM_RENT_CAP"]
    assert capped, "room cap did not fire"

    touched = {d.line_description for d in capped}
    assert "Nursing Charges" in touched
    assert "Surgeon Fee" in touched
    # Pharmacy does not scale with room class, so it must be untouched.
    assert "Paracetamol 500mg" not in touched

    # factor = 5000/10000 = 0.5 applied to room, nursing and surgery; pharmacy
    # untouched:  5000 + 1000 + 25000 + 500 = 31500
    assert result.payable == Decimal("31500.00")


def test_icu_uses_its_own_more_generous_cap() -> None:
    policy = load_policy(sum_insured=500000, co_pay_percent=0)
    items = [item("Room Rent - ICU", "9000", Category.ROOM_RENT, rate="9000")]
    in_icu = adjudicate(doc(items, ward="icu"), policy=policy)
    in_private = adjudicate(doc(items, ward="private"), policy=policy)
    assert in_icu.payable > in_private.payable


# --------------------------------------------------------------------------
# sub-limits, deductible, co-pay and their order
# --------------------------------------------------------------------------

def test_sub_limit_caps_a_procedure() -> None:
    policy = load_policy(co_pay_percent=0)
    result = adjudicate(
        doc([item("Cataract Surgery (Phaco)", "60000", Category.SURGERY)]),
        policy=policy,
    )
    assert result.by_rule()["SUB_LIMIT"] == Decimal("20000.00")
    assert result.payable == Decimal("40000.00")


def test_co_pay_applies_to_what_remains_not_to_the_gross() -> None:
    policy = load_policy(co_pay_percent=10, sum_insured=500000)
    result = adjudicate(
        doc([item("Consultant Visit", "1000", Category.PROFESSIONAL_FEES),
             item("Attendant Charges", "1000", Category.NON_MEDICAL)]),
        policy=policy,
    )
    # attendant removed first, so co-pay is 10% of 1000, not of 2000
    assert result.by_rule()["CO_PAY"] == Decimal("100.00")
    assert result.payable == Decimal("900.00")


def test_stage_order_changes_the_answer_and_is_fixed() -> None:
    """Deductible before co-pay is not the same as co-pay before deductible."""
    items = [item("Consultant Visit", "10000", Category.PROFESSIONAL_FEES)]
    normal = adjudicate(doc(items),
                        policy=load_policy(deductible=2000, co_pay_percent=10))
    swapped = load_policy(deductible=2000, co_pay_percent=10)
    object.__setattr__(swapped, "order",
                       ("non_payable", "sub_limits", "room_rent_cap", "co_pay",
                        "deductible"))
    other = adjudicate(doc(items), policy=swapped)
    assert normal.payable != other.payable


def test_policy_file_declares_a_known_order() -> None:
    policy = load_policy()
    assert policy.order == ("non_payable", "sub_limits", "room_rent_cap",
                            "deductible", "co_pay")


def test_totals_are_internally_consistent() -> None:
    result = adjudicate(doc([
        item("Room Rent - Deluxe", "10000", Category.ROOM_RENT, rate="10000"),
        item("Attendant Charges", "800", Category.NON_MEDICAL),
        item("Surgeon Fee", "40000", Category.SURGERY),
    ]))
    assert result.payable + result.total_deducted == result.gross


def test_money_is_decimal_throughout() -> None:
    result = adjudicate(doc([item("Attendant Charges", "500", Category.NON_MEDICAL)]))
    assert isinstance(result.payable, Decimal)
    assert all(isinstance(d.amount, Decimal) for d in result.deductions)


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------

def _capped_document() -> Document:
    return doc([
        item("Room Rent - Deluxe", "10000", Category.ROOM_RENT, rate="10000"),
        item("Nursing Charges", "4000", Category.NURSING),
        item("Surgeon Fee", "60000", Category.SURGERY),
        item("Paracetamol 500mg", "1200", Category.PHARMACY),
    ])


def test_room_rate_error_is_amplified_when_the_cap_is_active() -> None:
    """The interview point: a 1% misread costs far more than 1% of room rent."""
    points = sweep_field(_capped_document(),
                         Perturbation("room", perturb_room_rate),
                         relatives=[Decimal("0"), Decimal("0.01")])
    one_percent = next(p for p in points if p.relative_error == Decimal("0.01"))
    assert one_percent.payable_delta != 0
    # 1% on a 10,000 room rate is Rs 100 of input error; the payable moves by
    # much more, because the proportionate factor rescales nursing and surgery.
    assert abs(one_percent.payable_delta) > Decimal("100")


def test_error_below_the_cap_is_not_amplified() -> None:
    """The discontinuity a linear error analysis would miss."""
    under = doc([
        item("Room Rent - General", "1000", Category.ROOM_RENT, rate="1000"),
        item("Nursing Charges", "4000", Category.NURSING),
    ])
    points = sweep_field(under, Perturbation("room", perturb_room_rate),
                         relatives=[Decimal("0"), Decimal("0.01")])
    delta = next(p for p in points if p.relative_error == Decimal("0.01")).payable_delta
    assert abs(delta) <= Decimal("10.00")   # only the room line itself moves


def test_sweep_is_symmetric_around_zero_error() -> None:
    points = sweep_field(_capped_document(), Perturbation("room", perturb_room_rate))
    zero = next(p for p in points if p.relative_error == 0)
    assert zero.payable_delta == 0


def test_amplification_is_reported_as_a_ratio() -> None:
    points = sweep_field(_capped_document(), Perturbation("room", perturb_room_rate))
    factor = amplification(points, Decimal("0.01"))
    assert factor is not None and factor > 0


# --------------------------------------------------------------------------
# assembly layer
# --------------------------------------------------------------------------

def _raw_item(description: str, amount: str, quantity: str = "1",
              rate: str | None = None) -> RawLineItem:
    return RawLineItem(description=description, amount=amount,
                       quantity=quantity, unit_rate=rate)


def test_pages_concatenate_in_order() -> None:
    result = assemble([
        PageFragment(1, line_items=[_raw_item("B", "200")]),
        PageFragment(0, hospital=RawHospital(name="X"),
                     line_items=[_raw_item("A", "100")]),
    ])
    assert [i.description for i in result.document.line_items] == ["A", "B"]
    assert result.document.hospital.name == "X"


def test_row_reprinted_across_a_page_break_is_removed() -> None:
    repeated = _raw_item("Room Rent", "3500")
    result = assemble([
        PageFragment(0, line_items=[_raw_item("Nursing", "800"), repeated]),
        PageFragment(1, line_items=[_raw_item("Room Rent", "3500"),
                                    _raw_item("Pharmacy", "120")]),
    ])
    assert [i.description for i in result.document.line_items] == [
        "Nursing", "Room Rent", "Pharmacy"]
    assert result.report.duplicates_removed


def test_the_same_item_billed_twice_on_one_page_is_kept() -> None:
    """Two doses of the same drug on the same day is real money, not a duplicate."""
    result = assemble([
        PageFragment(0, line_items=[_raw_item("Ondansetron 4mg Inj", "85"),
                                    _raw_item("Ondansetron 4mg Inj", "85")]),
    ])
    assert len(result.document.line_items) == 2
    assert not result.report.duplicates_removed


def test_dedup_ignores_serial_renumbering() -> None:
    a = RawLineItem(serial_no="12", description="Room Rent", amount="3500", quantity="1")
    b = RawLineItem(serial_no="1", description="Room Rent", amount="3,500.00", quantity="1")
    result = assemble([PageFragment(0, line_items=[a]),
                       PageFragment(1, line_items=[b])])
    assert len(result.document.line_items) == 1


def test_reconciliation_flags_a_totals_mismatch() -> None:
    result = assemble([PageFragment(
        0,
        line_items=[_raw_item("A", "100"), _raw_item("B", "200")],
        totals=RawTotals(gross_amount="500.00", net_amount="500.00"),
    )])
    assert not result.report.balanced
    assert result.report.gross_delta == Decimal("-200.00")
    assert any("line items sum" in f for f in result.report.flags)


def test_reconciliation_is_clean_when_the_bill_adds_up() -> None:
    result = assemble([PageFragment(
        0,
        line_items=[_raw_item("A", "100", rate="100"), _raw_item("B", "200", rate="200")],
        totals=RawTotals(gross_amount="300.00", discount="0", cgst="0", sgst="0",
                         net_amount="300.00"),
    )])
    assert result.report.balanced, result.report.arithmetic_flags


def test_row_arithmetic_failures_are_reported() -> None:
    result = assemble([PageFragment(
        0, line_items=[RawLineItem(description="A", quantity="2",
                                   unit_rate="100", amount="250")])])
    assert result.report.row_arithmetic_failures


def test_conflicting_header_values_across_pages_are_surfaced() -> None:
    """Two pages disagreeing on the patient means the pages were mis-grouped."""
    result = assemble([
        PageFragment(0, patient=RawPatient(name="Ramesh Kumar")),
        PageFragment(1, patient=RawPatient(name="Suresh Babu")),
    ])
    assert "patient.name" in result.report.conflicting_header_fields
    assert not result.report.complete
    assert any("conflicting" in f for f in result.report.flags)


def test_reconciliation_never_patches_the_numbers() -> None:
    """Reporting a disagreement is the job; silently fixing it would hide it."""
    result = assemble([PageFragment(
        0, line_items=[_raw_item("A", "100")],
        totals=RawTotals(gross_amount="999.00", net_amount="999.00"))])
    assert result.document.totals.gross_amount == "999.00"
    assert result.report.gross_delta == Decimal("-899.00")

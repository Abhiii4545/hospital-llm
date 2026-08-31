"""Schema guarantees: no floats for money, raw/typed parity, stable registries."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from reckon.schema import (
    DATE_FIELD_PATHS,
    FIELD_PATHS,
    LINE_ITEM_FIELDS,
    MONEY_FIELD_PATHS,
    Category,
    Document,
    Hospital,
    Insurance,
    LineItem,
    Patient,
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
    Totals,
    donut_special_tokens,
)


@pytest.mark.parametrize("field", ["quantity", "unit_rate", "amount"])
def test_line_item_rejects_float(field: str) -> None:
    with pytest.raises(ValidationError, match="float is forbidden"):
        LineItem(**{field: 1.5})


@pytest.mark.parametrize("field", MONEY_FIELD_PATHS)
def test_totals_reject_float(field: str) -> None:
    name = field.split(".", 1)[1]
    with pytest.raises(ValidationError, match="float is forbidden"):
        Totals(**{name: 1234.56})


def test_money_accepts_str_int_and_decimal() -> None:
    assert Totals(net_amount="1234.56").net_amount == Decimal("1234.56")
    assert Totals(net_amount=1234).net_amount == Decimal("1234")
    assert Totals(net_amount=Decimal("1234.56")).net_amount == Decimal("1234.56")


def test_money_is_exact_through_the_schema() -> None:
    t = Totals(gross_amount="0.10", discount="0.20")
    assert t.gross_amount + t.discount == Decimal("0.30")


def test_defaults_are_all_none_or_empty() -> None:
    doc = Document()
    assert doc.line_items == []
    assert doc.totals.net_amount is None
    assert doc.patient.name is None


def test_extra_fields_are_forbidden() -> None:
    """A hallucinated key must fail loudly rather than be silently dropped."""
    with pytest.raises(ValidationError):
        Patient(nmae="typo")


def test_raw_mirrors_have_identical_field_names() -> None:
    pairs = [
        (Hospital, RawHospital),
        (Patient, RawPatient),
        (Insurance, RawInsurance),
        (LineItem, RawLineItem),
        (Totals, RawTotals),
    ]
    for typed, raw in pairs:
        assert tuple(typed.model_fields) == tuple(raw.model_fields), typed.__name__


def test_raw_fields_are_all_optional_strings() -> None:
    for model in (RawHospital, RawPatient, RawInsurance, RawLineItem, RawTotals):
        for name, info in model.model_fields.items():
            assert "str" in str(info.annotation), f"{model.__name__}.{name}"


def test_raw_document_accepts_only_strings() -> None:
    raw = RawDocument()
    raw.totals.net_amount = "Rs. 1,23,456.00"
    assert raw.totals.net_amount == "Rs. 1,23,456.00"
    with pytest.raises(ValidationError):
        RawTotals(net_amount=Decimal("1234"))


def test_field_registry_matches_the_brief() -> None:
    assert len(FIELD_PATHS) == 27
    for path in (
        "hospital.name", "hospital.gstin", "hospital.hospital_type",
        "patient.uhid", "patient.admission_date", "patient.ward_type",
        "insurance.tpa_name", "insurance.employee_id",
        "totals.net_amount", "totals.amount_in_words",
    ):
        assert path in FIELD_PATHS
    assert len(set(FIELD_PATHS)) == len(FIELD_PATHS)


def test_line_item_registry() -> None:
    assert LINE_ITEM_FIELDS == (
        "serial_no", "description", "service_date", "category", "quantity",
        "unit_rate", "amount", "hsn_code", "is_payable", "deduction_reason",
    )


def test_money_and_date_registries() -> None:
    assert MONEY_FIELD_PATHS == (
        "totals.gross_amount", "totals.discount", "totals.cgst", "totals.sgst",
        "totals.net_amount", "totals.advance_paid", "totals.balance_due",
    )
    assert DATE_FIELD_PATHS == (
        "patient.admission_date", "patient.discharge_date",
    )


def test_category_enum_is_exactly_the_brief() -> None:
    assert [c.value for c in Category] == [
        "room_rent", "nursing", "consumables", "pharmacy", "diagnostics",
        "radiology", "surgery", "professional_fees", "equipment",
        "administrative", "non_medical", "other",
    ]


def test_donut_special_tokens_are_unique_and_paired() -> None:
    tokens = donut_special_tokens()
    assert len(tokens) == len(set(tokens))
    opens = {t for t in tokens if t.startswith("<s_")}
    closes = {t for t in tokens if t.startswith("</s_")}
    assert len(opens) == len(closes)
    for t in opens:
        assert t.replace("<s_", "</s_") in closes
    assert "<sep/>" in tokens
    for key in ("hospital", "line_items", "net_amount", "deduction_reason", "ward_type"):
        assert f"<s_{key}>" in tokens


def test_closed_vocabulary_values_get_their_own_tokens() -> None:
    """Category values are emitted verbatim, so each is worth one decoder step."""
    tokens = set(donut_special_tokens())
    for category in Category:
        assert category.value in tokens, category.value


def test_document_round_trips_through_json() -> None:
    doc = Document(
        hospital=Hospital(name="apollo hospitals", gstin="27AAPFU0939F1ZV"),  # pii-allow
        patient=Patient(name="ramesh kumar", age=45, admission_date=date(2025, 1, 5)),
        line_items=[
            LineItem(
                description="room rent - deluxe",
                category=Category.ROOM_RENT,
                quantity=Decimal("2"),
                unit_rate=Decimal("4500.00"),
                amount=Decimal("9000.00"),
                is_payable=True,
            )
        ],
        totals=Totals(net_amount=Decimal("9000.00")),
    )
    restored = Document.model_validate_json(doc.model_dump_json())
    assert restored == doc
    assert restored.totals.net_amount == Decimal("9000.00")
    assert isinstance(restored.line_items[0].amount, Decimal)

"""End-to-end RawDocument -> Document normalization."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from reckon.normalize import NORMALIZERS, normalize_document
from reckon.schema import (
    FIELD_PATHS,
    LINE_ITEM_FIELDS,
    Category,
    Document,
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)


def _messy_raw() -> RawDocument:
    """A deliberately ugly but entirely plausible extractor output."""
    return RawDocument(
        hospital=RawHospital(
            name="  Dr. Rao's Multi-Speciality Hospital  ",
            address="8-2-120,  Road No. 2,\nBanjara Hills",
            city="Hyderabad",
            state="Telangana",
            gstin="36-aapfu 0939f1zv",
            hospital_type="Multi Speciality",
        ),
        patient=RawPatient(
            name="Mr. M. Aadithya Ram",
            age="45 Y",
            sex="M",
            uhid="UH/2025/00123",
            ip_number="IP 4521",
            admission_date="05-01-2025",
            discharge_date="9th Jan 2025",
            ward_type="Semi-Private",
        ),
        insurance=RawInsurance(
            insurer_name="Star Health & Allied Insurance Co. Ltd.",
            tpa_name="Medi Assist",
            policy_number="POL-1234 5678",
            claim_number="CLM/25/998877",
            employee_id="EMP-0042",
        ),
        line_items=[
            RawLineItem(
                serial_no="1.",
                description="Room  Rent - Semi Private",
                service_date="05/01/2025",
                category="Room Rent",
                quantity="4",
                unit_rate="Rs. 3,500.00",
                amount="Rs. 14,000.00",
                hsn_code="9993",
                is_payable="Y",
                deduction_reason=None,
            ),
            RawLineItem(
                serial_no="2)",
                description="Attendant Charges",
                service_date="6 Jan 2025",
                category="Non-Medical",
                quantity="4",
                unit_rate="250/-",
                amount="1,000/-",
                hsn_code=None,
                is_payable="Non-Payable",
                deduction_reason="IRDAI List I - attendant charges",
            ),
        ],
        totals=RawTotals(
            gross_amount="Rs. 1,23,456.00",
            discount="NIL",
            cgst="1,111.10",
            sgst="1,111.10",
            net_amount="₹1,25,678.20",
            advance_paid="50,000/-",
            balance_due="75,678.20",
            amount_in_words="Rupees One Lakh Twenty Five Thousand ...",
        ),
    )


def test_full_document_normalizes() -> None:
    doc = normalize_document(_messy_raw())
    assert isinstance(doc, Document)

    assert doc.hospital.name == "dr. rao's multi-speciality hospital"
    assert doc.hospital.address == "8-2-120, road no. 2, banjara hills"
    assert doc.hospital.gstin == "36AAPFU0939F1ZV"  # pii-allow

    assert doc.patient.name == "m aadithya ram"
    assert doc.patient.age == 45
    assert doc.patient.sex == "male"
    assert doc.patient.uhid == "UH202500123"
    assert doc.patient.ip_number == "IP4521"
    assert doc.patient.admission_date == date(2025, 1, 5)
    assert doc.patient.discharge_date == date(2025, 1, 9)
    assert doc.patient.ward_type == "semi_private"

    assert doc.insurance.policy_number == "POL12345678"
    assert doc.insurance.claim_number == "CLM25998877"


def test_line_items_normalize_with_decimal_money() -> None:
    doc = normalize_document(_messy_raw())
    assert len(doc.line_items) == 2

    first, second = doc.line_items
    assert first.serial_no == "1"
    assert first.category is Category.ROOM_RENT
    assert first.quantity == Decimal("4")
    assert first.unit_rate == Decimal("3500.00")
    assert first.amount == Decimal("14000.00")
    assert first.is_payable is True
    assert isinstance(first.amount, Decimal)

    assert second.serial_no == "2"
    assert second.category is Category.NON_MEDICAL
    assert second.amount == Decimal("1000")
    assert second.is_payable is False
    assert second.service_date == date(2025, 1, 6)


def test_totals_normalize() -> None:
    t = normalize_document(_messy_raw()).totals
    assert t.gross_amount == Decimal("123456.00")
    assert t.discount == Decimal("0")     # "NIL" is zero, not missing
    assert t.net_amount == Decimal("125678.20")
    assert t.advance_paid == Decimal("50000")
    assert t.balance_due == Decimal("75678.20")


def test_line_item_quantity_times_rate_reconciles() -> None:
    """Exact Decimal arithmetic: 4 x 3500.00 is 14000.00 with no rounding drift."""
    item = normalize_document(_messy_raw()).line_items[0]
    assert item.quantity * item.unit_rate == item.amount


def test_empty_raw_document_gives_empty_typed_document() -> None:
    assert normalize_document(RawDocument()) == Document()


def test_garbage_normalizes_to_none_and_never_raises() -> None:
    raw = RawDocument()
    raw.patient.admission_date = "!!!"
    raw.totals.net_amount = "not a number"
    raw.patient.age = "adult"
    raw.patient.sex = "???"
    doc = normalize_document(raw)
    assert doc.patient.admission_date is None
    assert doc.totals.net_amount is None
    assert doc.patient.age is None
    assert doc.patient.sex is None


def test_normalizer_registry_covers_every_field_exactly_once() -> None:
    """A new schema field without a normalizer must fail here, not in production."""
    assert set(NORMALIZERS) == set(FIELD_PATHS)


def test_line_item_normalizer_registry_is_complete() -> None:
    from reckon.normalize import LINE_ITEM_NORMALIZERS

    assert set(LINE_ITEM_NORMALIZERS) == set(LINE_ITEM_FIELDS)


def test_normalization_is_idempotent() -> None:
    """Normalizing an already-normalized value must not change it further.

    Without this, raw-vs-normalized metrics would depend on how many times the
    pipeline happened to normalize.
    """
    doc = normalize_document(_messy_raw())
    again = RawDocument(
        hospital=RawHospital(name=doc.hospital.name, gstin=doc.hospital.gstin),
        patient=RawPatient(
            name=doc.patient.name,
            uhid=doc.patient.uhid,
            sex=doc.patient.sex,
            ward_type=doc.patient.ward_type,
            admission_date=doc.patient.admission_date.isoformat(),
        ),
    )
    twice = normalize_document(again)
    assert twice.hospital.name == doc.hospital.name
    assert twice.hospital.gstin == doc.hospital.gstin
    assert twice.patient.name == doc.patient.name
    assert twice.patient.uhid == doc.patient.uhid
    assert twice.patient.sex == doc.patient.sex
    assert twice.patient.ward_type == doc.patient.ward_type
    assert twice.patient.admission_date == doc.patient.admission_date

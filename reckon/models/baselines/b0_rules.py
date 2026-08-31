"""B0 - the RECKON v1 rules engine, reimplemented faithfully.

This is the honest incumbent. It is label-driven regex over recognised text: for
each field there is a list of label spellings, and for line items there are a
handful of row patterns accumulated over time by looking at whichever layouts
happened to arrive.

Its weakness is deliberate and is the entire reason v2 exists: the row patterns
encode a fixed column ORDER. A hospital that puts rate before quantity, or
separates cells with pipes, or writes "3. Item x2 @ 500 = 1000", matches nothing
and the bill silently comes back with zero line items. Every such hospital
required a new pattern, and the patterns interfered with each other as they
accumulated.

Do not "improve" this to make v2's win smaller or larger. It is a measurement
instrument, and it is supposed to represent what v1 actually did.
"""

from __future__ import annotations

import re
from typing import Sequence

from reckon.models.baselines.ocr import OcrBackend, OcrPage, PlainTextBackend
from reckon.schema import (
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = ["B0RulesEngine", "extract"]

#: Field path -> label spellings, in priority order. This list is exactly the
#: kind of artefact that grows without bound in a rules engine.
_LABELS: dict[str, tuple[str, ...]] = {
    "hospital.gstin": ("gstin", "gst no", "gst number"),
    "patient.name": ("patient name", "patient", "name of patient", "name"),
    "patient.age": ("age / sex", "age/sex", "age"),
    "patient.uhid": ("uhid", "uhid no", "mr no", "hospital no"),
    "patient.ip_number": ("ip no", "ip number", "ipd no", "admission no"),
    "patient.admission_date": ("admission", "date of admission", "doa", "adm"),
    "patient.discharge_date": ("discharge", "date of discharge", "dod", "dis"),
    "patient.ward_type": ("ward type", "ward", "room category", "bed type"),
    "insurance.insurer_name": ("insurer", "insurance company", "insurance"),
    "insurance.tpa_name": ("tpa", "tpa name"),
    "insurance.policy_number": ("policy no", "policy number", "policy"),
    "insurance.claim_number": ("claim no", "claim number", "claim"),
    "insurance.employee_id": ("employee id", "emp id", "emp"),
    "totals.gross_amount": ("gross amount", "total amount", "gross", "sub total"),
    "totals.discount": ("discount", "concession"),
    "totals.cgst": ("cgst",),
    "totals.sgst": ("sgst",),
    "totals.net_amount": ("net amount", "net payable", "grand total", "net"),
    "totals.advance_paid": ("advance paid", "advance", "paid"),
    "totals.balance_due": ("balance due", "balance", "due"),
    "totals.amount_in_words": ("amount in words", "in words", "rupees in words"),
}

_AMOUNT = r"(?:Rs\.?|INR|₹)?\s*[\d,]+(?:\.\d{1,2})?(?:/-)?"
_DATE = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"

#: Row patterns, in the order they were added over v1's life. The order matters
#: and that is itself a bug that was never fixed.
_ROW_PATTERNS: tuple[re.Pattern[str], ...] = (
    # serial, description, date, qty, rate, amount  (the original layout)
    re.compile(
        rf"^\s*(?P<serial>\d+)\s+(?P<description>\S.*?)\s{{2,}}(?P<service_date>{_DATE})"
        rf"\s+(?P<quantity>\d+)\s+(?P<unit_rate>{_AMOUNT})\s+(?P<amount>{_AMOUNT})\s*$"
    ),
    # serial, description, qty, rate, amount  (added for the diagnostic centre)
    re.compile(
        rf"^\s*(?P<serial>\d+)\s+(?P<description>\S.*?)\s{{2,}}(?P<quantity>\d+)"
        rf"\s+(?P<unit_rate>{_AMOUNT})\s+(?P<amount>{_AMOUNT})\s*$"
    ),
    # serial, description, amount  (added in a hurry for a small nursing home)
    re.compile(
        rf"^\s*(?P<serial>\d+)[.)]?\s+(?P<description>\S.*?)\s{{2,}}(?P<amount>{_AMOUNT})\s*$"
    ),
)

#: v1's keyword rules for category and payability, kept verbatim.
_CATEGORY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("room|bed charge", "room_rent"),
    ("nurs", "nursing"),
    ("pharm|medicine|inject", "pharmacy"),
    ("glove|syringe|consumable|cannula|iv set", "consumables"),
    ("x-ray|xray|mri|ct |ultraso|scan", "radiology"),
    ("blood|test|lab|function", "diagnostics"),
    ("surge|ot |operat|anaesth", "surgery"),
    ("consult|physician|doctor", "professional_fees"),
    ("ventilat|oxygen|monitor", "equipment"),
    ("registration|record|admission fee", "administrative"),
    ("attendant|telephone|food|toiletr", "non_medical"),
)

_NON_PAYABLE_KEYWORDS = re.compile(
    r"attendant|telephone|toiletr|food|registration|record|admission fee",
    re.IGNORECASE,
)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().strip(":").strip()
    return text or None


def _find_labelled(lines: Sequence[str], labels: Sequence[str]) -> str | None:
    """First value found for any spelling of a label.

    Handles `Label : value` and two-column headers where a second label starts
    later on the same line.
    """
    for label in labels:
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(label)}\s*[:\-]\s*(?P<value>.+?)"
            rf"(?=\s{{2,}}[A-Z][A-Za-z/ ]{{2,}}\s*[:\-]|$)",
            re.IGNORECASE,
        )
        for line in lines:
            match = pattern.search(line)
            if match:
                value = _clean(match.group("value"))
                if value:
                    return value
    return None


def _category_for(description: str) -> str | None:
    for pattern, category in _CATEGORY_KEYWORDS:
        if re.search(pattern, description, re.IGNORECASE):
            return category
    return None


class B0RulesEngine:
    """Label regex plus fixed-order row patterns. The incumbent."""

    name = "B0 (v1 rules)"

    def __init__(self, backend: OcrBackend | None = None) -> None:
        self.backend = backend or PlainTextBackend()

    def extract(self, source: object) -> RawDocument:
        page: OcrPage = self.backend.read(source)
        lines = [line.text for line in page.lines]

        hospital = RawHospital(
            # v1 assumed the hospital name was simply the first non-empty line.
            name=next((line.strip() for line in lines if line.strip()), None),
            address=None,
            city=None,
            state=None,
            gstin=self._gstin(lines),
            hospital_type=None,
        )

        patient = RawPatient(
            name=_find_labelled(lines, _LABELS["patient.name"]),
            age=_find_labelled(lines, _LABELS["patient.age"]),
            sex=self._sex(lines),
            uhid=_find_labelled(lines, _LABELS["patient.uhid"]),
            ip_number=_find_labelled(lines, _LABELS["patient.ip_number"]),
            admission_date=_find_labelled(lines, _LABELS["patient.admission_date"]),
            discharge_date=_find_labelled(lines, _LABELS["patient.discharge_date"]),
            ward_type=_find_labelled(lines, _LABELS["patient.ward_type"]),
        )

        insurance = RawInsurance(
            insurer_name=_find_labelled(lines, _LABELS["insurance.insurer_name"]),
            tpa_name=_find_labelled(lines, _LABELS["insurance.tpa_name"]),
            policy_number=_find_labelled(lines, _LABELS["insurance.policy_number"]),
            claim_number=_find_labelled(lines, _LABELS["insurance.claim_number"]),
            employee_id=_find_labelled(lines, _LABELS["insurance.employee_id"]),
        )

        totals = RawTotals(
            **{
                path.split(".", 1)[1]: _find_labelled(lines, labels)
                for path, labels in _LABELS.items()
                if path.startswith("totals.")
            }
        )

        return RawDocument(
            hospital=hospital,
            patient=patient,
            insurance=insurance,
            line_items=self._rows(lines),
            totals=totals,
        )

    # -- pieces ----------------------------------------------------------

    def _gstin(self, lines: Sequence[str]) -> str | None:
        value = _find_labelled(lines, _LABELS["hospital.gstin"])
        if value:
            return value.split()[0] if " " in value else value
        for line in lines:
            match = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", line)
            if match:
                return match.group(0)
        return None

    def _sex(self, lines: Sequence[str]) -> str | None:
        """v1 read sex out of the combined 'Age / Sex : 45 Y / M' field."""
        combined = _find_labelled(lines, ("age / sex", "age/sex"))
        if combined and "/" in combined:
            return _clean(combined.rsplit("/", 1)[1])
        return _find_labelled(lines, ("sex", "gender"))

    def _rows(self, lines: Sequence[str]) -> list[RawLineItem]:
        items: list[RawLineItem] = []
        for line in lines:
            for pattern in _ROW_PATTERNS:
                match = pattern.match(line)
                if not match:
                    continue
                groups = match.groupdict()
                description = _clean(groups.get("description")) or ""
                if not description or description.lower().startswith(
                    ("gross", "net", "total", "discount", "cgst", "sgst", "advance", "balance")
                ):
                    break
                items.append(
                    RawLineItem(
                        serial_no=_clean(groups.get("serial")),
                        description=description,
                        service_date=_clean(groups.get("service_date")),
                        category=_category_for(description),
                        quantity=_clean(groups.get("quantity")),
                        unit_rate=_clean(groups.get("unit_rate")),
                        amount=_clean(groups.get("amount")),
                        hsn_code=None,
                        is_payable="N" if _NON_PAYABLE_KEYWORDS.search(description) else "Y",
                        deduction_reason=(
                            "IRDAI List I - non-medical expense"
                            if _NON_PAYABLE_KEYWORDS.search(description)
                            else None
                        ),
                    )
                )
                break
        return items


def extract(source: object, backend: OcrBackend | None = None) -> RawDocument:
    return B0RulesEngine(backend).extract(source)

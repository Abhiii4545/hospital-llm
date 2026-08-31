"""Assemble a complete synthetic document: values, formatting, pages, truth.

The single most important property here: **the ground truth is the exact string
that gets printed**. Formatting and truth come out of one code path, so it is
structurally impossible for the label to disagree with the pixels. A corpus where
truth is generated separately from the render is a corpus that quietly teaches a
model to be wrong.

Pages are first-class. The architecture is page-level and two-headed, so this
module decides which line items land on which page, which pages carry the header
block, and which carries the totals - and emits per-page targets for Head A and
Head B directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from reckon.data.generators.values import (
    BILINGUAL_STATES,
    STATE_REGION,
    GeneratedItem,
    money,
    sample_hospital,
    sample_insurer,
    sample_line_items,
    sample_patient_name,
    sample_policy_number,
    sample_ward,
)
from reckon.schema import (
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = ["Formatter", "Page", "GeneratedDocument", "generate_document", "MESSINESS_RATES"]

#: Controlled messiness, at the rates the brief specifies.
MESSINESS_RATES: dict[str, float] = {
    "missing_uhid": 0.08,
    "misaligned_totals": 0.05,
    "duplicate_row_across_page_break": 0.12,
    "handwritten_correction": 0.06,
}

_DATE_STYLES = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y", "%d-%m-%y")


@dataclass(frozen=True)
class Formatter:
    """How one hospital prints things. Fixed per document, varied across them."""

    date_style: str
    currency_prefix: str      # "", "Rs. ", "₹", "INR "
    trailing_slash: bool      # "500/-"
    group_indian: bool        # 1,23,456.00 vs 123456.00
    nil_token: str            # how a zero is printed: "NIL", "-", "0.00"

    def date(self, value: date) -> str:
        return value.strftime(self.date_style)

    def amount(self, value: Decimal, *, allow_nil: bool = False) -> str:
        if allow_nil and value == 0:
            return self.nil_token
        negative = value < 0
        whole = abs(value).quantize(Decimal("0.01"))
        rupees, paise = divmod(int(whole * 100), 100)
        digits = self._group(str(rupees)) if self.group_indian else str(rupees)
        text = f"{self.currency_prefix}{digits}.{paise:02d}"
        if self.trailing_slash:
            text += "/-"
        return f"({text})" if negative else text

    @staticmethod
    def _group(digits: str) -> str:
        """Indian digit grouping: last three, then pairs."""
        if len(digits) <= 3:
            return digits
        head, tail = digits[:-3], digits[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        return ",".join(groups + [tail])

    def quantity(self, value: Decimal) -> str:
        return str(int(value)) if value == value.to_integral_value() else str(value)


def sample_formatter(rng: random.Random) -> Formatter:
    return Formatter(
        date_style=rng.choice(_DATE_STYLES),
        currency_prefix=rng.choices(["", "Rs. ", "₹", "INR "],
                                    weights=[0.45, 0.32, 0.15, 0.08], k=1)[0],
        trailing_slash=rng.random() < 0.18,
        group_indian=rng.random() < 0.80,
        nil_token=rng.choices(["NIL", "-", "0.00"], weights=[0.4, 0.3, 0.3], k=1)[0],
    )


@dataclass
class Page:
    """One rendered page and its per-head targets."""

    index: int
    total_pages: int
    rows: list[dict[str, str]] = field(default_factory=list)
    items: list[RawLineItem] = field(default_factory=list)
    show_header: bool = False
    show_totals: bool = False
    continued: bool = False
    duplicate_of_previous_last: bool = False


@dataclass
class GeneratedDocument:
    doc_id: str
    layout_id: str
    formatter: "Formatter"
    context: dict[str, Any]          # everything the Jinja template needs
    pages: list[Page]
    truth: RawDocument
    meta: dict[str, Any]


def _rows_for(items: list[GeneratedItem], fmt: Formatter, start: int,
              service_dates: list[date]) -> list[dict[str, str]]:
    rows = []
    for offset, item in enumerate(items):
        rows.append({
            "serial_no": str(start + offset),
            "description": item.description,
            "service_date": fmt.date(service_dates[offset]),
            "category": item.category.replace("_", " ").title(),
            "quantity": fmt.quantity(item.quantity),
            "unit_rate": fmt.amount(item.unit_rate),
            "amount": fmt.amount(item.amount),
            "hsn_code": item.hsn_code or "",
            "batch": f"B{start + offset:04d}",
            "expiry": f"{(start + offset) % 12 + 1:02d}/202{(start + offset) % 6 + 5}",
        })
    return rows


def generate_document(
    rng: random.Random,
    doc_id: str,
    layout_id: str,
    rows_per_page: int,
    *,
    hospital_type: str | None = None,
    bilingual: bool = False,
) -> GeneratedDocument:
    fmt = sample_formatter(rng)
    # A bilingual layout may only sit in a state whose script this project has
    # header strings for; otherwise the corpus would assert, for instance, that
    # Tamil Nadu government hospitals print Telugu.
    hospital = sample_hospital(
        rng, hospital_type,
        allowed_states=tuple(BILINGUAL_STATES) if bilingual else None,
    )
    sex = rng.choice(["M", "F"])
    age = rng.randint(1, 92)
    patient_name, region = sample_patient_name(
        rng, sex, age, region=STATE_REGION.get(hospital.state_code)
    )
    insurer, tpa = sample_insurer(rng)
    ward, ward_label = sample_ward(rng)

    admission = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
    stay_days = max(1, int(rng.choices([1, 2, 3, 4, 5, 7, 10, 14, 21],
                                       weights=[8, 14, 18, 16, 12, 12, 8, 7, 5],
                                       k=1)[0]))
    discharge = admission + timedelta(days=stay_days)

    n_items = rng.choices([4, 7, 12, 18, 26, 34, 45, 60],
                          weights=[10, 16, 20, 18, 14, 10, 8, 4], k=1)[0]
    items = sample_line_items(rng, ward=ward, stay_days=stay_days,
                              n_items=n_items, include_surgery=rng.random() < 0.35)
    service_dates = [admission + timedelta(days=rng.randint(0, stay_days))
                     for _ in items]

    # -- messiness -------------------------------------------------------
    missing_uhid = rng.random() < MESSINESS_RATES["missing_uhid"]
    misaligned = rng.random() < MESSINESS_RATES["misaligned_totals"]
    duplicate_row = rng.random() < MESSINESS_RATES["duplicate_row_across_page_break"]
    handwritten = rng.random() < MESSINESS_RATES["handwritten_correction"]

    # -- totals ----------------------------------------------------------
    gross = money(sum((i.amount for i in items), Decimal(0)))
    discount = money(gross * Decimal(rng.choice(["0", "0", "0", "0.02", "0.05"])))
    taxable = gross - discount
    gst_rate = Decimal(rng.choice(["0", "0", "0.025", "0.06"]))
    cgst = money(taxable * gst_rate)
    sgst = money(cgst)
    net = money(taxable + cgst + sgst)
    if misaligned:
        # A genuinely inconsistent printed total, as happens on real bills.
        net = money(net + Decimal(rng.choice(["-13.40", "10.00", "-1.00", "100.00"])))
    advance = money(net * Decimal(rng.choice(["0", "0.25", "0.4", "0.5", "1.0"])))
    balance = money(net - advance)

    # -- paginate --------------------------------------------------------
    pages: list[Page] = []
    chunks = [items[i:i + rows_per_page] for i in range(0, len(items), rows_per_page)] or [[]]
    total_pages = len(chunks)
    cursor = 1
    for page_index, chunk in enumerate(chunks):
        dates_slice = service_dates[cursor - 1: cursor - 1 + len(chunk)]
        rows = _rows_for(chunk, fmt, cursor, dates_slice)
        raw_items = [
            RawLineItem(
                serial_no=row["serial_no"],
                description=item.description,
                service_date=row["service_date"],
                category=row["category"],
                quantity=row["quantity"],
                unit_rate=row["unit_rate"],
                amount=row["amount"],
                hsn_code=item.hsn_code,
                is_payable="Y" if item.is_payable else "N",
                deduction_reason=(
                    None if item.is_payable else "IRDAI List I - non-medical expense"
                ),
            )
            for row, item in zip(rows, chunk)
        ]

        duplicated = False
        if duplicate_row and page_index > 0 and pages[-1].rows:
            # The last row of the previous page is REPRINTED as the first row
            # here. Ground truth keeps one copy; the assembly layer has to notice.
            rows.insert(0, dict(pages[-1].rows[-1]))
            duplicated = True

        pages.append(Page(
            index=page_index,
            total_pages=total_pages,
            rows=rows,
            items=raw_items,
            show_header=page_index == 0,
            show_totals=page_index == total_pages - 1,
            continued=page_index > 0,
            duplicate_of_previous_last=duplicated,
        ))
        cursor += len(chunk)

    # -- raw truth: exactly what is printed -------------------------------
    uhid = None if missing_uhid else f"UH{rng.randint(10**5, 10**6 - 1)}"
    truth = RawDocument(
        hospital=RawHospital(
            name=hospital.name, address=hospital.address, city=hospital.city,
            state=hospital.state, gstin=hospital.gstin,
            hospital_type=hospital.hospital_type,
        ),
        patient=RawPatient(
            name=patient_name,
            age=f"{age}",
            sex=(sex if rng.random() < 0.8
                 else ("Male" if sex == "M" else "Female")),
            uhid=uhid,
            ip_number=f"IP{rng.randint(1000, 99999)}",
            admission_date=fmt.date(admission),
            discharge_date=fmt.date(discharge),
            ward_type=ward_label,
        ),
        insurance=RawInsurance(
            insurer_name=insurer, tpa_name=tpa,
            policy_number=sample_policy_number(rng, insurer),
            claim_number=f"CLM{rng.randint(10**6, 10**7 - 1)}",
            employee_id=f"EMP{rng.randint(1000, 99999)}" if rng.random() < 0.5 else None,
        ),
        line_items=[i for page in pages for i in page.items],
        totals=RawTotals(
            gross_amount=fmt.amount(gross),
            discount=fmt.amount(discount, allow_nil=True),
            cgst=fmt.amount(cgst, allow_nil=True),
            sgst=fmt.amount(sgst, allow_nil=True),
            net_amount=fmt.amount(net),
            advance_paid=fmt.amount(advance, allow_nil=True),
            balance_due=fmt.amount(balance),
            amount_in_words=None,
        ),
    )

    context: dict[str, Any] = {
        "hospital": truth.hospital.model_dump(),
        "patient": truth.patient.model_dump(),
        "insurance": truth.insurance.model_dump(),
        "totals": truth.totals.model_dump(),
        "bilingual": bilingual,
        "script": BILINGUAL_STATES.get(hospital.state_code, "hindi"),
        "handwritten": handwritten,
        "doc_id": doc_id,
        "invoice_no": f"INV/{rng.randint(1000, 9999)}/{admission.year}",
    }

    return GeneratedDocument(
        doc_id=doc_id,
        layout_id=layout_id,
        formatter=fmt,
        context=context,
        pages=pages,
        truth=truth,
        meta={
            "source": "synthetic",
            "layout": layout_id,
            "template": layout_id,
            "pages": str(total_pages),
            "page_bucket": "1" if total_pages == 1 else ("2-3" if total_pages <= 3 else "4+"),
            "language": "bilingual" if bilingual else "english",
            "region": region,
            "state_code": hospital.state_code,
            "script": BILINGUAL_STATES.get(hospital.state_code, "hindi") if bilingual else "none",
            "ward": ward,
            "hospital_type": hospital.hospital_type,
            "n_line_items": str(len(items)),
            "messiness": ",".join(
                tag for tag, on in (
                    ("missing_uhid", missing_uhid),
                    ("misaligned_totals", misaligned),
                    ("duplicate_row", duplicate_row and total_pages > 1),
                    ("handwritten", handwritten),
                ) if on
            ) or "none",
        },
    )

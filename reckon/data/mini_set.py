"""The 20-document Phase 2 mini-set.

WHAT THIS IS: a harness smoke-test. Twenty documents across five structurally
different layouts, each rendered from ground-truth values that are known exactly
by construction, so the metrics can be exercised end-to-end against B0 and B1
before any real or synthetic corpus exists.

WHAT THIS IS NOT: a benchmark. Numbers measured here are optimistic in ways that
must be stated wherever they are quoted:

* the text is noise-free, so B1's OCR stage is never actually exercised - its
  score here is an upper bound it will not reach on a scanned page;
* five layouts is not layout variance, and layout variance is exactly what broke
  RECKON v1;
* the renderer and the parsers were written by the same person on the same day,
  which is the classic way a baseline gets accidentally flattered.

The real numbers arrive in Phase 3 (synthetic corpus) and Phase 6 (real-test).

Output is regenerated from this file rather than committed, so the fixtures
cannot drift from the code that produces them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reckon.schema import (
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = ["MiniDoc", "build_mini_set", "write_mini_set", "load_mini_set", "DEFAULT_DIR"]

DEFAULT_DIR = Path("data/mini")
SEED = 1337

_HOSPITALS = [
    # (name, address, city, state, gstin, type)
    ("Sunrise Multi-Speciality Hospital", "Plot 42, Jubilee Hills", "Hyderabad",
     "Telangana", "36AABCS1429B1Z6", "multi_speciality"),  # pii-allow
    ("Lakeview Medical Centre", "18 Ring Road", "Nagpur",
     "Maharashtra", "27AABCL2938K1Z2", "general"),  # pii-allow
    ("Sri Venkatesh Nursing Home", "Door 5-8-21, Market Street", "Guntur",
     "Andhra Pradesh", "37AACCS8812M1Z4", "nursing_home"),  # pii-allow
    ("District Government Hospital", "Collectorate Road", "Warangal",
     "Telangana", "36AAAGD1122P1Z8", "government"),  # pii-allow
    ("Precision Diagnostics", "2nd Floor, Anand Complex", "Vijayawada",
     "Andhra Pradesh", "37AADCP4471N1Z0", "diagnostic_centre"),  # pii-allow
]

_PATIENTS = [
    ("Mr. Ramesh Kumar", "45", "M"), ("Smt. Lakshmi Devi", "62", "F"),
    ("M. Aadithya Ram", "29", "M"), ("Mrs. Kavitha Rao", "38", "F"),
    ("Master Arjun Reddy", "7", "M"), ("Sri Venkatesh Naidu", "71", "M"),
    ("Ms. Priya Sharma", "33", "F"), ("Baby of Anitha", "0", "F"),
    ("Dr. Suresh Babu", "54", "M"), ("Mrudula Devi", "48", "F"),
]

_INSURERS = [
    ("Star Health & Allied Insurance", "Medi Assist"),
    ("New India Assurance", "Paramount TPA"),
    ("HDFC ERGO General Insurance", "Health India TPA"),
    ("Oriental Insurance Company", "Vidal Health TPA"),
]

_WARDS = ["General Ward", "Semi-Private", "Private", "Deluxe Room", "ICU"]

# (description, category label, unit rate)
_SERVICES = [
    ("Room Rent", "Room Rent", 3500), ("Nursing Charges", "Nursing", 800),
    ("Consultation Charges", "Professional Fees", 900),
    ("Surgeon Fee", "Surgery", 25000), ("OT Charges", "Surgery", 12000),
    ("Complete Blood Count", "Diagnostics", 350),
    ("Liver Function Test", "Diagnostics", 750),
    ("X-Ray Chest PA", "Radiology", 450), ("MRI Brain", "Radiology", 8500),
    ("Ultrasound Abdomen", "Radiology", 1200),
    ("Pharmacy - Injectables", "Pharmacy", 2400),
    ("Pharmacy - Oral Medication", "Pharmacy", 860),
    ("Surgical Gloves", "Consumables", 180),
    ("IV Set and Cannula", "Consumables", 320),
    ("Ventilator Charges", "Equipment", 6000),
    ("Oxygen Charges", "Equipment", 1500),
    ("Registration Fee", "Administrative", 250),
    ("Medical Records Charge", "Administrative", 150),
    ("Attendant Charges", "Non-Medical", 500),
    ("Telephone Charges", "Non-Medical", 120),
]

#: Categories IRDAI List I treats as non-payable. Used to set is_payable in the
#: ground truth so the business metric has something real to measure.
_NON_PAYABLE_LABELS = {"Non-Medical", "Administrative"}


@dataclass
class MiniDoc:
    doc_id: str
    text: str
    truth: RawDocument
    meta: dict[str, Any]


def _rupees(value: int | float) -> str:
    """Indian digit grouping, two decimals."""
    whole = int(round(float(value) * 100))
    sign = "-" if whole < 0 else ""
    whole = abs(whole)
    rupees, paise = divmod(whole, 100)
    digits = str(rupees)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{sign}{digits}.{paise:02d}"


def _date(day: int, month: int = 1, year: int = 2025) -> str:
    return f"{day:02d}/{month:02d}/{year}"


def _build_one(index: int, rng: random.Random) -> MiniDoc:
    layout = index % 5
    hospital = _HOSPITALS[layout]
    patient = _PATIENTS[index % len(_PATIENTS)]
    insurer = _INSURERS[index % len(_INSURERS)]
    ward = _WARDS[index % len(_WARDS)]

    admit_day = 1 + (index % 20)
    stay = 2 + (index % 5)
    discharge_day = min(28, admit_day + stay)

    # --- deliberate messiness, at the rates the brief specifies -------------
    missing_uhid = index % 5 == 2
    misaligned_totals = index % 10 == 4
    duplicate_row = index % 6 == 3
    handwritten_note = index % 9 == 6

    chosen = rng.sample(_SERVICES, k=4 + (index % 4))
    items: list[RawLineItem] = []
    rendered_rows: list[tuple[str, ...]] = []
    gross = 0

    for position, (description, category, rate) in enumerate(chosen, start=1):
        quantity = stay if description == "Room Rent" else 1 + (position % 3)
        amount = rate * quantity
        gross += amount
        payable = "N" if category in _NON_PAYABLE_LABELS else "Y"
        items.append(
            RawLineItem(
                serial_no=str(position),
                description=description,
                service_date=_date(admit_day),
                category=category,
                quantity=str(quantity),
                unit_rate=_rupees(rate),
                amount=_rupees(amount),
                is_payable=payable,
                deduction_reason=(
                    "IRDAI List I - non-medical expense" if payable == "N" else None
                ),
            )
        )
        rendered_rows.append((
            str(position), description, _date(admit_day),
            str(quantity), _rupees(rate), _rupees(amount),
        ))

    if duplicate_row and rendered_rows:
        # A row repeated across a page break. Ground truth keeps ONE copy; the
        # assembly layer is what has to notice. A parser that emits both is
        # scored with an insertion, which is the correct penalty.
        rendered_rows.append(rendered_rows[-1])

    discount = 0
    cgst = round(gross * 0.025, 2)
    sgst = round(gross * 0.025, 2)
    net = gross - discount + cgst + sgst
    if misaligned_totals:
        net = round(net - 13.40, 2)   # a genuinely inconsistent printed total

    advance = round(net * 0.4, 2)
    balance = round(net - advance, 2)

    truth = RawDocument(
        hospital=RawHospital(
            name=hospital[0], address=hospital[1], city=hospital[2],
            state=hospital[3], gstin=hospital[4], hospital_type=hospital[5],
        ),
        patient=RawPatient(
            name=patient[0], age=patient[1] + " Y", sex=patient[2],
            uhid=None if missing_uhid else f"UH/2025/{1000 + index}",
            ip_number=f"IP{4500 + index}",
            admission_date=_date(admit_day),
            discharge_date=_date(discharge_day),
            ward_type=ward,
        ),
        insurance=RawInsurance(
            insurer_name=insurer[0], tpa_name=insurer[1],
            policy_number=f"POL-{100000 + index * 7}",
            claim_number=f"CLM/25/{900000 + index * 13}",
            employee_id=f"EMP-{index:04d}",
        ),
        line_items=items,
        totals=RawTotals(
            gross_amount=_rupees(gross), discount="NIL",
            cgst=_rupees(cgst), sgst=_rupees(sgst), net_amount=_rupees(net),
            advance_paid=_rupees(advance), balance_due=_rupees(balance),
            amount_in_words=None,
        ),
    )

    renderer = (_render_corporate, _render_grouped, _render_nursing_home,
                _render_government, _render_diagnostic)[layout]
    text = renderer(truth, rendered_rows, handwritten_note)

    return MiniDoc(
        doc_id=f"mini_{index:02d}",
        text=text,
        truth=truth,
        meta={
            "source": "mini",
            "template": f"L{layout + 1}",
            "pages": "1",
            "scan_quality": "clean",
            "language": "bilingual" if layout == 3 else "english",
            "messiness": ",".join(
                tag for tag, on in (
                    ("missing_uhid", missing_uhid),
                    ("misaligned_totals", misaligned_totals),
                    ("duplicate_row", duplicate_row),
                    ("handwritten", handwritten_note),
                ) if on
            ) or "none",
        },
    )


# --------------------------------------------------------------------------
# renderers - five structurally different layouts
# --------------------------------------------------------------------------

def _totals_block(t: Any, indent: int = 44) -> list[str]:
    pad = " " * indent
    return [
        f"{pad}Gross Amount : {t.gross_amount:>14}",
        f"{pad}Discount     : {t.discount:>14}",
        f"{pad}CGST         : {t.cgst:>14}",
        f"{pad}SGST         : {t.sgst:>14}",
        f"{pad}Net Amount   : {t.net_amount:>14}",
        f"{pad}Advance Paid : {t.advance_paid:>14}",
        f"{pad}Balance Due  : {t.balance_due:>14}",
    ]


def _render_corporate(doc: Any, rows: list[tuple[str, ...]], note: bool) -> str:
    h, p, i = doc.hospital, doc.patient, doc.insurance
    lines = [
        h.name.upper().center(78),
        f"{h.address}, {h.city}, {h.state}".center(78),
        f"GSTIN: {h.gstin}".center(78),
        "",
        "FINAL BILL OF SUPPLY".center(78),
        "",
        f"Patient Name : {p.name:<28} UHID      : {p.uhid or '-'}",
        f"Age / Sex    : {p.age} / {p.sex:<20} IP No     : {p.ip_number}",
        f"Admission    : {p.admission_date:<28} Discharge : {p.discharge_date}",
        f"Ward Type    : {p.ward_type}",
        f"Insurer      : {i.insurer_name:<28} TPA       : {i.tpa_name}",
        f"Policy No    : {i.policy_number:<28} Claim No  : {i.claim_number}",
        f"Employee ID  : {i.employee_id}",
        "",
        "S.No  Description                     Date         Qty   Rate         Amount",
        "-" * 78,
    ]
    for serial, description, date, quantity, rate, amount in rows:
        lines.append(
            f"{serial:<5} {description:<31} {date:<12} {quantity:<5} "
            f"{rate:>10} {amount:>12}"
        )
    lines.append("-" * 78)
    lines.extend(_totals_block(doc.totals))
    if note:
        lines.append("")
        lines.append("   [handwritten] corrected by billing dept - see annexure")
    return "\n".join(lines)


def _render_grouped(doc: Any, rows: list[tuple[str, ...]], note: bool) -> str:
    h, p, i = doc.hospital, doc.patient, doc.insurance
    lines = [
        f"{h.name}",
        f"{h.address} | {h.city} | {h.state}",
        f"GST No {h.gstin}",
        "",
        "=" * 74,
        f"UHID: {p.uhid or 'NOT ISSUED'}   IP No: {p.ip_number}   Ward: {p.ward_type}",
        f"Name: {p.name}   Age: {p.age}   Sex: {p.sex}",
        f"DOA: {p.admission_date}   DOD: {p.discharge_date}",
        f"Insurance: {i.insurer_name} / {i.tpa_name}",
        f"Policy: {i.policy_number}   Claim: {i.claim_number}   Emp: {i.employee_id}",
        "=" * 74,
        "",
        "Sr | Particulars                    | Qty | Rate       | Amount",
    ]
    for serial, description, _date_value, quantity, rate, amount in rows:
        lines.append(f"{serial:>2} | {description:<30} | {quantity:>3} | {rate:>10} | {amount:>12}")
    lines.append("")
    lines.extend(_totals_block(doc.totals, indent=30))
    if note:
        lines.append("*** rate revised manually ***")
    return "\n".join(lines)


def _render_nursing_home(doc: Any, rows: list[tuple[str, ...]], note: bool) -> str:
    h, p, i = doc.hospital, doc.patient, doc.insurance
    lines = [
        h.name,
        h.address,
        f"{h.city} - {h.state}",
        f"GSTIN {h.gstin}",
        "",
        "BILL",
        "",
        f"Patient : {p.name}",
        f"Age     : {p.age}",
        f"Sex     : {p.sex}",
        f"UHID    : {p.uhid or ''}",
        f"IP No   : {p.ip_number}",
        f"Ward    : {p.ward_type}",
        f"Adm     : {p.admission_date}",
        f"Dis     : {p.discharge_date}",
        f"Insurer : {i.insurer_name}",
        f"TPA     : {i.tpa_name}",
        f"Policy  : {i.policy_number}",
        f"Claim   : {i.claim_number}",
        "",
    ]
    for serial, description, _d, quantity, rate, amount in rows:
        lines.append(f"{serial}. {description}  x{quantity}  @ {rate}  = {amount}")
    lines.append("")
    lines.extend(_totals_block(doc.totals, indent=8))
    if note:
        lines.append("(corrected in pen)")
    return "\n".join(lines)


def _render_government(doc: Any, rows: list[tuple[str, ...]], note: bool) -> str:
    """Minimal formatting, bilingual header (English + Telugu)."""
    h, p, i = doc.hospital, doc.patient, doc.insurance
    lines = [
        "ప్రభుత్వ ఆసుపత్రి / GOVERNMENT HOSPITAL",
        h.name,
        f"{h.address}, {h.city}",
        f"GSTIN {h.gstin}",
        "",
        "రోగి వివరాలు / PATIENT DETAILS",
        f"పేరు / Name {p.name}",
        f"వయస్సు / Age {p.age}    లింగం / Sex {p.sex}",
        f"UHID {p.uhid or '--'}    IP No {p.ip_number}",
        f"Ward {p.ward_type}",
        f"Admission {p.admission_date}    Discharge {p.discharge_date}",
        f"Insurer {i.insurer_name}    TPA {i.tpa_name}",
        f"Policy {i.policy_number}    Claim {i.claim_number}",
        "",
        "వివరాలు / DETAILS",
    ]
    for serial, description, _d, quantity, rate, amount in rows:
        lines.append(f"{serial} {description} {quantity} {rate} {amount}")
    lines.append("")
    lines.extend(_totals_block(doc.totals, indent=4))
    if note:
        lines.append("గమనిక / note: manual correction")
    return "\n".join(lines)


def _render_diagnostic(doc: Any, rows: list[tuple[str, ...]], note: bool) -> str:
    h, p, i = doc.hospital, doc.patient, doc.insurance
    lines = [
        f"{h.name:^70}",
        f"{h.address}, {h.city}, {h.state}".center(70),
        f"GSTIN: {h.gstin}".center(70),
        "",
        "TEST PANEL INVOICE".center(70),
        "",
        f"Patient Name : {p.name}",
        f"Age / Sex    : {p.age} / {p.sex}",
        f"UHID         : {p.uhid or 'N/A'}",
        f"IP No        : {p.ip_number}",
        f"Ward Type    : {p.ward_type}",
        f"Admission    : {p.admission_date}",
        f"Discharge    : {p.discharge_date}",
        f"Insurer      : {i.insurer_name}",
        f"TPA          : {i.tpa_name}",
        f"Policy No    : {i.policy_number}",
        f"Claim No     : {i.claim_number}",
        f"Employee ID  : {i.employee_id}",
        "",
        "  #  TEST / PANEL                      QTY      RATE        AMOUNT",
        "  " + "=" * 66,
    ]
    for serial, description, _d, quantity, rate, amount in rows:
        lines.append(f"  {serial:<3}{description:<34}{quantity:<8} {rate:>10} {amount:>12}")
    lines.append("  " + "=" * 66)
    lines.extend(_totals_block(doc.totals, indent=36))
    if note:
        lines.append("  note: amount overwritten by hand")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# build / load
# --------------------------------------------------------------------------

def build_mini_set(n: int = 20, seed: int = SEED) -> list[MiniDoc]:
    """Deterministic: same seed, byte-identical corpus."""
    rng = random.Random(seed)
    return [_build_one(index, rng) for index in range(n)]


def write_mini_set(directory: Path | str = DEFAULT_DIR, n: int = 20) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for doc in build_mini_set(n):
        (out / f"{doc.doc_id}.txt").write_text(doc.text, encoding="utf-8")
        (out / f"{doc.doc_id}.json").write_text(
            json.dumps(
                {"doc_id": doc.doc_id, "meta": doc.meta,
                 "truth": doc.truth.model_dump(mode="json")},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return out


def load_mini_set(directory: Path | str = DEFAULT_DIR) -> list[MiniDoc]:
    """Load from disk, building it first if it is not there."""
    out = Path(directory)
    if not out.exists() or not list(out.glob("*.json")):
        write_mini_set(out)

    docs = []
    for json_path in sorted(out.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        docs.append(
            MiniDoc(
                doc_id=payload["doc_id"],
                text=(out / f"{payload['doc_id']}.txt").read_text(encoding="utf-8"),
                truth=RawDocument.model_validate(payload["truth"]),
                meta=payload["meta"],
            )
        )
    return docs


if __name__ == "__main__":
    path = write_mini_set()
    print(f"wrote {len(list(path.glob('*.txt')))} documents to {path}")

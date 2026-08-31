"""Pure normalization functions for RECKON v2.

Every function here is total and side-effect free: given a string (or None) it
returns a normalized value or None. Nothing raises on bad input - an extractor is
allowed to emit garbage, and garbage must normalize to None rather than blow up
an evaluation run over 10,000 documents.

Section 5 of the brief requires every metric to be computed twice, on raw strings
and on the values produced here. That only means something if this module's
behaviour is written down, so each function documents its contract precisely.

Deliberate policy decisions, stated once so they can be argued with:

* Numeric dates are ALWAYS read day-first. Indian hospital bills are DD/MM.
  A US-style ``05/13/2025`` therefore yields None (month 13 is invalid) rather
  than a silently wrong date. A miss is more honest than a wrong answer.
* Two-digit years pivot at 70: 00-69 -> 2000-2069, 70-99 -> 1970-1999.
* In an amount column, ``NIL`` and a lone dash mean zero, not missing. This is
  universal billing convention. ``N/A`` means missing and yields None.
* Money is Decimal, always, constructed from a cleaned string. Never float.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

from reckon.schema import Category

__all__ = [
    "TWO_DIGIT_YEAR_PIVOT",
    "HONORIFICS",
    "WARD_TYPES",
    "collapse_whitespace",
    "normalize_text",
    "normalize_org_name",
    "normalize_name",
    "normalize_identifier",
    "normalize_date",
    "normalize_amount",
    "normalize_quantity",
    "normalize_age",
    "normalize_sex",
    "normalize_ward_type",
    "normalize_category",
    "normalize_bool",
    "normalize_gstin",
    "gstin_check_digit",
    "is_valid_gstin",
    "normalize_document",
    "NORMALIZERS",
]

TWO_DIGIT_YEAR_PIVOT = 70

# Honorifics stripped from PERSON names only. Never applied to organisation
# names: "Dr. Rao's Nursing Home" must keep its "Dr.".
HONORIFICS: tuple[str, ...] = (
    "baby of",
    "b/o",
    "master",
    "mstr",
    "shri",
    "smt",
    "sri",
    "mrs",
    "miss",
    "mr",
    "ms",
    "dr",
)

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_DASHES = "‐‑‒–—―−"
_ZERO_TOKENS = {"nil", "nill", "none"} | {"-" * n for n in (1, 2, 3)}
_CURRENCY_RE = re.compile(r"(?:₹|rs\.?|inr|r\.s\.?)", re.IGNORECASE)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

def collapse_whitespace(s: str) -> str:
    """Collapse every run of whitespace (including NBSP/newlines) to one space."""
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def _prepare(s: str | None) -> str | None:
    """Shared entry guard: None-safe, NFKC, unify dashes, collapse whitespace."""
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = "".join("-" if ch in _DASHES else ch for ch in s)
    s = collapse_whitespace(s)
    return s or None


def normalize_text(s: str | None) -> str | None:
    """Free text: NFKC, collapse whitespace, casefold. Punctuation preserved."""
    s = _prepare(s)
    return s.casefold() if s else None


def normalize_org_name(s: str | None) -> str | None:
    """Organisation name: like :func:`normalize_text` but trims edge punctuation.

    Honorifics are NOT stripped here - they are part of an organisation's name.
    """
    s = _prepare(s)
    if not s:
        return None
    s = s.strip(" .,:;-_/\\|")
    s = collapse_whitespace(s)
    return s.casefold() or None


def normalize_name(s: str | None) -> str | None:
    """Person name: strip honorifics, casefold, drop initial dots, collapse.

    ``"Dr. Mrs. Kavitha  Rao"`` -> ``"kavitha rao"``
    ``"M. Aadithya Ram"``       -> ``"m aadithya ram"``
    ``"Baby of Smt. Lakshmi"``  -> ``"lakshmi"``

    Honorific stripping repeats while the leading token is an honorific, so
    stacked titles are removed. A dot following a single letter is an initial
    separator and is dropped, because ``M.`` and ``M`` are the same name written
    two ways - that is formatting variance, not a different value.
    """
    s = _prepare(s)
    if not s:
        return None
    s = s.casefold()

    changed = True
    while changed:
        changed = False
        head = s.lstrip(" .,")
        for h in HONORIFICS:
            # honorific must be followed by a separator, so "mr" does not eat
            # the first two letters of "mrudula"
            m = re.match(rf"{re.escape(h)}(?=$|[\s.,])", head)
            if m:
                s = head[m.end():].lstrip(" .,")
                changed = True
                break

    # Initials: the dot is a separator, so it becomes a space rather than being
    # deleted - otherwise "A.B.C. Naidu" would collapse to "abc naidu".
    s = re.sub(r"\b([a-z])\.", r"\1 ", s)
    s = re.sub(r"[.,;:]+", " ", s)
    s = s.strip(" -_/\\|")
    return collapse_whitespace(s) or None


def normalize_identifier(s: str | None) -> str | None:
    """UHID / policy / claim / IP / HSN / serial: uppercase alphanumerics only.

    ``"POL-1234 5678"`` -> ``"POL12345678"``; ``"UH/2025/00123"`` -> ``"UH202500123"``.
    Separators inside an ID are presentation, so they are removed to make raw and
    normalized comparison meaningfully different.
    """
    s = _prepare(s)
    if not s:
        return None
    out = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return out or None


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

_TIME_TAIL_RE = re.compile(
    r"[\s,]+\d{1,2}[:.]\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?\s*$", re.IGNORECASE
)
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


def _expand_year(y: int) -> int:
    if y >= 100:
        return y
    return 1900 + y if y >= TWO_DIGIT_YEAR_PIVOT else 2000 + y


def _build(year: int, month: int, day: int) -> date | None:
    try:
        return date(_expand_year(year), month, day)
    except ValueError:
        return None


def normalize_date(s: str | None) -> date | None:
    """Parse an Indian-convention date string to ``datetime.date``.

    Accepted: ``DD/MM/YYYY``, ``DD-MM-YY``, ``DD.MM.YYYY``, ``DD MM YYYY``,
    ``DD-MMM-YYYY``, ``DD MMM YY``, ``1st Jan 2025``, ``Jan 1, 2025``,
    ``YYYY-MM-DD``. A trailing clock time is discarded.

    Numeric forms are day-first. Anything else, or an impossible calendar date,
    returns None.
    """
    s = _prepare(s)
    if not s:
        return None
    s = _TIME_TAIL_RE.sub("", s).strip()
    s = _ORDINAL_RE.sub(r"\1", s)
    s = s.strip(" .,")
    if not s:
        return None

    # ISO / year-first: 2025-01-05, 2025/01/05
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _build(int(m[1]), int(m[2]), int(m[3]))

    # all-numeric, day-first: 05/01/2025, 5-1-25, 05.01.2025, 05 01 2025
    m = re.fullmatch(r"(\d{1,2})[-/.\s](\d{1,2})[-/.\s](\d{2,4})", s)
    if m:
        return _build(int(m[3]), int(m[2]), int(m[1]))

    # day-month-name-year: 5-Jan-2025, 5 January 25, 5 Jan, 2025
    m = re.fullmatch(r"(\d{1,2})[-/.\s]+([A-Za-z]+)[-/.,\s]+(\d{2,4})", s)
    if m:
        month = _MONTHS.get(m[2].casefold())
        return _build(int(m[3]), month, int(m[1])) if month else None

    # month-name-day-year: Jan 5, 2025
    m = re.fullmatch(r"([A-Za-z]+)[-/.\s]+(\d{1,2})[-/.,\s]+(\d{2,4})", s)
    if m:
        month = _MONTHS.get(m[1].casefold())
        return _build(int(m[3]), month, int(m[2])) if month else None

    return None


# --------------------------------------------------------------------------
# money and numbers
# --------------------------------------------------------------------------

def _to_decimal(s: str | None, *, allow_currency: bool) -> Decimal | None:
    s = _prepare(s)
    if not s:
        return None

    if s.casefold() in _ZERO_TOKENS:
        return Decimal(0)

    negative = False
    body = s

    # Currency is stripped before AND inside the parentheses check, so both
    # "Rs. (2,500.00)" and "(Rs. 2,500.00)" are read as negative.
    if allow_currency:
        body = collapse_whitespace(_CURRENCY_RE.sub(" ", body))
    if body.startswith("(") and body.endswith(")"):
        negative = True
        body = body[1:-1].strip()
        if allow_currency:
            body = collapse_whitespace(_CURRENCY_RE.sub(" ", body))

    body = re.sub(r"/-*\s*$", "", body).strip()   # trailing "/-" or "/"
    body = re.sub(r"-\s*$", "", body).strip()     # trailing "-" used as "/-"
    body = body.strip("=* ")

    if body.startswith("+"):
        body = body[1:].strip()
    if body.startswith("-"):
        negative = True
        body = body[1:].strip()

    if not body or body.casefold() in _ZERO_TOKENS:
        return Decimal(0) if body.casefold() in _ZERO_TOKENS else None

    # Indian digit grouping (1,23,456.00) and OCR space grouping (1 23 456.00).
    body = re.sub(r"(?<=\d)[,\s](?=\d)", "", body)

    if not re.fullmatch(r"\d*\.?\d+|\d+\.", body):
        return None

    try:
        value = Decimal(body if not body.endswith(".") else body[:-1])
    except InvalidOperation:
        return None
    return -value if negative else value


def normalize_amount(s: str | None) -> Decimal | None:
    """Money string -> Decimal.

    Handles ``Rs.``/``INR``/``₹`` prefixes, Indian grouping ``1,23,456.00``,
    a trailing ``/-``, and parentheses for negatives ``(1,234.00)`` -> -1234.00.
    ``NIL`` and a lone dash are zero. Unparseable input returns None.
    """
    return _to_decimal(s, allow_currency=True)


def normalize_quantity(s: str | None) -> Decimal | None:
    """Quantity string -> Decimal. Fractional quantities (``0.5`` day) are real."""
    s = _prepare(s)
    if s:
        s = re.sub(r"\s*(?:x|nos?\.?|units?|qty\.?)\s*$", "", s, flags=re.IGNORECASE)
    return _to_decimal(s, allow_currency=False)


def normalize_age(s: str | None) -> int | None:
    """Age string -> whole years.

    ``"45"``, ``"45 Y"``, ``"45 yrs"``, ``"45Y/M"`` -> 45. An age given only in
    months, weeks or days floors to 0, which is lossy but is what an adjudication
    rule needs; the raw string is always retained for the raw-string metric.
    """
    s = _prepare(s)
    if not s:
        return None
    s = s.casefold()
    m = re.search(r"(\d+)\s*(?:\.\d+)?\s*(years|year|yrs|yr|y)?", s)
    if not m:
        return None
    value, unit = int(m[1]), m[2]
    if unit is None and re.search(r"\b(m|mon|month|months|d|day|days|w|week|weeks)\b", s):
        return 0
    if unit is None and re.match(r"^\d+\s*(m|d|w)\b", s):
        return 0
    return value


def normalize_sex(s: str | None) -> str | None:
    """Sex -> ``male`` | ``female`` | ``other``. Anything ambiguous is None."""
    s = _prepare(s)
    if not s:
        return None
    t = re.sub(r"[^a-z]", "", s.casefold())
    if t in {"m", "male"}:
        return "male"
    if t in {"f", "female"}:
        return "female"
    if t in {"o", "other", "t", "ts", "transgender"}:
        return "other"
    return None


#: Canonical ward vocabulary. Room-rent capping rules key off these, so the set
#: is closed; unrecognised wards fall through to "other" rather than inventing a
#: value that no adjudication rule can match.
WARD_TYPES: tuple[str, ...] = (
    "general", "semi_private", "private", "deluxe", "suite",
    "icu", "iccu", "nicu", "picu", "hdu", "day_care", "emergency", "other",
)

_WARD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"semi[\s_-]*(private|pvt|special)|twin[\s_-]*shar", "semi_private"),
    (r"\bnicu\b|neonatal", "nicu"),
    (r"\bpicu\b|p(a)?ediatric intensive", "picu"),
    (r"\biccu\b|coronary", "iccu"),
    (r"\bicu\b|intensive care|critical care", "icu"),
    (r"\bhdu\b|high dependen", "hdu"),
    (r"day[\s_-]*care|\bdaycare\b", "day_care"),
    (r"emergenc|casualty|\ber\b", "emergency"),
    (r"deluxe|delux", "deluxe"),
    (r"suite", "suite"),
    (r"private|\bpvt\b|single[\s_-]*room|\bsingle\b", "private"),
    (r"general|\bgen\b|\bward\b|shar(ed|ing)|economy", "general"),
)


def normalize_ward_type(s: str | None) -> str | None:
    """Ward description -> canonical ward token, or ``other`` if unrecognised."""
    s = _prepare(s)
    if not s:
        return None
    t = s.casefold()
    if t.replace(" ", "_").replace("-", "_") in WARD_TYPES:
        return t.replace(" ", "_").replace("-", "_")
    for pattern, canon in _WARD_PATTERNS:
        if re.search(pattern, t):
            return canon
    return "other"


_CATEGORY_PATTERNS: tuple[tuple[str, Category], ...] = (
    (r"room|bed\s*charge|accommodation|rent", Category.ROOM_RENT),
    (r"nurs", Category.NURSING),
    (r"pharm|medicine|drug|injection|tablet", Category.PHARMACY),
    (r"consumable|disposable|glove|syringe|cotton", Category.CONSUMABLES),
    (r"radiolog|x[\s-]*ray|\bct\b|\bmri\b|ultraso|sonograph|scan", Category.RADIOLOGY),
    (r"diagnost|lab\b|patholog|test|investigation|biochem", Category.DIAGNOSTICS),
    (r"surg|operat|\bot\b|procedure|anaesth|anesth", Category.SURGERY),
    (r"consult|doctor|physician|professional|visit\s*charge", Category.PROFESSIONAL_FEES),
    (r"equipment|ventilat|monitor|oxygen|pump", Category.EQUIPMENT),
    (r"admin|registration|admission\s*(fee|charge)|record|file\s*charge",
     Category.ADMINISTRATIVE),
    (r"non[\s_-]*medical|toiletr|attendant|food|telephone|tv\b", Category.NON_MEDICAL),
)


def normalize_category(s: str | None) -> Category | None:
    """Category label -> :class:`Category`. Exact enum value wins over patterns."""
    s = _prepare(s)
    if not s:
        return None
    t = re.sub(r"[\s-]+", "_", s.casefold())
    try:
        return Category(t)
    except ValueError:
        pass
    for pattern, cat in _CATEGORY_PATTERNS:
        if re.search(pattern, s.casefold()):
            return cat
    return Category.OTHER


_TRUE_TOKENS = {"y", "yes", "true", "1", "payable", "admissible", "allowed"}
_FALSE_TOKENS = {
    "n", "no", "false", "0", "nonpayable", "non_payable", "notpayable",
    "deducted", "disallowed", "inadmissible",
}


def normalize_bool(s: str | None) -> bool | None:
    """Payability flag -> bool. Unrecognised input is None, never a guess."""
    s = _prepare(s)
    if not s:
        return None
    t = re.sub(r"[\s-]+", "_", s.casefold()).strip("_")
    if t in _TRUE_TOKENS:
        return True
    if t in _FALSE_TOKENS:
        return False
    if t.replace("_", "") in _FALSE_TOKENS:
        return False
    return None


# --------------------------------------------------------------------------
# GSTIN
# --------------------------------------------------------------------------

_GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_check_digit(first_fourteen: str) -> str | None:
    """15th GSTIN character for the first 14, via the mod-36 scheme GSTN uses."""
    body = first_fourteen.upper()
    if len(body) != 14 or any(c not in _GSTIN_ALPHABET for c in body):
        return None
    total = 0
    factor = 2
    for ch in reversed(body):
        addend = factor * _GSTIN_ALPHABET.index(ch)
        factor = 1 if factor == 2 else 2
        total += addend // 36 + addend % 36
    return _GSTIN_ALPHABET[(36 - total % 36) % 36]


def is_valid_gstin(s: str | None) -> bool:
    """True when *s* is a structurally valid GSTIN with a correct check digit."""
    t = normalize_gstin(s)
    if t is None or len(t) != 15:
        return False
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", t):
        return False
    return gstin_check_digit(t[:14]) == t[14]


def normalize_gstin(s: str | None) -> str | None:
    """GSTIN -> uppercase alphanumerics. Validity is NOT enforced here.

    A wrong GSTIN must stay representable so it can be scored as wrong;
    :func:`is_valid_gstin` is the separate check used by generators and by
    data-quality slices.
    """
    return normalize_identifier(s)


# --------------------------------------------------------------------------
# document-level
# --------------------------------------------------------------------------

#: Field path -> normalizer. Single source of truth shared by the pipeline and
#: by eval, so a metric can never be computed with a different rule than the one
#: the pipeline applied.
NORMALIZERS = {
    "hospital.name": normalize_org_name,
    "hospital.address": normalize_text,
    "hospital.city": normalize_org_name,
    "hospital.state": normalize_org_name,
    "hospital.gstin": normalize_gstin,
    "hospital.hospital_type": normalize_org_name,
    "patient.name": normalize_name,
    "patient.age": normalize_age,
    "patient.sex": normalize_sex,
    "patient.uhid": normalize_identifier,
    "patient.ip_number": normalize_identifier,
    "patient.admission_date": normalize_date,
    "patient.discharge_date": normalize_date,
    "patient.ward_type": normalize_ward_type,
    "insurance.insurer_name": normalize_org_name,
    "insurance.tpa_name": normalize_org_name,
    "insurance.policy_number": normalize_identifier,
    "insurance.claim_number": normalize_identifier,
    "insurance.employee_id": normalize_identifier,
    "totals.gross_amount": normalize_amount,
    "totals.discount": normalize_amount,
    "totals.cgst": normalize_amount,
    "totals.sgst": normalize_amount,
    "totals.net_amount": normalize_amount,
    "totals.advance_paid": normalize_amount,
    "totals.balance_due": normalize_amount,
    "totals.amount_in_words": normalize_text,
}

LINE_ITEM_NORMALIZERS = {
    "serial_no": normalize_identifier,
    "description": normalize_text,
    "service_date": normalize_date,
    "category": normalize_category,
    "quantity": normalize_quantity,
    "unit_rate": normalize_amount,
    "amount": normalize_amount,
    "hsn_code": normalize_identifier,
    "is_payable": normalize_bool,
    "deduction_reason": normalize_text,
}


def normalize_document(raw):  # type: ignore[no-untyped-def]
    """``RawDocument`` -> ``Document``, applying :data:`NORMALIZERS` field by field.

    Imported lazily inside the function body to keep this module importable with
    no cost, and to keep the schema the only module that owns model classes.
    """
    from reckon.schema import (
        Document, Hospital, Insurance, LineItem, Patient, Totals,
    )

    blocks = {"hospital": Hospital, "patient": Patient,
              "insurance": Insurance, "totals": Totals}
    built = {}
    for block_name, model in blocks.items():
        raw_block = getattr(raw, block_name)
        values = {}
        for fname in model.model_fields:
            fn = NORMALIZERS[f"{block_name}.{fname}"]
            values[fname] = fn(getattr(raw_block, fname))
        built[block_name] = model(**values)

    items = []
    for raw_item in raw.line_items:
        values = {
            fname: fn(getattr(raw_item, fname))
            for fname, fn in LINE_ITEM_NORMALIZERS.items()
        }
        items.append(LineItem(**values))

    return Document(line_items=items, **built)

"""RECKON v2 document schema. THE single source of truth.

Synthetic generation, training targets, evaluation, adjudication and the API all
import from this module. Nothing else defines the shape of a document.

Two parallel model families are defined:

* ``Raw*``  - every leaf is ``str | None``. This is verbatim what an extractor
  (Donut decoder, OCR+heuristics, LiLT) emits, before any interpretation.
* typed     - ``Decimal`` for money, ``date`` for dates, enums for closed sets.
  Produced from a ``Raw*`` by ``reckon.normalize.normalize_document``.

The split exists because Section 5 of the project brief requires every metric to
be computed twice - once on raw strings, once on normalized values. The gap
between the two is the share of error that is mere formatting.

Design rule: these models are PERMISSIVE. An extractor is allowed to be wrong,
and a wrong prediction must still be representable, or it cannot be scored.
Business invariants (net == gross - discount + taxes, and so on) are checked in
the reconciliation and adjudication layers, never here. The one thing the schema
does refuse is a ``float`` in a money field.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict

__all__ = [
    "Category",
    "Document",
    "Hospital",
    "Insurance",
    "LineItem",
    "Patient",
    "RawDocument",
    "RawHospital",
    "RawInsurance",
    "RawLineItem",
    "RawPatient",
    "RawTotals",
    "Totals",
    "Money",
    "Quantity",
    "FIELD_PATHS",
    "LINE_ITEM_FIELDS",
    "MONEY_FIELD_PATHS",
    "DATE_FIELD_PATHS",
    "donut_special_tokens",
]


# --------------------------------------------------------------------------
# money
# --------------------------------------------------------------------------

def _reject_float(v: Any) -> Any:
    """Refuse ``float`` for monetary and quantity values.

    Section 1 of the brief: never use floats for money, anywhere. Pydantic would
    otherwise coerce 0.1 into a Decimal carrying the binary-float error, and that
    error would be invisible for the rest of the pipeline. The guard lives at the
    schema boundary because that is the one place it cannot be forgotten.
    """
    if isinstance(v, float):
        raise ValueError(
            "float is forbidden for money/quantity; pass str, int or Decimal"
        )
    return v


Money = Annotated[Decimal, BeforeValidator(_reject_float)]
Quantity = Annotated[Decimal, BeforeValidator(_reject_float)]


class Category(str, Enum):
    """Line-item category. Closed set, fixed by the brief."""

    ROOM_RENT = "room_rent"
    NURSING = "nursing"
    CONSUMABLES = "consumables"
    PHARMACY = "pharmacy"
    DIAGNOSTICS = "diagnostics"
    RADIOLOGY = "radiology"
    SURGERY = "surgery"
    PROFESSIONAL_FEES = "professional_fees"
    EQUIPMENT = "equipment"
    ADMINISTRATIVE = "administrative"
    NON_MEDICAL = "non_medical"
    OTHER = "other"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# typed (normalized) models
# --------------------------------------------------------------------------

class Hospital(_Base):
    name: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    gstin: str | None = None
    hospital_type: str | None = None


class Patient(_Base):
    name: str | None = None
    age: int | None = None          # whole years; sub-year ages floor to 0
    sex: str | None = None          # canonical: "male" | "female" | "other"
    uhid: str | None = None
    ip_number: str | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    ward_type: str | None = None    # canonical vocabulary in normalize.WARD_TYPES


class Insurance(_Base):
    insurer_name: str | None = None
    tpa_name: str | None = None
    policy_number: str | None = None
    claim_number: str | None = None
    employee_id: str | None = None


class LineItem(_Base):
    serial_no: str | None = None    # str: real bills use "1", "1a", "12)", "-"
    description: str | None = None
    service_date: date | None = None
    category: Category | None = None
    quantity: Quantity | None = None
    unit_rate: Money | None = None
    amount: Money | None = None
    hsn_code: str | None = None
    is_payable: bool | None = None
    deduction_reason: str | None = None


class Totals(_Base):
    gross_amount: Money | None = None
    discount: Money | None = None
    cgst: Money | None = None
    sgst: Money | None = None
    net_amount: Money | None = None
    advance_paid: Money | None = None
    balance_due: Money | None = None
    amount_in_words: str | None = None


class Document(_Base):
    hospital: Hospital = Hospital()
    patient: Patient = Patient()
    insurance: Insurance = Insurance()
    line_items: list[LineItem] = []
    totals: Totals = Totals()


# --------------------------------------------------------------------------
# raw (verbatim string) mirrors
# --------------------------------------------------------------------------

def _field_names(model: type[BaseModel]) -> tuple[str, ...]:
    return tuple(model.model_fields)


def _raw_mirror(name: str, source: type[BaseModel]) -> Any:
    """Build the ``str | None`` mirror of a typed block.

    Generated rather than hand-written so the raw and typed families cannot drift
    apart - adding a field in one place adds it in both.
    """
    fields = _field_names(source)
    ns: dict[str, Any] = {
        "__annotations__": {f: (str | None) for f in fields},
        "__doc__": f"Verbatim string mirror of {source.__name__}.",
        "__module__": __name__,
    }
    for f in fields:
        ns[f] = None
    return type(name, (_Base,), ns)


RawHospital: Any = _raw_mirror("RawHospital", Hospital)
RawPatient: Any = _raw_mirror("RawPatient", Patient)
RawInsurance: Any = _raw_mirror("RawInsurance", Insurance)
RawLineItem: Any = _raw_mirror("RawLineItem", LineItem)
RawTotals: Any = _raw_mirror("RawTotals", Totals)


class RawDocument(_Base):
    """Verbatim extractor output, before normalization."""

    hospital: RawHospital = RawHospital()
    patient: RawPatient = RawPatient()
    insurance: RawInsurance = RawInsurance()
    line_items: list[RawLineItem] = []
    totals: RawTotals = RawTotals()


# --------------------------------------------------------------------------
# field registries - eval iterates these, never a hand-maintained list
# --------------------------------------------------------------------------

_BLOCKS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("hospital", Hospital),
    ("patient", Patient),
    ("insurance", Insurance),
    ("totals", Totals),
)

#: Dotted paths of every scalar (non line-item) field, e.g. "totals.net_amount".
FIELD_PATHS: tuple[str, ...] = tuple(
    f"{block}.{f}" for block, model in _BLOCKS for f in _field_names(model)
)

#: Attribute names inside a single line item.
LINE_ITEM_FIELDS: tuple[str, ...] = _field_names(LineItem)


def _paths_with_type(marker: str) -> tuple[str, ...]:
    out: list[str] = []
    for block, model in _BLOCKS:
        for fname, info in model.model_fields.items():
            if marker in str(info.annotation):
                out.append(f"{block}.{fname}")
    return tuple(out)


#: Scalar fields carrying money. Compared with Decimal semantics, never floats.
MONEY_FIELD_PATHS: tuple[str, ...] = _paths_with_type("Decimal")

#: Scalar fields carrying dates. Compared as ISO-8601 after normalization.
DATE_FIELD_PATHS: tuple[str, ...] = _paths_with_type("datetime.date")


# --------------------------------------------------------------------------
# Donut tokenizer support
# --------------------------------------------------------------------------

def donut_special_tokens() -> list[str]:
    """Schema keys as Donut-style special tokens.

    Section 2.1: schema keys are added to the tokenizer before training. Left
    untokenized, a key like ``professional_fees`` costs several decoder steps,
    which both wastes the (already tight) sequence budget and slows convergence.
    Derived from the models so the token set cannot go stale.

    Two kinds of token are returned:

    * ``<s_key>`` / ``</s_key>`` pairs for every schema key.
    * bare tokens for closed-vocabulary VALUES. ``Category`` values are emitted
      verbatim into the target string, and since the set is closed and known in
      advance each one is worth exactly one decoder step. Values are not wrapped
      in key-style tags, because they are not keys.
    """
    keys: list[str] = ["hospital", "patient", "insurance", "totals", "line_items"]
    for _, model in _BLOCKS:
        keys.extend(_field_names(model))
    keys.extend(LINE_ITEM_FIELDS)

    ordered: list[str] = []
    for k in keys:
        if k not in ordered:
            ordered.append(k)
    tokens = [t for k in ordered for t in (f"<s_{k}>", f"</s_{k}>")]
    tokens.extend(c.value for c in Category if c.value not in ordered)
    tokens.append("<sep/>")  # separates repeated line_items in the target string
    return tokens

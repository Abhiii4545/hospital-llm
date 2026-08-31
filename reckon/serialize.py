"""Donut target-string serialisation, and its inverse.

Donut is an image-to-text model: the target is a flat token string, not JSON. The
schema key ``<s_amount>`` is one token because it was added to the tokenizer, so
this format is far cheaper in decoder steps than raw JSON with its braces, quotes
and colons - which matters enormously when the sequence budget is the binding
constraint on line-item recall.

Format::

    <s_line_items><s_serial_no>1</s_serial_no><s_description>Room Rent
    </s_description><s_amount>3,500.00</s_amount></s_line_items><sep/>...

Lives at the top level, beside ``schema`` and ``normalize``, because BOTH
training and serving need it and the layer contracts forbid serving from
importing training. A shared need between two layers that may not see each other
is a leaf.

Round-tripping is the property that matters: ``parse(render(x)) == x`` for every
document the corpus can produce, or training targets and inference outputs are
describing different things. Values are emitted verbatim - the model learns to
read what is printed, and normalization happens afterwards.
"""

from __future__ import annotations

import re
from typing import Any

from reckon.schema import (
    LINE_ITEM_FIELDS,
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = [
    "SEP",
    "render_head_a",
    "render_head_b",
    "parse_head_a",
    "parse_head_b",
    "render_block",
    "parse_block",
    "escape_value",
    "unescape_value",
]

SEP = "<sep/>"

_BLOCKS = {
    "hospital": RawHospital,
    "patient": RawPatient,
    "insurance": RawInsurance,
    "totals": RawTotals,
}

#: A value containing a literal "<" would break parsing, and OCR of a real bill
#: does produce stray angle brackets. Escaped rather than dropped, so the model
#: is still trained to reproduce what is on the page.
_ESCAPES = ((("&"), "&amp;"), ("<", "&lt;"), (">", "&gt;"))


def escape_value(value: str) -> str:
    for raw, encoded in _ESCAPES:
        value = value.replace(raw, encoded)
    return value


def unescape_value(value: str) -> str:
    for raw, encoded in reversed(_ESCAPES):
        value = value.replace(encoded, raw)
    return value


def _tag(key: str, value: str) -> str:
    return f"<s_{key}>{escape_value(value)}</s_{key}>"


def render_block(name: str, model: Any) -> str:
    """One header block. Absent fields are OMITTED, not emitted as empty.

    Emitting ``<s_uhid></s_uhid>`` for a missing UHID would teach the model that
    the token pair is always present and only its content varies, which makes
    "this field is not on the page" much harder to express. Omission is the
    signal.
    """
    if model is None:
        return ""
    parts = [
        _tag(field, getattr(model, field))
        for field in type(model).model_fields
        if getattr(model, field) not in (None, "")
    ]
    return f"<s_{name}>{''.join(parts)}</s_{name}>" if parts else ""


def parse_block(text: str, name: str) -> dict[str, str]:
    """Extract one block's key/value pairs. Tolerant of a truncated sequence."""
    match = re.search(rf"<s_{name}>(.*?)</s_{name}>", text, re.DOTALL)
    if not match:
        # A decoder that ran out of budget leaves an unclosed block. Salvage it
        # rather than returning nothing: a partial header still has value.
        match = re.search(rf"<s_{name}>(.*)", text, re.DOTALL)
        if not match:
            return {}
    body = match.group(1)
    out: dict[str, str] = {}
    for field in _ALL_FIELDS:
        found = re.search(rf"<s_{field}>(.*?)</s_{field}>", body, re.DOTALL)
        if found:
            out[field] = unescape_value(found.group(1)).strip()
    return out


_ALL_FIELDS = sorted(
    {f for model in _BLOCKS.values() for f in model.model_fields}
    | set(LINE_ITEM_FIELDS),
    key=len,
    reverse=True,
)


# --------------------------------------------------------------------------
# Head A - header and totals
# --------------------------------------------------------------------------

def render_head_a(
    document: RawDocument,
    *,
    include_header: bool = True,
    include_totals: bool = True,
) -> str:
    """Target string for Head A on one page.

    Which blocks appear depends on what is actually printed on that page: a
    continuation page carries neither, and the totals page carries only totals.
    """
    parts = []
    if include_header:
        parts += [render_block(name, getattr(document, name))
                  for name in ("hospital", "patient", "insurance")]
    if include_totals:
        parts.append(render_block("totals", document.totals))
    return "".join(p for p in parts if p)


def parse_head_a(text: str) -> RawDocument:
    """Inverse of :func:`render_head_a`. Unknown keys are ignored, not fatal."""
    document = RawDocument()
    for name, model in _BLOCKS.items():
        fields = parse_block(text, name)
        allowed = {k: v for k, v in fields.items() if k in model.model_fields}
        if allowed:
            setattr(document, name, model(**allowed))
    return document


# --------------------------------------------------------------------------
# Head B - line items for one page
# --------------------------------------------------------------------------

def render_head_b(items: list[RawLineItem]) -> str:
    """Target string for Head B: this page's rows, separated by ``<sep/>``."""
    rendered = []
    for item in items:
        parts = [
            _tag(field, getattr(item, field))
            for field in LINE_ITEM_FIELDS
            if getattr(item, field) not in (None, "")
        ]
        if parts:
            rendered.append("".join(parts))
    if not rendered:
        return "<s_line_items></s_line_items>"
    return "<s_line_items>" + SEP.join(rendered) + "</s_line_items>"


def parse_head_b(text: str) -> list[RawLineItem]:
    """Inverse of :func:`render_head_b`.

    A truncated final row is kept if it has any usable field. Dropping it would
    mean a decoder that ran one token short loses a whole line item, which is the
    exact failure mode the two-head split exists to avoid.
    """
    match = re.search(r"<s_line_items>(.*?)</s_line_items>", text, re.DOTALL)
    body = match.group(1) if match else None
    if body is None:
        opened = re.search(r"<s_line_items>(.*)", text, re.DOTALL)
        if not opened:
            return []
        body = opened.group(1)

    items: list[RawLineItem] = []
    for chunk in body.split(SEP):
        if not chunk.strip():
            continue
        values: dict[str, str] = {}
        for field in LINE_ITEM_FIELDS:
            found = re.search(rf"<s_{field}>(.*?)</s_{field}>", chunk, re.DOTALL)
            if found:
                values[field] = unescape_value(found.group(1)).strip()
        if values:
            items.append(RawLineItem(**values))
    return items

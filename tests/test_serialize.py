"""Donut target-string serialisation.

The property that matters is round-tripping: if `parse(render(x)) != x` then the
training target and the inference output describe different things, and every
downstream number is measuring the wrong thing.
"""

from __future__ import annotations

import random

import pytest

from reckon.data.generators.document import generate_document
from reckon.data.layouts import LAYOUTS
from reckon.schema import (
    LINE_ITEM_FIELDS,
    RawDocument,
    RawHospital,
    RawLineItem,
    RawPatient,
    RawTotals,
    donut_special_tokens,
)
from reckon.serialize import (
    SEP,
    escape_value,
    parse_head_a,
    parse_head_b,
    render_head_a,
    render_head_b,
    unescape_value,
)


def test_head_a_round_trips() -> None:
    document = RawDocument(
        hospital=RawHospital(name="Sunrise Hospital", city="Hyderabad"),
        patient=RawPatient(name="Mr. Ramesh Kumar", uhid="UH123", age="45"),
        totals=RawTotals(gross_amount="Rs. 1,23,456.00", net_amount="1,25,678.20"),
    )
    restored = parse_head_a(render_head_a(document))
    assert restored.hospital.name == "Sunrise Hospital"
    assert restored.patient.uhid == "UH123"
    assert restored.totals.net_amount == "1,25,678.20"


def test_head_b_round_trips() -> None:
    items = [
        RawLineItem(serial_no="1", description="Room Rent", quantity="3",
                    unit_rate="3,500.00", amount="10,500.00", is_payable="Y"),
        RawLineItem(serial_no="2", description="Attendant Charges",
                    amount="500.00", is_payable="N"),
    ]
    restored = parse_head_b(render_head_b(items))
    assert len(restored) == 2
    assert restored[0].description == "Room Rent"
    assert restored[0].amount == "10,500.00"
    assert restored[1].is_payable == "N"


def test_absent_fields_are_omitted_not_emitted_empty() -> None:
    """Omission is how the model says 'this field is not on the page'."""
    rendered = render_head_a(RawDocument(patient=RawPatient(name="X")))
    assert "<s_uhid>" not in rendered
    assert "<s_name>X</s_name>" in rendered
    assert parse_head_a(rendered).patient.uhid is None


def test_empty_line_items_render_and_parse() -> None:
    rendered = render_head_b([])
    assert parse_head_b(rendered) == []


def test_separator_divides_rows() -> None:
    rendered = render_head_b([
        RawLineItem(description="A", amount="1"),
        RawLineItem(description="B", amount="2"),
    ])
    assert rendered.count(SEP) == 1


def test_angle_brackets_in_a_value_are_escaped() -> None:
    """OCR of a real bill does produce stray angle brackets."""
    item = RawLineItem(description="Rate <500 special", amount="100")
    rendered = render_head_b([item])
    assert "<500" not in rendered
    restored = parse_head_b(rendered)
    assert restored[0].description == "Rate <500 special"


def test_escape_round_trips() -> None:
    for value in ("a & b", "<x>", "plain", "&amp; already", "a<b>c&d"):
        assert unescape_value(escape_value(value)) == value


def test_truncated_line_items_salvage_the_partial_row() -> None:
    """A decoder one token short must not lose a whole line item."""
    full = render_head_b([
        RawLineItem(description="Room Rent", amount="3500"),
        RawLineItem(description="Nursing", amount="800"),
    ])
    truncated = full[: full.rindex("</s_amount>") + len("</s_amount>")]
    restored = parse_head_b(truncated)
    assert len(restored) == 2
    assert restored[1].description == "Nursing"


def test_truncated_header_block_is_salvaged() -> None:
    rendered = render_head_a(RawDocument(patient=RawPatient(name="Ramesh", uhid="UH1")))
    truncated = rendered[: rendered.index("</s_patient>")]
    assert parse_head_a(truncated).patient.name == "Ramesh"


def test_page_specific_rendering() -> None:
    """A continuation page carries neither header nor totals."""
    document = RawDocument(
        hospital=RawHospital(name="H"),
        totals=RawTotals(net_amount="100"),
    )
    assert render_head_a(document, include_header=False, include_totals=False) == ""
    totals_only = render_head_a(document, include_header=False, include_totals=True)
    assert "<s_totals>" in totals_only and "<s_hospital>" not in totals_only


def test_every_tag_used_is_a_registered_special_token() -> None:
    """A tag the tokenizer does not know costs several decoder steps each time."""
    tokens = set(donut_special_tokens())
    document = RawDocument(
        hospital=RawHospital(name="H", gstin="G"),
        patient=RawPatient(name="P", ward_type="W"),
        totals=RawTotals(net_amount="1"),
    )
    text = render_head_a(document) + render_head_b(
        [RawLineItem(**{f: "x" for f in LINE_ITEM_FIELDS})]
    )
    import re

    for tag in set(re.findall(r"</?s_[a-z_]+>", text)):
        assert tag in tokens, tag
    assert SEP in tokens


@pytest.mark.parametrize("index", range(12))
def test_round_trip_over_real_corpus_documents(index: int) -> None:
    """The check that counts: every document the corpus can produce."""
    spec = LAYOUTS[index % len(LAYOUTS)]
    doc = generate_document(random.Random(f"s:{index}"), f"s{index}", spec.id,
                            spec.rows_per_page, hospital_type=spec.hospital_type,
                            bilingual=spec.bilingual)

    for page in doc.pages:
        rendered_b = render_head_b(page.items)
        assert parse_head_b(rendered_b) == page.items

        rendered_a = render_head_a(
            doc.truth,
            include_header=page.show_header,
            include_totals=page.show_totals,
        )
        restored = parse_head_a(rendered_a)
        if page.show_header:
            assert restored.patient.name == doc.truth.patient.name
            assert restored.hospital.gstin == doc.truth.hospital.gstin
        if page.show_totals:
            assert restored.totals.net_amount == doc.truth.totals.net_amount


def test_bilingual_values_survive_serialisation() -> None:
    """Telugu and Devanagari must not be mangled by the tag format."""
    item = RawLineItem(description="వివరాలు / Room Rent", amount="100")
    assert parse_head_b(render_head_b([item]))[0].description == "వివరాలు / Room Rent"


def _approx_tokens(text: str, specials: set[str]) -> int:
    """Rough decoder-step count: a registered special token costs exactly one.

    Everything else is approximated at ~4 characters per BPE token, applied
    identically to both formats so the comparison is fair.
    """
    count = 0
    for token in sorted(specials, key=len, reverse=True):
        occurrences = text.count(token)
        count += occurrences
        text = text.replace(token, "")
    return count + len(text) // 4


def test_target_costs_fewer_decoder_steps_than_json() -> None:
    """The whole reason for this format: decoder steps are the binding budget.

    Note the format is LONGER in characters than JSON. It only wins once the
    schema keys are registered as single tokens - which is why
    `donut_special_tokens()` must actually be added to the tokenizer before
    training. Without that step this encoding would be strictly worse than JSON,
    and this test is what pins that dependency down.
    """
    import json

    items = [RawLineItem(serial_no=str(i), description=f"Item {i}",
                         quantity="1", unit_rate="100.00", amount="100.00")
             for i in range(20)]
    specials = set(donut_special_tokens())
    donut = render_head_b(items)
    as_json = json.dumps([i.model_dump(exclude_none=True) for i in items])

    assert len(donut) > len(as_json), "expected more characters, fewer tokens"
    assert _approx_tokens(donut, specials) < _approx_tokens(as_json, set())

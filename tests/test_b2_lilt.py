"""B2 LiLT: label scheme, distant supervision and span decoding.

The pure-Python half is tested here. Training needs torch, transformers and an
OCR pass over the corpus, none of which exist yet - but the label scheme is where
a token-classification baseline usually goes quietly wrong, so it is checked
first.
"""

from __future__ import annotations

import pytest

from reckon.models.baselines.b2_lilt import (
    LABELS,
    MODEL_NAME,
    B2LiltBaseline,
    OcrToken,
    assign_bio_labels,
    decode_spans,
    group_rows_by_band,
    label_id,
    spans_to_document,
)
from reckon.schema import FIELD_PATHS, LINE_ITEM_FIELDS


def tokens(*pairs: tuple[str, int]) -> list[OcrToken]:
    """(text, y) -> tokens laid out left to right on the given row."""
    out = []
    x = 0
    for text, y in pairs:
        out.append(OcrToken(text=text, box=(x, y, x + 40, y + 10)))
        x += 50
    return out


# --------------------------------------------------------------------------
# label scheme
# --------------------------------------------------------------------------

def test_label_scheme_covers_every_field_and_line_item_attribute() -> None:
    for path in FIELD_PATHS:
        assert f"B-{path}" in LABELS and f"I-{path}" in LABELS
    for name in LINE_ITEM_FIELDS:
        assert f"B-item.{name}" in LABELS


def test_outside_is_label_zero() -> None:
    """Majority class at id 0 keeps a confusion matrix readable."""
    assert LABELS[0] == "O"
    assert label_id("O") == 0


def test_labels_are_unique() -> None:
    assert len(set(LABELS)) == len(LABELS)


def test_item_labels_cannot_collide_with_header_labels() -> None:
    """`description` exists only on line items, but `name` could collide."""
    assert "B-item.serial_no" in LABELS
    assert "B-patient.name" in LABELS
    assert "B-name" not in LABELS


def test_model_is_the_mit_licensed_one() -> None:
    assert MODEL_NAME == "SCUT-DLVCLab/lilt-roberta-en-base"
    assert "layoutlm" not in MODEL_NAME.casefold()   # CC-BY-NC-SA, forbidden


# --------------------------------------------------------------------------
# distant supervision
# --------------------------------------------------------------------------

def test_a_field_value_is_tagged_bio() -> None:
    page = tokens(("Patient", 10), ("Name", 10), ("Ramesh", 10), ("Kumar", 10))
    labels = assign_bio_labels(page, {"patient.name": "Ramesh Kumar"})
    assert labels == ["O", "O", "B-patient.name", "I-patient.name"]


def test_longer_values_claim_their_tokens_first() -> None:
    """A short value must not steal tokens from a longer, more specific one."""
    page = tokens(("Ramesh", 10), ("Kumar", 10))
    labels = assign_bio_labels(page, {
        "patient.name": "Ramesh Kumar",
        "hospital.city": "Ramesh",
    })
    assert labels[0] == "B-patient.name"
    assert labels[1] == "I-patient.name"


def test_a_value_ocr_never_produced_is_simply_unlabelled() -> None:
    """The recall ceiling of distant supervision, asserted rather than assumed."""
    page = tokens(("Total", 10), ("999", 10))
    labels = assign_bio_labels(page, {"totals.net_amount": "1,23,456.00"})
    assert set(labels) == {"O"}


def test_matching_ignores_case_and_trailing_punctuation() -> None:
    page = tokens(("RAMESH,", 10), ("kumar.", 10))
    labels = assign_bio_labels(page, {"patient.name": "Ramesh Kumar"})
    assert labels[0].startswith("B-")


def test_tokens_are_not_claimed_twice() -> None:
    page = tokens(("1", 10), ("Room", 10), ("Rent", 10), ("1", 10))
    labels = assign_bio_labels(page, {
        "item.serial_no": "1",
        "item.description": "Room Rent",
        "item.quantity": "1",
    })
    assert labels.count("B-item.serial_no") == 1
    assert labels.count("B-item.quantity") == 1


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def test_spans_decode_back_to_text() -> None:
    page = tokens(("Ramesh", 10), ("Kumar", 10), ("UH123", 10))
    labels = ["B-patient.name", "I-patient.name", "B-patient.uhid"]
    spans = decode_spans(page, labels)
    assert spans["patient.name"][0][0] == "Ramesh Kumar"
    assert spans["patient.uhid"][0][0] == "UH123"


def test_orphan_inside_tag_starts_a_span_rather_than_being_dropped() -> None:
    """Real model output produces this; discarding it loses a correct value."""
    page = tokens(("Kumar", 10),)
    spans = decode_spans(page, ["I-patient.name"])
    assert spans["patient.name"][0][0] == "Kumar"


def test_round_trip_label_then_decode() -> None:
    page = tokens(("Name", 10), ("Ramesh", 10), ("Kumar", 10),
                  ("UHID", 10), ("UH123", 10))
    values = {"patient.name": "Ramesh Kumar", "patient.uhid": "UH123"}
    spans = decode_spans(page, assign_bio_labels(page, values))
    assert spans["patient.name"][0][0] == "Ramesh Kumar"
    assert spans["patient.uhid"][0][0] == "UH123"


# --------------------------------------------------------------------------
# row grouping - the documented weakness
# --------------------------------------------------------------------------

def test_line_items_group_into_rows_by_vertical_band() -> None:
    page = [
        OcrToken("Room", (0, 100, 40, 110)), OcrToken("3500", (200, 100, 240, 110)),
        OcrToken("Nursing", (0, 140, 40, 150)), OcrToken("800", (200, 140, 240, 150)),
    ]
    labels = ["B-item.description", "B-item.amount",
              "B-item.description", "B-item.amount"]
    rows = group_rows_by_band(decode_spans(page, labels))
    assert len(rows) == 2
    assert rows[0] == {"description": "Room", "amount": "3500"}
    assert rows[1] == {"description": "Nursing", "amount": "800"}


def test_a_repeated_attribute_in_one_band_splits_the_row() -> None:
    page = [
        OcrToken("Room", (0, 100, 40, 110)),
        OcrToken("Nursing", (300, 102, 360, 112)),
    ]
    rows = group_rows_by_band(
        decode_spans(page, ["B-item.description", "B-item.description"])
    )
    assert len(rows) == 2


def test_row_grouping_is_geometric_and_documented_as_a_weakness() -> None:
    """The brief asks for weaknesses to be stated, not buried."""
    from pathlib import Path

    source = Path("reckon/models/baselines/b2_lilt.py").read_text(encoding="utf-8")
    assert "honest weakness" in source
    assert "Donut has no equivalent failure" in source


def test_spans_assemble_into_a_document() -> None:
    page = [
        OcrToken("Ramesh", (0, 10, 40, 20)),
        OcrToken("Room", (0, 100, 40, 110)),
        OcrToken("3500", (200, 100, 240, 110)),
    ]
    labels = ["B-patient.name", "B-item.description", "B-item.amount"]
    document = spans_to_document(decode_spans(page, labels))
    assert document.patient.name == "Ramesh"
    assert len(document.line_items) == 1
    assert document.line_items[0].amount == "3500"


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------

def test_b2_reports_its_absence_clearly() -> None:
    baseline = B2LiltBaseline()
    assert not baseline.available
    with pytest.raises(NotImplementedError, match="has to be beaten"):
        baseline.extract("anything")

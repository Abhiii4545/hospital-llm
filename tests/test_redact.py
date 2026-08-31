"""Redaction: geometry preservation, one-wayness, and the key location rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reckon.data.redact import (
    PII_FIELDS,
    RedactionKey,
    RedactionManifest,
    redact_document,
    redact_text,
    redact_value,
    surrogate,
)
from reckon.schema import RawDocument, RawHospital, RawInsurance, RawLineItem, RawPatient


@pytest.fixture
def key(tmp_path: Path) -> RedactionKey:
    return RedactionKey.load_or_create(tmp_path / "k" / "redaction.key")


def _document() -> RawDocument:
    return RawDocument(
        hospital=RawHospital(name="Sunrise Hospital", city="Hyderabad",
                             gstin="36AABCS1429B1Z6"),   # pii-allow
        patient=RawPatient(name="Mr. Ramesh Kumar", uhid="UH253950",
                           ip_number="IP4521", age="45", sex="M",
                           ward_type="Semi-Private"),
        insurance=RawInsurance(insurer_name="Star Health",
                               policy_number="P/123456789012",
                               claim_number="CLM5872436"),
        line_items=[RawLineItem(description="Room Rent - Semi Private",
                                amount="3,500.00")],
    )


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_surrogate_preserves_length(key: RedactionKey) -> None:
    """Page geometry is signal. A shorter surrogate reflows the layout."""
    for value in ("Mr. Ramesh Kumar", "UH253950", "P/123456789012", "A"):
        assert len(surrogate(value, key)) == len(value)


def test_surrogate_preserves_character_class(key: RedactionKey) -> None:
    for value in ("UH253950", "Mr. Ramesh Kumar", "OIC/73183865587"):
        out = surrogate(value, key)
        for original, replaced in zip(value, out):
            assert original.isdigit() == replaced.isdigit()
            assert original.isupper() == replaced.isupper()
            assert original.islower() == replaced.islower()


def test_punctuation_and_spacing_are_untouched(key: RedactionKey) -> None:
    """Slashes and dots are layout, not identity."""
    out = surrogate("OIC/73183865587", key)
    assert out[3] == "/"
    assert surrogate("Mr. Ramesh Kumar", key)[2] == "."   # "Mr." -> index 2
    assert surrogate("Mr. Ramesh Kumar", key).count(" ") == 2


def test_surrogate_actually_changes_the_value(key: RedactionKey) -> None:
    for value in ("Mr. Ramesh Kumar", "UH253950", "P/123456789012"):
        assert surrogate(value, key) != value


# --------------------------------------------------------------------------
# one-wayness and determinism
# --------------------------------------------------------------------------

def test_same_value_maps_consistently(key: RedactionKey) -> None:
    """One patient across three pages must stay one patient."""
    assert surrogate("Ramesh Kumar", key) == surrogate("Ramesh Kumar", key)


def test_different_keys_give_different_surrogates(tmp_path: Path) -> None:
    a = RedactionKey.load_or_create(tmp_path / "a" / "k.key")
    b = RedactionKey.load_or_create(tmp_path / "b" / "k.key")
    assert surrogate("Ramesh Kumar", a) != surrogate("Ramesh Kumar", b)


def test_kind_separates_namespaces(key: RedactionKey) -> None:
    """The same string as a name and as a UHID must not collide."""
    assert surrogate("A1234", key, "patient.name") != surrogate("A1234", key, "patient.uhid")


def test_a_short_key_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        RedactionKey(b"too short")


def test_key_inside_the_repository_is_refused() -> None:
    """A one-way mapping whose key sits beside the data is not one-way."""
    repo = Path(__file__).resolve().parent.parent
    with pytest.raises(ValueError, match="inside the repository"):
        RedactionKey.load_or_create(repo / "data" / "redaction.key")


def test_key_file_is_created_with_enough_entropy(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "redaction.key"
    RedactionKey.load_or_create(path)
    assert path.exists() and len(path.read_bytes()) >= 32
    # reloading must give the same surrogates, not a fresh key
    assert surrogate("x", RedactionKey.load_or_create(path)) == surrogate(
        "x", RedactionKey.load_or_create(path))


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

def test_every_pii_field_is_replaced(key: RedactionKey) -> None:
    redacted, manifest = redact_document(_document(), key, "doc1")
    for path in PII_FIELDS:
        block, name = path.split(".", 1)
        original = getattr(getattr(_document(), block), name)
        if original:
            assert getattr(getattr(redacted, block), name) != original
            assert path in manifest.fields_redacted


def test_the_original_is_not_mutated(key: RedactionKey) -> None:
    """A caller must not be able to keep working with a half-redacted object."""
    document = _document()
    redact_document(document, key, "doc1")
    assert document.patient.name == "Mr. Ramesh Kumar"


def test_non_identifying_fields_survive(key: RedactionKey) -> None:
    """Ward, age and amounts are the data; redacting them destroys the corpus."""
    redacted, _ = redact_document(_document(), key, "doc1")
    assert redacted.patient.ward_type == "Semi-Private"
    assert redacted.patient.age == "45"
    assert redacted.line_items[0].amount == "3,500.00"
    assert redacted.hospital.city == "Hyderabad"


def test_the_hospital_is_not_the_data_subject(key: RedactionKey) -> None:
    """Provider identity is needed to slice results by hospital."""
    redacted, _ = redact_document(_document(), key, "doc1")
    assert redacted.hospital.name == "Sunrise Hospital"


def test_pii_inside_a_line_item_description_is_caught(key: RedactionKey) -> None:
    document = _document()
    document.line_items[0].description = "Room Rent - contact 9876543210"  # pii-allow
    redacted, manifest = redact_document(document, key, "doc1")
    assert "9876543210" not in redacted.line_items[0].description  # pii-allow
    assert manifest.text_patterns.get("phone") == 1


# --------------------------------------------------------------------------
# free text
# --------------------------------------------------------------------------

def test_text_patterns_are_redacted(key: RedactionKey) -> None:
    text = "Call 9876543210 or mail a@b.com, PAN ABCDE1234F"  # pii-allow
    out, counts = redact_text(text, key)
    assert "9876543210" not in out and "a@b.com" not in out  # pii-allow
    assert "ABCDE1234F" not in out                            # pii-allow
    assert set(counts) == {"phone", "email", "pan"}
    assert len(out) == len(text)          # geometry again


def test_benign_text_is_untouched(key: RedactionKey) -> None:
    text = "Room Rent - Semi Private  3,500.00  qty 2"
    assert redact_text(text, key) == (text, {})


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def test_manifest_records_what_not_the_value(key: RedactionKey) -> None:
    """A manifest listing originals would recreate the problem it documents."""
    _, manifest = redact_document(_document(), key, "doc1", consent_reference="CONSENT#4")
    blob = json.dumps(manifest.to_dict())
    assert "Ramesh" not in blob
    assert "UH253950" not in blob
    assert "patient.name" in blob
    assert manifest.consent_reference == "CONSENT#4"
    assert manifest.key_fingerprint


def test_manifest_appends(tmp_path: Path, key: RedactionKey) -> None:
    path = tmp_path / "manifest.jsonl"
    for index in range(3):
        _, manifest = redact_document(_document(), key, f"doc{index}")
        manifest.append_to(path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_redact_value_is_none_safe(key: RedactionKey) -> None:
    assert redact_value(None, key, "patient.name") is None

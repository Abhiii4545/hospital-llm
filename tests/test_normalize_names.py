"""Person, organisation and identifier normalization."""

from __future__ import annotations

import pytest

from reckon.normalize import (
    collapse_whitespace,
    normalize_identifier,
    normalize_name,
    normalize_org_name,
    normalize_text,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mr. Ramesh Kumar", "ramesh kumar"),
        ("Mrs Kavitha Rao", "kavitha rao"),
        ("Ms. Priya", "priya"),
        ("Dr. Suresh Babu", "suresh babu"),
        ("Smt. Lakshmi Devi", "lakshmi devi"),
        ("Sri Venkatesh", "venkatesh"),
        ("Shri Ramesh Kumar", "ramesh kumar"),
        ("SHRI RAMESH KUMAR", "ramesh kumar"),
        ("Master Arjun", "arjun"),
        ("Baby of Lakshmi", "lakshmi"),
        ("B/O Lakshmi", "lakshmi"),
        # stacked honorifics
        ("Dr. Mrs. Kavitha  Rao", "kavitha rao"),
        ("Mr Sri Ramesh", "ramesh"),
        # whitespace and case
        ("  RAMESH   KUMAR  ", "ramesh kumar"),
        ("Ramesh\tKumar", "ramesh kumar"),
    ],
)
def test_honorifics_and_case(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("M. Aadithya Ram", "m aadithya ram"),
        ("M Aadithya Ram", "m aadithya ram"),
        ("K. V. Subba Rao", "k v subba rao"),
        ("A.B.C. Naidu", "a b c naidu"),
    ],
)
def test_initials_first_telugu_convention(raw: str, expected: str) -> None:
    """``M.`` and ``M`` are the same name written two ways - formatting, not value."""
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mrudula Devi", "mrudula devi"),   # "Mr" must not eat "Mrudula"
        ("Srinivas Reddy", "srinivas reddy"),  # "Sri" must not eat "Srinivas"
        ("Msaddi Rao", "msaddi rao"),       # "Ms" must not eat "Msaddi"
        ("Drupad Sharma", "drupad sharma"),  # "Dr" must not eat "Drupad"
    ],
)
def test_honorific_stripping_requires_a_word_boundary(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_single_name_patient() -> None:
    assert normalize_name("Lakshmi") == "lakshmi"
    assert normalize_name("LAKSHMI") == "lakshmi"


@pytest.mark.parametrize("raw", [None, "", "   ", ".", "Mr.", "Dr"])
def test_name_empty_cases(raw: str | None) -> None:
    assert normalize_name(raw) is None


def test_org_name_keeps_honorifics() -> None:
    """"Dr." is part of an organisation's name and must survive."""
    assert normalize_org_name("Dr. Rao's Nursing Home") == "dr. rao's nursing home"
    assert normalize_org_name("  Apollo Hospitals Ltd.  ") == "apollo hospitals ltd"
    assert normalize_org_name("KIMS - Secunderabad") == "kims - secunderabad"


def test_org_name_empty_cases() -> None:
    assert normalize_org_name(None) is None
    assert normalize_org_name("  ...  ") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("POL-1234 5678", "POL12345678"),
        ("pol-1234 5678", "POL12345678"),
        ("UH/2025/00123", "UH202500123"),
        ("  IP 4521  ", "IP4521"),
        ("1.", "1"),
        ("12)", "12"),
        ("30049099", "30049099"),
    ],
)
def test_identifier(raw: str, expected: str) -> None:
    assert normalize_identifier(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "  ", "---", "//"])
def test_identifier_empty_cases(raw: str | None) -> None:
    assert normalize_identifier(raw) is None


def test_text_collapses_but_keeps_punctuation() -> None:
    assert normalize_text("Room  Rent - Deluxe,  1 Day") == "room rent - deluxe, 1 day"
    assert normalize_text(None) is None


def test_collapse_whitespace_handles_nbsp_and_newlines() -> None:
    assert collapse_whitespace("a  b\n\nc\t d") == "a b c d"

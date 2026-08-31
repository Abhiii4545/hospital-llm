"""Age, sex, ward, category, payability and GSTIN normalization."""

from __future__ import annotations

import pytest

from reckon.normalize import (
    WARD_TYPES,
    gstin_check_digit,
    is_valid_gstin,
    normalize_age,
    normalize_bool,
    normalize_category,
    normalize_gstin,
    normalize_sex,
    normalize_ward_type,
)
from reckon.schema import Category


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("45", 45),
        ("45 Y", 45),
        ("45Y", 45),
        ("45 yrs", 45),
        ("45 years", 45),
        ("45Y/M", 45),
        ("0", 0),
        ("  72  ", 72),
    ],
)
def test_age_in_years(raw: str, expected: int) -> None:
    assert normalize_age(raw) == expected


@pytest.mark.parametrize("raw", ["6 M", "6 months", "6M", "3 weeks", "10 days"])
def test_sub_year_age_floors_to_zero(raw: str) -> None:
    """Lossy on purpose: adjudication rules key off whole years.

    The raw string is always retained separately, so the raw-string metric still
    sees the original value.
    """
    assert normalize_age(raw) == 0


@pytest.mark.parametrize("raw", [None, "", "  ", "adult", "N/A"])
def test_age_rejected(raw: str | None) -> None:
    assert normalize_age(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("M", "male"), ("m", "male"), ("Male", "male"), ("MALE", "male"),
        ("F", "female"), ("Female", "female"),
        ("O", "other"), ("Other", "other"), ("Transgender", "other"),
    ],
)
def test_sex(raw: str, expected: str) -> None:
    assert normalize_sex(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "x", "M/F", "unknown"])
def test_sex_ambiguous_is_none_not_a_guess(raw: str | None) -> None:
    assert normalize_sex(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("General Ward", "general"),
        ("GENERAL", "general"),
        ("Semi-Private", "semi_private"),
        ("Semi Private Room", "semi_private"),
        ("Twin Sharing", "semi_private"),
        ("Private", "private"),
        ("Single Room", "private"),
        ("Deluxe Room", "deluxe"),
        ("Suite", "suite"),
        ("ICU", "icu"),
        ("Intensive Care Unit", "icu"),
        ("ICCU", "iccu"),
        ("NICU", "nicu"),
        ("HDU", "hdu"),
        ("Day Care", "day_care"),
        ("Casualty", "emergency"),
    ],
)
def test_ward_type(raw: str, expected: str) -> None:
    result = normalize_ward_type(raw)
    assert result == expected
    assert result in WARD_TYPES


def test_unrecognised_ward_falls_through_to_other() -> None:
    assert normalize_ward_type("Zzz Block 4") == "other"
    assert normalize_ward_type(None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("room_rent", Category.ROOM_RENT),
        ("Room Rent", Category.ROOM_RENT),
        ("Bed Charges", Category.ROOM_RENT),
        ("Nursing Charges", Category.NURSING),
        ("Pharmacy", Category.PHARMACY),
        ("Medicines", Category.PHARMACY),
        ("Consumables", Category.CONSUMABLES),
        ("X-Ray Chest PA", Category.RADIOLOGY),
        ("MRI Brain", Category.RADIOLOGY),
        ("Laboratory Investigations", Category.DIAGNOSTICS),
        ("Surgeon Fee - OT", Category.SURGERY),
        ("Consultation Charges", Category.PROFESSIONAL_FEES),
        ("Ventilator Charges", Category.EQUIPMENT),
        ("Registration Fee", Category.ADMINISTRATIVE),
        ("Attendant Charges", Category.NON_MEDICAL),
    ],
)
def test_category(raw: str, expected: Category) -> None:
    assert normalize_category(raw) is expected


def test_category_unknown_is_other_and_missing_is_none() -> None:
    assert normalize_category("Zzzz") is Category.OTHER
    assert normalize_category(None) is None


@pytest.mark.parametrize("raw", ["Y", "y", "Yes", "TRUE", "1", "Payable", "Admissible"])
def test_bool_true(raw: str) -> None:
    assert normalize_bool(raw) is True


@pytest.mark.parametrize(
    "raw", ["N", "no", "FALSE", "0", "Non-Payable", "non payable", "Deducted"]
)
def test_bool_false(raw: str) -> None:
    assert normalize_bool(raw) is False


@pytest.mark.parametrize("raw", [None, "", "maybe", "?"])
def test_bool_unrecognised_is_none(raw: str | None) -> None:
    assert normalize_bool(raw) is None


@pytest.mark.parametrize("gstin", ["27AAPFU0939F1ZV", "29AAGCB7383J1Z4"])  # pii-allow
def test_valid_gstin_checksums(gstin: str) -> None:
    assert is_valid_gstin(gstin)
    assert gstin_check_digit(gstin[:14]) == gstin[14]


@pytest.mark.parametrize(
    "gstin",
    [
        "27AAPFU0939F1ZX",   # right structure, wrong check digit  (pii-allow)
        "27AAPFU0939F1AV",   # 14th char must be Z
        "27AAPFU0939F1Z",    # too short
        "27AAPFU0939F1ZVV",  # too long
        "AAPFU0939F1ZV27",   # wrong structure
        "abc", "", None,
    ],
)
def test_invalid_gstin(gstin: str | None) -> None:
    assert not is_valid_gstin(gstin)


def test_gstin_normalization_does_not_enforce_validity() -> None:
    """A wrong GSTIN must stay representable so it can be scored as wrong."""
    assert normalize_gstin("27-aapfu 0939f1zx") == "27AAPFU0939F1ZX"  # pii-allow
    assert not is_valid_gstin("27AAPFU0939F1ZX")  # pii-allow


def test_gstin_check_digit_rejects_bad_input() -> None:
    assert gstin_check_digit("short") is None
    assert gstin_check_digit("27AAPFU0939F1!") is None

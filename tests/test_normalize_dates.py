"""Date normalization. Every format the brief names, plus the rejection cases."""

from __future__ import annotations

from datetime import date

import pytest

from reckon.normalize import TWO_DIGIT_YEAR_PIVOT, normalize_date


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # DD/MM/YYYY and separator variants
        ("05/01/2025", date(2025, 1, 5)),
        ("05-01-2025", date(2025, 1, 5)),
        ("05.01.2025", date(2025, 1, 5)),
        ("05 01 2025", date(2025, 1, 5)),
        ("5/1/2025", date(2025, 1, 5)),
        # DD-MM-YY
        ("15-08-24", date(2024, 8, 15)),
        ("15-08-75", date(1975, 8, 15)),
        # DD-MMM-YYYY
        ("05-Jan-2025", date(2025, 1, 5)),
        ("5 January 2025", date(2025, 1, 5)),
        ("05-SEP-2025", date(2025, 9, 5)),
        ("05-Sept-2025", date(2025, 9, 5)),
        ("5 Jan 25", date(2025, 1, 5)),
        # ordinals
        ("1st Jan 2025", date(2025, 1, 1)),
        ("22nd January 2025", date(2025, 1, 22)),
        ("3rd Feb 2025", date(2025, 2, 3)),
        ("11th Dec 2024", date(2024, 12, 11)),
        # month-first with an alphabetic month is unambiguous, so accepted
        ("Jan 5, 2025", date(2025, 1, 5)),
        # ISO passthrough
        ("2025-01-05", date(2025, 1, 5)),
        # day > 12 confirms day-first reading
        ("25/12/2024", date(2024, 12, 25)),
    ],
)
def test_accepted_formats(raw: str, expected: date) -> None:
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "05/01/2025 10:30",
        "05/01/2025 10:30 AM",
        "05/01/2025 10:30:45",
        "05-Jan-2025 18:05",
        "05/01/2025, 10:30 a.m.",
    ],
)
def test_trailing_clock_time_is_discarded(raw: str) -> None:
    assert normalize_date(raw) == date(2025, 1, 5)


def test_numeric_dates_are_always_day_first() -> None:
    """Indian convention. A US-style date yields None, never a wrong date.

    This is the whole point: 05/13/2025 has no valid day-first reading, and
    returning None (a miss) is more honest than returning 13 May.
    """
    assert normalize_date("13/05/2025") == date(2025, 5, 13)
    assert normalize_date("05/13/2025") is None


def test_two_digit_year_pivot_boundary() -> None:
    assert TWO_DIGIT_YEAR_PIVOT == 70
    assert normalize_date("01-01-69") == date(2069, 1, 1)
    assert normalize_date("01-01-70") == date(1970, 1, 1)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "N/A",
        "not a date",
        "31/02/2025",   # impossible calendar date
        "32/01/2025",
        "00/01/2025",
        "05/00/2025",
        "05-Xyz-2025",  # unknown month name
        "2025",
        "05/2025",
    ],
)
def test_rejected(raw: str | None) -> None:
    assert normalize_date(raw) is None


def test_whitespace_and_unicode_dashes_are_tolerated() -> None:
    assert normalize_date("  05–01–2025  ") == date(2025, 1, 5)
    assert normalize_date("05 Jan 2025") == date(2025, 1, 5)


def test_is_pure_function() -> None:
    """Same input, same output, no hidden state."""
    for _ in range(3):
        assert normalize_date("05/01/2025") == date(2025, 1, 5)

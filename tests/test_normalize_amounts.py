"""Money normalization. Decimal only - a float appearing here is a defect."""

from __future__ import annotations

from decimal import Decimal

import pytest

from reckon.normalize import normalize_amount, normalize_quantity


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # plain
        ("1234", "1234"),
        ("1234.56", "1234.56"),
        (".50", "0.50"),
        ("1234.", "1234"),
        # Indian digit grouping
        ("1,23,456.00", "123456.00"),
        ("12,34,567.89", "1234567.89"),
        ("1,234", "1234"),
        # OCR space grouping
        ("1 23 456.00", "123456.00"),
        # currency prefixes
        ("Rs. 1,23,456.00", "123456.00"),
        ("Rs 500", "500"),
        ("RS.500", "500"),
        ("INR 500", "500"),
        ("₹1,234.50", "1234.50"),
        ("₹ 1,234.50", "1234.50"),
        # trailing /-
        ("500/-", "500"),
        ("1,234.00/-", "1234.00"),
        ("Rs. 2,500/-", "2500"),
        ("500-", "500"),
        # negatives
        ("(1,234.00)", "-1234.00"),
        ("-500", "-500"),
        ("Rs. (2,500.00)", "-2500.00"),
        # zero conventions used in bill columns
        ("NIL", "0"),
        ("nil", "0"),
        ("-", "0"),
        ("--", "0"),
    ],
)
def test_accepted_amounts(raw: str, expected: str) -> None:
    assert normalize_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "N/A", "1.2.3", "Rs.", "??"])
def test_rejected_amounts(raw: str | None) -> None:
    assert normalize_amount(raw) is None


def test_returns_decimal_never_float() -> None:
    value = normalize_amount("1,23,456.78")
    assert isinstance(value, Decimal)
    assert not isinstance(value, float)


def test_no_binary_float_error_is_introduced() -> None:
    """The classic failure: 0.1 + 0.2 != 0.3 in binary floating point.

    Parsing through Decimal(str) keeps the decimal value exact, which is why the
    schema refuses floats at its boundary.
    """
    total = normalize_amount("0.10") + normalize_amount("0.20")
    assert total == Decimal("0.30")
    assert str(total) == "0.30"


def test_large_bill_sums_exactly() -> None:
    parts = ["1,23,456.78", "45,000.05", "9,999.99", "0.18"]
    assert sum(normalize_amount(p) for p in parts) == Decimal("178457.00")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", "1"),
        ("2.5", "2.5"),
        ("0.5", "0.5"),
        ("01", "1"),
        ("1 x", "1"),
        ("2 nos", "2"),
        ("3 Nos.", "3"),
        ("4 units", "4"),
    ],
)
def test_quantity(raw: str, expected: str) -> None:
    assert normalize_quantity(raw) == Decimal(expected)


def test_quantity_rejects_garbage() -> None:
    assert normalize_quantity("as required") is None
    assert normalize_quantity(None) is None

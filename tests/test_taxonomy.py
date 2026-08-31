"""Error taxonomy classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from reckon.data.mini_set import build_mini_set
from reckon.eval.run import build_pairs
from reckon.eval.taxonomy import (
    CATEGORIES,
    classify_field_error,
    collect,
    summarise,
    write_taxonomy,
)
from reckon.models.baselines.b0_rules import B0RulesEngine
from reckon.normalize import normalize_amount, normalize_date


def classify(field, truth, pred, others=None):
    """Helper mirroring how collect() calls the classifier."""
    norm = normalize_amount if "amount" in field else (
        normalize_date if "date" in field else (lambda v: v)
    )
    return classify_field_error(field, truth, pred, norm(truth), norm(pred), others)


def test_every_category_declares_an_action() -> None:
    """A bucket that does not imply a different fix is not worth having."""
    for name, (meaning, action) in CATEGORIES.items():
        assert meaning and action, name
    actions = [action for _, action in CATEGORIES.values()]
    assert len(set(actions)) == len(actions), "two categories imply the same fix"


def test_identical_values_are_not_an_error() -> None:
    assert classify("patient.name", "Ramesh", "Ramesh") is None


def test_formatting_only_is_separated_from_real_error() -> None:
    assert classify("totals.net_amount", "Rs. 1,000.00", "1000") == "formatting_only"


def test_missed_and_hallucinated_are_distinct() -> None:
    assert classify("patient.uhid", "UH1", None) == "missed_field"
    assert classify("patient.uhid", None, "UH1") == "hallucinated_field"


def test_digit_confusion_is_detected() -> None:
    assert classify("totals.net_amount", "1568.00", "1668.00") == "digit_confusion"


def test_magnitude_error_is_detected() -> None:
    """Misreading Indian lakh grouping is an order-of-magnitude failure."""
    assert classify("totals.net_amount", "1,23,456.00", "12345.60") == "magnitude_error"


def test_unparseable_date_is_its_own_category() -> None:
    assert classify("patient.admission_date", "05/01/2025", "05|01|2O25") == "date_unparsed"


def test_value_taken_from_another_field_is_flagged() -> None:
    others = {"patient.uhid": "UH999", "patient.ip_number": "IP123"}
    assert classify("patient.ip_number", "IP123", "UH999", others) == "wrong_field"


def test_genuinely_different_values_fall_through_to_the_hard_bucket() -> None:
    assert classify("patient.name", "Ramesh Kumar", "Suresh Babu") == "value_confusion"


def test_collect_finds_failures_on_a_real_baseline() -> None:
    docs = build_mini_set()[:8]
    cases = collect(build_pairs(docs, B0RulesEngine().extract))
    assert cases
    assert all(case.category in CATEGORIES for case in cases)
    assert any(case.field == "line_items[]" for case in cases)


def test_summary_breaks_down_by_slice() -> None:
    docs = build_mini_set()[:8]
    counts = summarise(collect(build_pairs(docs, B0RulesEngine().extract)))
    assert counts["category"]
    assert counts["field"]
    # Discovered, not hard-coded: the corpus calls this key `template`, and an
    # earlier hard-coded list of ("layout", "quality", ...) silently produced no
    # slice tables at all.
    assert "template" in counts
    assert "messiness" in counts
    # A key with one value everywhere is not a slice. `scan_quality` is "clean"
    # for the whole mini-set, so it is correctly excluded.
    assert "scan_quality" not in counts
    assert "source" not in counts
    # Identifier-ish keys must never become slices.
    assert "page_id" not in counts and "doc_id" not in counts


def test_taxonomy_report_has_hypothesis_slots(tmp_path: Path) -> None:
    docs = build_mini_set()[:6]
    cases = collect(build_pairs(docs, B0RulesEngine().extract))
    path = write_taxonomy(cases, tmp_path / "t.md", source="mini-set (B0)", sample=5)
    text = path.read_text(encoding="utf-8")

    assert "# Error taxonomy" in text
    assert "hypothesised cause" in text
    assert "proposed fix" in text
    assert "real-test" in text            # the lock is restated in the artefact
    assert "exactly TWO fixes" in text


def test_report_states_the_real_dev_only_rule(tmp_path: Path) -> None:
    path = write_taxonomy([], tmp_path / "t.md", source="none")
    assert "never inspected" in path.read_text(encoding="utf-8")

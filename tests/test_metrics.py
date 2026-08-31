"""Metric behaviour, pinned against hand-written fixtures.

The harness is what every later claim rests on, so these tests care most about
the cases where a metric could flatter a model: perfect scores, absent values,
hallucinations, and reordered line items.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from reckon.eval.metrics import (
    DEFAULT_MATCH_THRESHOLD,
    DocumentPair,
    auto_processing_rate,
    coverage_curve,
    deduction_total,
    default_net_payable,
    deployment_sentence,
    match_line_items,
    median,
    percentile,
    score_business,
    score_documents,
    score_fields,
    score_line_items,
)
from reckon.normalize import normalize_document
from reckon.schema import (
    FIELD_PATHS,
    RawDocument,
    RawHospital,
    RawLineItem,
    RawPatient,
    RawTotals,
)


def make_raw(
    *,
    hospital_name: str | None = "Apollo Hospitals",
    patient_name: str | None = "Mr. Ramesh Kumar",
    net: str | None = "Rs. 1,000.00",
    items: list[RawLineItem] | None = None,
) -> RawDocument:
    return RawDocument(
        hospital=RawHospital(name=hospital_name),
        patient=RawPatient(name=patient_name),
        totals=RawTotals(net_amount=net),
        line_items=items or [],
    )


def item(description: str, amount: str, payable: str | None = "Y") -> RawLineItem:
    return RawLineItem(description=description, amount=amount, is_payable=payable)


def pair(pred: RawDocument, truth: RawDocument, doc_id: str = "d1") -> DocumentPair:
    return DocumentPair(doc_id=doc_id, pred=pred, truth=truth)


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------

def test_median_and_percentile_are_decimal_exact() -> None:
    values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
    assert median(values) == Decimal("2.5")
    assert median([Decimal("5")]) == Decimal("5")
    assert median([]) == Decimal(0)
    assert isinstance(median(values), Decimal)


def test_percentile_uses_nearest_rank() -> None:
    values = [Decimal(str(i)) for i in range(1, 101)]
    assert percentile(values, 0.95) == Decimal("95")
    assert percentile(values, 1.0) == Decimal("100")
    assert percentile([], 0.95) == Decimal(0)


# --------------------------------------------------------------------------
# field metrics
# --------------------------------------------------------------------------

def test_identical_documents_score_perfectly() -> None:
    doc = make_raw()
    scores = score_fields([pair(doc, doc)])
    assert scores["hospital.name"].exact == 1.0
    assert scores["hospital.name"].normalized_exact == 1.0
    assert scores["hospital.name"].cer == 0.0


def test_every_field_is_reported_separately() -> None:
    """No averaging: an average would hide a collapsed field."""
    scores = score_fields([pair(make_raw(), make_raw())])
    assert set(scores) == set(FIELD_PATHS)


def test_formatting_only_difference_shows_up_as_the_raw_normalized_gap() -> None:
    """The whole reason both metrics exist."""
    pred = make_raw(patient_name="RAMESH KUMAR")
    truth = make_raw(patient_name="Mr. Ramesh Kumar")
    score = score_fields([pair(pred, truth)])["patient.name"]

    assert score.exact == 0.0           # raw strings differ
    assert score.normalized_exact == 1.0  # same person
    assert score.formatting_gap == 1.0


def test_a_genuinely_wrong_value_is_wrong_under_both_metrics() -> None:
    score = score_fields(
        [pair(make_raw(patient_name="Suresh Babu"), make_raw(patient_name="Ramesh Kumar"))]
    )["patient.name"]
    assert score.exact == 0.0
    assert score.normalized_exact == 0.0
    assert score.cer > 0.0


def test_both_absent_counts_as_correct() -> None:
    """Correctly reporting that a field is absent is a correct answer."""
    score = score_fields(
        [pair(make_raw(hospital_name=None), make_raw(hospital_name=None))]
    )["hospital.name"]
    assert score.exact == 1.0
    assert score.cer == 0.0
    assert score.misses == 0
    assert score.hallucinations == 0


def test_misses_and_hallucinations_are_counted_separately() -> None:
    miss = score_fields(
        [pair(make_raw(hospital_name=None), make_raw(hospital_name="Apollo"))]
    )["hospital.name"]
    assert miss.misses == 1 and miss.hallucinations == 0

    invented = score_fields(
        [pair(make_raw(hospital_name="Apollo"), make_raw(hospital_name=None))]
    )["hospital.name"]
    assert invented.hallucinations == 1 and invented.misses == 0


def test_cer_is_capped_at_one() -> None:
    """A long hallucination must not dominate the average."""
    score = score_fields(
        [pair(make_raw(hospital_name="x" * 500), make_raw(hospital_name="ab"))]
    )["hospital.name"]
    assert score.cer == 1.0


# --------------------------------------------------------------------------
# line items
# --------------------------------------------------------------------------

def test_line_items_match_regardless_of_order() -> None:
    """Line items are a set, not a sequence."""
    pred = make_raw(items=[item("Nursing Charges", "200"), item("Room Rent", "500")])
    truth = make_raw(items=[item("Room Rent", "500"), item("Nursing Charges", "200")])
    score = score_line_items([pair(pred, truth)])
    assert score.n_matched == 2
    assert score.f1 == 1.0
    assert score.insertions == 0 and score.deletions == 0


def test_near_miss_descriptions_still_match() -> None:
    pred = make_raw(items=[item("Room Rent - Deluxe", "500")])
    truth = make_raw(items=[item("Room Rent Deluxe", "500")])
    assert score_line_items([pair(pred, truth)]).n_matched == 1


def test_unrelated_rows_are_not_matched() -> None:
    """Hungarian returns a complete pairing, so the threshold has to do the work."""
    pred = make_raw(items=[item("Ambulance Charges", "500")])
    truth = make_raw(items=[item("MRI Brain Contrast", "500")])
    score = score_line_items([pair(pred, truth)])
    assert score.n_matched == 0
    assert score.insertions == 1
    assert score.deletions == 1


def test_insertions_and_deletions_are_reported_separately() -> None:
    pred = make_raw(items=[item("Room Rent", "500"), item("Invented Row", "999")])
    truth = make_raw(items=[item("Room Rent", "500"), item("Nursing", "200")])
    score = score_line_items([pair(pred, truth)])
    assert score.n_matched == 1
    assert score.insertions == 1   # the invented row
    assert score.deletions == 1    # nursing, never produced


def test_precision_recall_f1_arithmetic() -> None:
    pred = make_raw(items=[item("Room Rent", "500"), item("Zzz Unrelated", "1")])
    truth = make_raw(items=[item("Room Rent", "500")])
    score = score_line_items([pair(pred, truth)])
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(2 / 3)


def test_attribute_accuracy_within_matched_rows() -> None:
    pred = make_raw(items=[item("Room Rent", "999")])
    truth = make_raw(items=[item("Room Rent", "500")])
    score = score_line_items([pair(pred, truth)])
    assert score.n_matched == 1
    assert score.attribute_accuracy["description"] == 1.0
    assert score.attribute_accuracy["amount"] == 0.0


def test_empty_predictions_score_zero_not_a_crash() -> None:
    score = score_line_items([pair(make_raw(), make_raw(items=[item("Room Rent", "500")]))])
    assert score.n_matched == 0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.deletions == 1


def test_match_threshold_is_respected() -> None:
    pred = [normalize_document(make_raw(items=[item("Room Rent", "1")])).line_items[0]]
    truth = [normalize_document(make_raw(items=[item("Room Rate", "1")])).line_items[0]]
    assert match_line_items(pred, truth, threshold=0.0)
    assert not match_line_items(pred, truth, threshold=0.99)
    assert DEFAULT_MATCH_THRESHOLD == 0.60


# --------------------------------------------------------------------------
# document metrics
# --------------------------------------------------------------------------

def test_document_scores_are_perfect_for_identical_input() -> None:
    doc = make_raw(items=[item("Room Rent", "500")])
    score = score_documents([pair(doc, doc)])
    assert score.strict_exact_match == 1.0
    assert score.ted_accuracy == 1.0


def test_strict_exact_match_is_all_or_nothing() -> None:
    pred = make_raw(net="Rs. 999.00")
    truth = make_raw(net="Rs. 1,000.00")
    score = score_documents([pair(pred, truth)])
    assert score.strict_exact_match == 0.0
    assert score.ted_accuracy < 1.0     # partial credit survives
    assert score.ted_accuracy > 0.5


def test_strict_match_ignores_formatting_because_it_uses_normalized_values() -> None:
    pred = make_raw(net="1000", patient_name="RAMESH KUMAR")
    truth = make_raw(net="Rs. 1,000.00", patient_name="Mr. Ramesh Kumar")
    assert score_documents([pair(pred, truth)]).strict_exact_match == 1.0


# --------------------------------------------------------------------------
# business metrics
# --------------------------------------------------------------------------

def test_net_payable_prefers_line_items_and_excludes_non_payable() -> None:
    doc = normalize_document(
        make_raw(items=[item("Room Rent", "500"), item("Attendant", "100", payable="N")])
    )
    assert default_net_payable(doc) == Decimal("500")
    assert deduction_total(doc) == Decimal("100")


def test_net_payable_falls_back_to_stated_total() -> None:
    doc = normalize_document(make_raw(net="Rs. 1,000.00"))
    assert default_net_payable(doc) == Decimal("1000")


def test_business_error_is_zero_for_a_perfect_extraction() -> None:
    doc = make_raw(items=[item("Room Rent", "500")])
    score = score_business([pair(doc, doc)])
    assert score.median_error == Decimal(0)
    assert score.pct_error_over_threshold == 0.0
    assert score.pct_deduction_total_exact == 1.0
    assert score.rupees_per_1000_claims == Decimal(0)


def test_business_error_is_measured_in_rupees() -> None:
    pred = make_raw(items=[item("Room Rent", "450")])
    truth = make_raw(items=[item("Room Rent", "500")])
    score = score_business([pair(pred, truth)])
    assert score.median_error == Decimal("50")
    assert score.pct_error_over_threshold == 0.0     # 50 <= 100
    assert score.rupees_per_1000_claims == Decimal("50000")


def test_material_error_threshold() -> None:
    pred = make_raw(items=[item("Room Rent", "300")])
    truth = make_raw(items=[item("Room Rent", "500")])
    score = score_business([pair(pred, truth)])
    assert score.median_error == Decimal("200")
    assert score.pct_error_over_threshold == 1.0


def test_a_missing_prediction_is_charged_the_whole_amount() -> None:
    """Abstaining silently must not look better than being wrong."""
    score = score_business([pair(make_raw(net=None), make_raw(net="Rs. 1,000.00"))])
    assert score.median_error == Decimal("1000")


def test_documents_without_a_true_payable_are_excluded_not_scored_as_perfect() -> None:
    score = score_business([pair(make_raw(net=None), make_raw(net=None))])
    assert score.n == 1
    assert score.n_scored == 0
    assert score.median_error == Decimal(0)


# --------------------------------------------------------------------------
# coverage and abstention
# --------------------------------------------------------------------------

def test_coverage_falls_as_the_threshold_rises() -> None:
    doc = make_raw()
    pairs = [pair(doc, doc, "d1")]
    confidences = {"d1": {path: 0.5 for path in FIELD_PATHS}}
    curve = coverage_curve(pairs, confidences, thresholds=[0.0, 0.4, 0.6, 1.0])

    assert curve[0].field_coverage == 1.0
    assert curve[1].field_coverage == 1.0
    assert curve[2].field_coverage == 0.0
    assert curve[3].field_coverage == 0.0


def test_missing_confidence_is_treated_as_zero() -> None:
    """A field with no score must never inflate coverage."""
    doc = make_raw()
    curve = coverage_curve([pair(doc, doc, "d1")], {}, thresholds=[0.1])
    assert curve[0].field_coverage == 0.0


def test_abstaining_on_the_wrong_fields_raises_accuracy() -> None:
    pred = make_raw(patient_name="Totally Wrong Person")
    truth = make_raw(patient_name="Ramesh Kumar")
    confidences = {"d1": {path: 0.9 for path in FIELD_PATHS}}
    confidences["d1"]["patient.name"] = 0.1

    low = coverage_curve([pair(pred, truth, "d1")], confidences, thresholds=[0.0])[0]
    high = coverage_curve([pair(pred, truth, "d1")], confidences, thresholds=[0.5])[0]

    assert high.field_accuracy > low.field_accuracy
    assert high.field_coverage < low.field_coverage


def test_auto_processing_rate_picks_the_widest_coverage_meeting_the_target() -> None:
    doc = make_raw()
    confidences = {"d1": {path: 0.8 for path in FIELD_PATHS}}
    curve = coverage_curve([pair(doc, doc, "d1")], confidences)
    point = auto_processing_rate(curve, target_accuracy=0.95)
    assert point is not None
    assert point.field_accuracy == 1.0
    assert point.field_coverage == 1.0


def test_unreachable_target_returns_none_rather_than_zero() -> None:
    """'No threshold works' is a result, not a 0% to be quietly rendered."""
    pred = make_raw(hospital_name="Wrong", patient_name="Wrong", net="1")
    truth = make_raw()
    confidences = {"d1": {path: 1.0 for path in FIELD_PATHS}}
    curve = coverage_curve([pair(pred, truth, "d1")], confidences, thresholds=[0.0, 1.0])
    assert auto_processing_rate(curve, target_accuracy=0.999) is None
    assert "every document requires human review" in deployment_sentence(curve, 0.999)


def test_deployment_sentence_reads_like_a_deployment_decision() -> None:
    doc = make_raw()
    confidences = {"d1": {path: 0.8 for path in FIELD_PATHS}}
    curve = coverage_curve([pair(doc, doc, "d1")], confidences)
    sentence = deployment_sentence(curve, 0.95)
    assert "auto-processes" in sentence
    assert "human review" in sentence
    assert "%" in sentence


# --------------------------------------------------------------------------
# eval cache
# --------------------------------------------------------------------------

def test_cache_does_not_change_any_number() -> None:
    """A stale cache here would silently corrupt every reported result.

    The cache is the one optimisation in the harness, so it is checked against
    the uncached path rather than trusted.
    """
    from reckon.eval.metrics import EvalCache

    pred = make_raw(items=[item("Room Rent", "450"), item("Attendant", "100", "N")])
    truth = make_raw(items=[item("Room Rent", "500"), item("Attendant", "100", "N")])
    pairs = [pair(pred, truth, "d1"), pair(truth, truth, "d2")]

    cache = EvalCache()
    assert score_fields(pairs) == score_fields(pairs, cache)
    assert score_line_items(pairs) == score_line_items(pairs, cache=cache)
    assert score_documents(pairs) == score_documents(pairs, cache)
    assert score_business(pairs) == score_business(pairs, cache=cache)


def test_cache_keys_on_doc_id_not_position() -> None:
    """Two different documents must never collide in the cache."""
    from reckon.eval.metrics import EvalCache

    a = pair(make_raw(net="100"), make_raw(net="100"), "a")
    b = pair(make_raw(net="999"), make_raw(net="100"), "b")
    cache = EvalCache()

    first = score_business([a, b], cache=cache)
    second = score_business([a, b], cache=cache)
    assert first == second
    assert first.median_error == score_business([a, b]).median_error


# --------------------------------------------------------------------------
# the metric that cannot be earned by abstaining
# --------------------------------------------------------------------------

def test_a_system_that_extracts_nothing_scores_the_absence_rate() -> None:
    """The flaw that scoring zero-shot Donut-CORD exposed.

    `normalized_exact` counts a correct absence as correct - right in principle,
    but it means a system predicting nothing scores exactly the rate at which the
    field happens to be absent. Donut-CORD scored 1.00 on `amount_in_words` and
    0.65 on `employee_id` that way, having extracted neither.
    """
    empty = RawDocument()
    pairs = [
        pair(empty, make_raw(hospital_name="Apollo"), "d1"),
        pair(empty, make_raw(hospital_name=None), "d2"),
        pair(empty, make_raw(hospital_name=None), "d3"),
        pair(empty, make_raw(hospital_name=None), "d4"),
    ]
    score = score_fields(pairs)["hospital.name"]

    assert score.normalized_exact == 0.75          # flattered by 3 absences
    assert score.accuracy_when_present == 0.0      # the honest number
    assert score.truth_present == 1


def test_present_only_accuracy_rewards_real_extraction() -> None:
    good = [pair(make_raw(hospital_name="Apollo"),
                 make_raw(hospital_name="Apollo"), f"d{i}") for i in range(3)]
    score = score_fields(good)["hospital.name"]
    assert score.accuracy_when_present == 1.0
    assert score.truth_present == 3


def test_present_only_is_zero_when_the_field_never_appears() -> None:
    """No support means no claim either way, not a free 100%."""
    pairs = [pair(make_raw(hospital_name=None), make_raw(hospital_name=None), "d1")]
    score = score_fields(pairs)["hospital.name"]
    assert score.normalized_exact == 1.0
    assert score.truth_present == 0
    assert score.accuracy_when_present == 0.0


def test_report_explains_the_divergence() -> None:
    from reckon.eval.report import build_report, evaluate_system

    empty = RawDocument()
    pairs = [pair(empty, make_raw(), "d1")]
    text = build_report([evaluate_system("nothing", pairs)], dataset_name="t")
    assert "present-only" in text
    assert "extracts NOTHING" in text

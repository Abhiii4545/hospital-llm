"""Synthetic corpus: generators, layouts, augmentation, pagination, manifest.

The property that matters most is checked first and hardest: **the ground truth
is the string that gets printed**. A corpus whose labels disagree with its pixels
trains a model to be confidently wrong, and no downstream metric can detect it.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reckon.data.augment import (
    MIN_EDGE_ENERGY,
    MIN_LEGIBLE_CONTRAST,
    QUALITY_BUCKETS,
    QUALITY_WEIGHTS,
    augment_page,
    edge_energy,
    ink_contrast,
    is_legible,
    sample_quality,
)
from reckon.data.contact_sheet import sample_stratified
from reckon.data.generators.document import MESSINESS_RATES, generate_document
from reckon.data.generators.values import (
    BILINGUAL_STATES,
    STATE_GST_CODES,
    WARD_RATE_BANDS,
    sample_gstin,
    sample_hospital,
    sample_line_items,
    sample_patient_name,
    sample_policy_number,
)
from reckon.data.layouts import ALL_COLUMNS, ARCHETYPES, LAYOUTS, by_id
from reckon.normalize import is_valid_gstin, normalize_amount, normalize_document
from reckon.schema import Category


def _doc(index: int = 0, spec=None):
    spec = spec or LAYOUTS[index % len(LAYOUTS)]
    return generate_document(
        random.Random(f"t:{index}"), f"t_{index:04d}", spec.id, spec.rows_per_page,
        hospital_type=spec.hospital_type, bilingual=spec.bilingual,
    ), spec


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

def test_layout_count_is_in_the_briefs_range() -> None:
    assert 16 <= len(LAYOUTS) <= 21


def test_every_archetype_the_brief_names_is_present() -> None:
    assert {spec.archetype for spec in LAYOUTS} == set(ARCHETYPES)


def test_layout_ids_are_unique() -> None:
    ids = [spec.id for spec in LAYOUTS]
    assert len(set(ids)) == len(ids)
    assert by_id(ids[0]).id == ids[0]


def test_diversity_is_structural_not_cosmetic() -> None:
    """Column ORDER, header placement and totals placement all have to vary.

    If layouts differed only by font, a fixed-order regex would survive them all
    and the corpus would not be testing the thing that broke v1.
    """
    orders = {spec.columns for spec in LAYOUTS}
    assert len(orders) >= 12, "column orders are not diverse enough"
    assert len({spec.header_position for spec in LAYOUTS}) >= 4
    assert len({spec.totals_position for spec in LAYOUTS}) >= 4
    assert len({spec.table_style for spec in LAYOUTS}) >= 4

    # Specifically: rate-before-quantity must exist somewhere.
    def order(spec, a, b):
        return spec.columns.index(a) < spec.columns.index(b)

    with_both = [s for s in LAYOUTS if "unit_rate" in s.columns and "quantity" in s.columns]
    assert any(order(s, "unit_rate", "quantity") for s in with_both)
    assert any(order(s, "quantity", "unit_rate") for s in with_both)


def test_columns_are_drawn_from_the_declared_vocabulary() -> None:
    for spec in LAYOUTS:
        assert set(spec.columns) <= set(ALL_COLUMNS), spec.id
        assert "description" in spec.columns and "amount" in spec.columns


# --------------------------------------------------------------------------
# value generators
# --------------------------------------------------------------------------

def test_generated_gstins_have_correct_check_digits() -> None:
    """An invalid checksum would be a free giveaway that a document is fake."""
    rng = random.Random(1)
    for code in STATE_GST_CODES:
        for _ in range(20):
            gstin = sample_gstin(rng, code)
            assert is_valid_gstin(gstin), gstin
            assert gstin.startswith(code)


def test_honorific_matches_sex() -> None:
    """"Ms. Rohit Kumar / Sex: F" is exactly the tell that a corpus is synthetic."""
    rng = random.Random(2)
    female_only = {"mrs.", "ms.", "smt.", "kumari"}
    male_only = {"mr.", "sri", "shri", "master"}
    for _ in range(300):
        for sex in ("M", "F"):
            name, _ = sample_patient_name(rng, sex, 40)
            first = name.split()[0].casefold()
            if sex == "M":
                assert first not in female_only, name
            else:
                assert first not in male_only, name


def test_bilingual_documents_sit_in_a_state_whose_script_we_have() -> None:
    """A Telugu header on a Tamil Nadu hospital would teach a false association."""
    for spec in LAYOUTS:
        if not spec.bilingual:
            continue
        for index in range(12):
            doc, _ = _doc(index, spec)
            assert doc.meta["state_code"] in BILINGUAL_STATES
            assert doc.context["script"] == BILINGUAL_STATES[doc.meta["state_code"]]


def test_room_rent_tracks_ward_class() -> None:
    """A correlation that is true of real bills, so worth learning."""
    rng = random.Random(3)
    for ward, (low, high) in WARD_RATE_BANDS.items():
        items = sample_line_items(rng, ward=ward, stay_days=3, n_items=6,
                                  include_surgery=False)
        room = next(i for i in items if i.category == "room_rent")
        assert low <= room.unit_rate <= high, ward


def test_line_item_amount_is_quantity_times_rate_exactly() -> None:
    rng = random.Random(4)
    for _ in range(40):
        for item in sample_line_items(rng, ward="private", stay_days=4,
                                      n_items=10, include_surgery=True):
            assert item.quantity * item.unit_rate == item.amount


def test_policy_number_format_follows_the_insurer() -> None:
    rng = random.Random(5)
    star = {sample_policy_number(rng, "Star Health & Allied Insurance Co. Ltd.")
            for _ in range(20)}
    hdfc = {sample_policy_number(rng, "HDFC ERGO General Insurance") for _ in range(20)}
    assert all(p.startswith("P/") for p in star)
    assert all(p.startswith("HE") for p in hdfc)


def test_no_real_hospital_names_are_used() -> None:
    """Templates are built from observed structure, never copied identity."""
    rng = random.Random(6)
    forbidden = {"apollo", "fortis", "manipal", "narayana", "aiims", "medanta",
                 "max healthcare", "kims", "yashoda", "care hospitals"}
    for _ in range(200):
        name = sample_hospital(rng).name.casefold()
        assert not any(brand in name for brand in forbidden), name


# --------------------------------------------------------------------------
# the property that matters: truth == what is printed
# --------------------------------------------------------------------------

def test_printed_rows_match_the_ground_truth_items() -> None:
    for index in range(25):
        doc, _ = _doc(index)
        for page in doc.pages:
            printed = [r for r in page.rows][len(page.rows) - len(page.items):] \
                if page.duplicate_of_previous_last else page.rows
            assert len(printed) == len(page.items)
            for row, item in zip(printed, page.items):
                assert row["description"] == item.description
                assert row["amount"] == item.amount
                assert row["quantity"] == item.quantity


def test_document_truth_is_the_concatenation_of_page_truth() -> None:
    for index in range(20):
        doc, _ = _doc(index)
        from_pages = [i for page in doc.pages for i in page.items]
        assert from_pages == doc.truth.line_items


def test_totals_reconcile_unless_deliberately_misaligned() -> None:
    """The 5% misaligned-totals rate has to be the ONLY source of disagreement."""
    checked = misaligned = 0
    for index in range(120):
        doc, _ = _doc(index)
        typed = normalize_document(doc.truth)
        summed = sum((i.amount for i in typed.line_items if i.amount), Decimal(0))
        expected = (summed - (typed.totals.discount or 0)
                    + (typed.totals.cgst or 0) + (typed.totals.sgst or 0))
        checked += 1
        if "misaligned_totals" in doc.meta["messiness"]:
            misaligned += 1
        else:
            assert typed.totals.net_amount == expected, doc.doc_id
    assert checked > 0 and misaligned > 0


def test_truth_survives_normalization() -> None:
    for index in range(20):
        doc, _ = _doc(index)
        typed = normalize_document(doc.truth)
        assert typed.patient.admission_date is not None
        assert typed.totals.net_amount is not None
        for item in typed.line_items:
            assert item.amount is not None
            assert item.category in set(Category)


def test_generation_is_deterministic() -> None:
    a = generate_document(random.Random("x"), "d", LAYOUTS[0].id, 20)
    b = generate_document(random.Random("x"), "d", LAYOUTS[0].id, 20)
    assert a.truth == b.truth
    assert a.meta == b.meta


# --------------------------------------------------------------------------
# pagination
# --------------------------------------------------------------------------

def test_header_on_first_page_and_totals_on_last() -> None:
    for index in range(30):
        doc, _ = _doc(index)
        assert doc.pages[0].show_header
        assert doc.pages[-1].show_totals
        assert sum(p.show_header for p in doc.pages) == 1
        assert sum(p.show_totals for p in doc.pages) == 1


def test_multi_page_documents_are_produced() -> None:
    counts = {len(_doc(i)[0].pages) for i in range(80)}
    assert max(counts) > 1, "corpus has no multi-page documents"


def test_duplicated_row_is_printed_but_not_in_the_target() -> None:
    """The assembly layer has to catch it; the model must not be taught to emit it."""
    found = False
    for index in range(200):
        doc, _ = _doc(index)
        for page in doc.pages:
            if page.duplicate_of_previous_last:
                found = True
                assert len(page.rows) == len(page.items) + 1
                previous = doc.pages[page.index - 1]
                assert page.rows[0] == previous.rows[-1]
    assert found, "no duplicated-row page generated in 200 documents"


def test_messiness_rates_are_roughly_as_specified() -> None:
    tally = {key: 0 for key in MESSINESS_RATES}
    n = 400
    for index in range(n):
        doc, _ = _doc(index)
        for tag in str(doc.meta["messiness"]).split(","):
            key = {"duplicate_row": "duplicate_row_across_page_break",
                   "handwritten": "handwritten_correction"}.get(tag, tag)
            if key in tally:
                tally[key] += 1
    for key, target in MESSINESS_RATES.items():
        observed = tally[key] / n
        if key == "duplicate_row_across_page_break":
            assert observed <= target + 0.05   # only fires on multi-page docs
        else:
            assert abs(observed - target) < 0.06, f"{key}: {observed:.3f} vs {target}"


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def _blank_page() -> np.ndarray:
    page = np.full((1123, 794, 3), 255, dtype=np.uint8)
    page[100:110, 60:700] = 20          # some "text"
    for y in range(200, 900, 24):
        page[y:y + 8, 60:740] = 30
    return page


def test_quality_weights_are_a_distribution() -> None:
    assert len(QUALITY_BUCKETS) == len(QUALITY_WEIGHTS)
    assert abs(sum(QUALITY_WEIGHTS) - 1.0) < 1e-9


def test_sampled_quality_is_always_a_known_bucket() -> None:
    rng = random.Random(7)
    assert {sample_quality(rng) for _ in range(200)} <= set(QUALITY_BUCKETS)


@pytest.mark.parametrize("quality", QUALITY_BUCKETS)
def test_every_bucket_stays_legible(quality: str) -> None:
    """The guard that stops the `heavy` bucket becoming label noise again."""
    page = _blank_page()
    for trial in range(4):
        out, _ = augment_page(page.copy(), quality, random.Random(trial))
        assert is_legible(out), (
            f"{quality} trial {trial}: contrast={ink_contrast(out):.1f} "
            f"edges={edge_energy(out):.1f}"
        )


def test_legibility_metrics_reject_a_destroyed_page() -> None:
    """Calibration: the guards must actually fail on something unreadable.

    Without this the thresholds could drift to zero and still 'pass'.
    """
    destroyed = np.full((400, 400, 3), 128, dtype=np.uint8)
    assert ink_contrast(destroyed) < MIN_LEGIBLE_CONTRAST
    assert edge_energy(destroyed) < MIN_EDGE_ENERGY
    assert not is_legible(destroyed)


def test_clean_render_passes_the_guards_comfortably() -> None:
    page = _blank_page()
    assert ink_contrast(page) > MIN_LEGIBLE_CONTRAST * 2
    assert edge_energy(page) > MIN_EDGE_ENERGY


def _sparse_page() -> np.ndarray:
    """A nursing-home bill: four line items on an otherwise blank A4 page."""
    page = np.full((1123, 794, 3), 255, dtype=np.uint8)
    for y in range(120, 220, 24):
        page[y:y + 7, 60:420] = 25
    return page


def test_a_sparse_but_perfectly_legible_page_passes() -> None:
    """The regression that cost 967 pages of the first corpus build.

    Fixed-percentile contrast measures ink DENSITY, not legibility. On a page
    that is 99% paper even the 2nd percentile is paper, so `p90 - p2` collapsed
    and rejected 36% of the CLEAN bucket. Otsu finds the split from the image, so
    a 1%-ink page and a 16%-ink page are judged on the same basis.
    """
    sparse = _sparse_page()
    assert (sparse.mean(axis=2) < 128).mean() < 0.02, "fixture is not sparse"
    assert is_legible(sparse)
    assert ink_contrast(sparse) > MIN_LEGIBLE_CONTRAST * 2


def test_sparse_and_dense_legible_pages_score_comparably() -> None:
    """The property the old metric lacked: independence from ink density."""
    sparse, dense = _sparse_page(), _blank_page()
    assert abs(ink_contrast(sparse) - ink_contrast(dense)) < 60


def test_a_destroyed_page_scores_below_every_legible_one() -> None:
    """The old metric scored a real sparse page BELOW the destroyed control."""
    from PIL import Image, ImageFilter

    dense = _blank_page()
    blurred = np.array(
        Image.fromarray(dense).filter(ImageFilter.GaussianBlur(4))
    ).astype(float)
    destroyed = (110 + (blurred / 255.0) * 40).astype(np.uint8)

    assert not is_legible(destroyed)
    assert ink_contrast(destroyed) < ink_contrast(_sparse_page())
    assert ink_contrast(destroyed) < ink_contrast(dense)


def test_augmentation_never_raises_on_odd_input() -> None:
    """One bad page must not kill a 10,000-page corpus run."""
    for shape in ((40, 40, 3), (1123, 794, 3), (200, 1400, 3)):
        out, failures = augment_page(np.full(shape, 255, np.uint8), "medium",
                                     random.Random(0))
        assert out is not None
        assert isinstance(failures, dict)


# --------------------------------------------------------------------------
# contact sheet sampling
# --------------------------------------------------------------------------

def test_contact_sheet_sampling_spreads_across_cells() -> None:
    rows = [{"layout": f"L{i%21:02d}", "quality": q, "image": "x"}
            for i in range(2000) for q in ("clean", "heavy")]
    picked = sample_stratified(rows, 100, seed=1)
    assert len(picked) == 100
    cells = {(r["layout"], r["quality"]) for r in picked}
    assert len(cells) >= 40, "sampling collapsed onto a few cells"


def test_contact_sheet_sampling_is_deterministic() -> None:
    rows = [{"layout": f"L{i%5:02d}", "quality": "clean", "image": "x"}
            for i in range(200)]
    assert sample_stratified(rows, 20, 5) == sample_stratified(rows, 20, 5)

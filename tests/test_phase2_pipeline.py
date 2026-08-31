"""Mini-set, baselines, slices, report and the make-eval entrypoint."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reckon.data.mini_set import build_mini_set, load_mini_set, write_mini_set
from reckon.eval.metrics import DocumentPair, score_line_items
from reckon.eval.report import build_report, evaluate_system, write_report
from reckon.eval.run import build_pairs, ensemble_confidence, run
from reckon.eval.slices import discover_slice_keys, group_by, slice_report
from reckon.models.baselines.b0_rules import B0RulesEngine
from reckon.models.baselines.b1_ocr_heuristic import B1OcrHeuristic
from reckon.models.baselines.ocr import PaddleOcrBackend, PlainTextBackend
from reckon.normalize import normalize_document
from reckon.schema import RawDocument


# --------------------------------------------------------------------------
# mini-set
# --------------------------------------------------------------------------

def test_mini_set_is_deterministic() -> None:
    """Same seed, byte-identical corpus, or nothing built on it is reproducible."""
    first, second = build_mini_set(), build_mini_set()
    assert [d.text for d in first] == [d.text for d in second]
    assert [d.truth.model_dump_json() for d in first] == [
        d.truth.model_dump_json() for d in second
    ]


def test_mini_set_has_twenty_documents_across_five_layouts() -> None:
    docs = build_mini_set()
    assert len(docs) == 20
    assert {d.meta["template"] for d in docs} == {"L1", "L2", "L3", "L4", "L5"}


def test_mini_set_includes_the_messiness_the_brief_asks_for() -> None:
    tags = {t for d in build_mini_set() for t in str(d.meta["messiness"]).split(",")}
    for expected in ("missing_uhid", "misaligned_totals", "duplicate_row", "handwritten"):
        assert expected in tags


def test_a_bilingual_layout_exists() -> None:
    docs = build_mini_set()
    bilingual = [d for d in docs if d.meta["language"] == "bilingual"]
    assert bilingual
    assert any(ord(ch) > 0x0C00 for ch in bilingual[0].text)  # Telugu present


def test_ground_truth_line_items_are_internally_consistent() -> None:
    """quantity x unit_rate == amount, exactly, in Decimal."""
    for doc in build_mini_set():
        for item in normalize_document(doc.truth).line_items:
            assert item.quantity * item.unit_rate == item.amount


def test_round_trip_through_disk(tmp_path: Path) -> None:
    write_mini_set(tmp_path)
    loaded = load_mini_set(tmp_path)
    built = build_mini_set()
    assert [d.doc_id for d in loaded] == [d.doc_id for d in built]
    assert loaded[0].truth == built[0].truth


def test_load_builds_the_set_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "not-yet"
    assert len(load_mini_set(target)) == 20


# --------------------------------------------------------------------------
# OCR seam
# --------------------------------------------------------------------------

def test_plain_text_backend_records_geometry() -> None:
    page = PlainTextBackend().read("alpha\n    indented\n")
    assert page.lines[0].left == 0
    assert page.lines[1].left == 4
    assert page.lines[1].top == 1


def test_paddle_backend_fails_loudly_rather_than_at_import_time() -> None:
    with pytest.raises(NotImplementedError, match="Phase 3"):
        PaddleOcrBackend().read("anything")


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def _by_template() -> dict[str, list]:
    grouped: dict[str, list] = {}
    for doc in build_mini_set():
        grouped.setdefault(str(doc.meta["template"]), []).append(doc)
    return grouped


def test_both_baselines_return_a_document_for_every_layout() -> None:
    for docs in _by_template().values():
        for engine in (B0RulesEngine(), B1OcrHeuristic()):
            assert isinstance(engine.extract(docs[0].text), RawDocument)


def test_b0_is_brittle_to_column_layout() -> None:
    """The reason v2 exists, asserted as a mechanism rather than an aggregate.

    B0's row patterns encode a fixed column order and a two-space separator, so
    on the layout they were written for (L1) they work. On the pipe-delimited
    layout the loose fallback pattern still MATCHES - and that is the genuinely
    dangerous behaviour, worse than matching nothing: the table delimiters end up
    inside the description and an unrelated number ends up in the amount, and the
    bill comes back looking plausible.
    """
    grouped = _by_template()
    b0 = B0RulesEngine()

    clean = b0.extract(grouped["L1"][0].text).line_items
    assert clean
    assert all("|" not in (i.description or "") for i in clean)

    corrupted = b0.extract(grouped["L2"][0].text).line_items
    assert corrupted, "expected B0 to match-and-corrupt, not to abstain"
    assert any("|" in (i.description or "") for i in corrupted), (
        "B0 should drag table delimiters into the description on this layout"
    )


def test_b0_corruption_is_visible_to_the_metrics() -> None:
    """Silent corruption has to cost B0 something, or the metric is useless."""
    grouped = _by_template()
    b0 = B0RulesEngine().extract
    l1 = score_line_items(build_pairs(grouped["L1"], b0))
    l2 = score_line_items(build_pairs(grouped["L2"], b0))
    assert l1.f1 > l2.f1


def test_b1_recovers_line_items_from_every_layout() -> None:
    """B1 infers column order instead of hard-coding it."""
    b1 = B1OcrHeuristic()
    for template, docs in _by_template().items():
        items = b1.extract(docs[0].text).line_items
        assert items, f"B1 found no line items in {template}"


def test_b1_beats_b0_on_line_item_recall() -> None:
    """The designed difference between the two baselines, measured.

    This asserts the mechanism holds in aggregate. If it ever flips, something
    real changed and the comparison in the README needs rewriting - so failing
    here is the correct behaviour, not a brittle test.
    """
    docs = build_mini_set()
    b0 = score_line_items(build_pairs(docs, B0RulesEngine().extract))
    b1 = score_line_items(build_pairs(docs, B1OcrHeuristic().extract))
    assert b1.recall > b0.recall


def test_baselines_never_invent_a_schema_field() -> None:
    for doc in build_mini_set()[:5]:
        for engine in (B0RulesEngine(), B1OcrHeuristic()):
            extracted = engine.extract(doc.text)
            assert isinstance(normalize_document(extracted).model_dump(), dict)


# --------------------------------------------------------------------------
# confidence proxy
# --------------------------------------------------------------------------

def test_ensemble_confidence_reflects_agreement() -> None:
    docs = build_mini_set()[:3]
    pairs = {
        "b0": build_pairs(docs, B0RulesEngine().extract),
        "b1": build_pairs(docs, B1OcrHeuristic().extract),
    }
    confidences = ensemble_confidence(pairs)

    assert set(confidences) == {d.doc_id for d in docs}
    scores = confidences[docs[0].doc_id]
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert scores["hospital.gstin"] == 1.0        # both find it, and agree
    assert scores["hospital.address"] == 0.0      # neither extracts it


def test_ensemble_confidence_is_empty_without_systems() -> None:
    assert ensemble_confidence({}) == {}


# --------------------------------------------------------------------------
# slices
# --------------------------------------------------------------------------

def test_slice_keys_are_discovered_not_hard_coded() -> None:
    docs = build_mini_set()
    pairs = build_pairs(docs, B1OcrHeuristic().extract)
    keys = discover_slice_keys(pairs)
    assert keys[:2] == ["source", "template"]
    assert "messiness" in keys      # a key nobody added to a list by hand


def test_grouping_partitions_the_pairs() -> None:
    pairs = build_pairs(build_mini_set(), B1OcrHeuristic().extract)
    groups = group_by(pairs, "template")
    assert sum(len(g) for g in groups.values()) == len(pairs)
    assert set(groups) == {"L1", "L2", "L3", "L4", "L5"}


def test_underpowered_slices_are_flagged() -> None:
    pairs = build_pairs(build_mini_set(), B1OcrHeuristic().extract)
    summaries = slice_report(pairs)["template"]
    assert all(s.n == 4 for s in summaries)
    assert all(s.underpowered for s in summaries)   # 4 < 5, correctly flagged


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _report_text() -> str:
    docs = build_mini_set()
    pairs = {
        "B0": build_pairs(docs, B0RulesEngine().extract),
        "B1": build_pairs(docs, B1OcrHeuristic().extract),
    }
    confidences = ensemble_confidence(pairs)
    results = [evaluate_system(name, p, confidences) for name, p in pairs.items()]
    return build_report(results, dataset_name="test", dataset_hash="abc")


def test_report_contains_every_required_section() -> None:
    text = _report_text()
    for heading in (
        "## Provenance", "## Headline", "## Field level, per field",
        "## Line items", "## Document level", "## Business impact",
        "## Coverage and abstention", "## Slices",
    ):
        assert heading in text, heading


def test_every_markdown_table_is_well_formed() -> None:
    """Guards against a stray pipe inside a cell silently breaking a table.

    A header reading `median |err|` renders as two extra empty columns, which is
    exactly the sort of defect that survives review because the file still looks
    fine in a terminal.
    """
    lines = _report_text().splitlines()
    index = 0
    tables_checked = 0
    while index < len(lines):
        line = lines[index].strip()
        is_header = line.startswith("|") and line.endswith("|")
        separator = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if is_header and re.fullmatch(r"\|(?:---\|)+", separator):
            width = line.count("|")
            assert separator.count("|") == width, f"separator width at line {index}"
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                assert lines[cursor].strip().count("|") == width, (
                    f"row {cursor} has {lines[cursor].count('|')} pipes, header has {width}"
                )
                cursor += 1
            tables_checked += 1
            index = cursor
        else:
            index += 1
    assert tables_checked >= 6


def test_report_states_the_mini_set_caveats() -> None:
    """The caveats are part of the result, not a disclaimer to be dropped."""
    from reckon.eval.run import MINI_SET_NOTES

    docs = build_mini_set()
    results = [evaluate_system("B0", build_pairs(docs, B0RulesEngine().extract))]
    text = build_report(results, dataset_name="mini", notes=MINI_SET_NOTES)
    assert "not a benchmark" in text
    assert "OCR stage is never exercised" in text


def test_report_never_shows_one_system_alone_in_the_headline() -> None:
    text = _report_text()
    headline = text.split("## Headline")[1].split("## Field")[0]
    assert "B0" in headline and "B1" in headline


def test_write_report_names_the_file_after_the_git_sha(tmp_path: Path) -> None:
    docs = build_mini_set()[:2]
    results = [evaluate_system("B0", build_pairs(docs, B0RulesEngine().extract))]
    path = write_report(results, dataset_name="t", reports_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("eval_")
    assert path.read_text(encoding="utf-8").startswith("# RECKON evaluation")


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_make_eval_runs_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "mini"
    reports_dir = tmp_path / "reports"
    path = run(data_dir=data_dir, reports_dir=reports_dir)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "B0 (v1 rules)" in text
    assert "B1 (OCR + heuristics)" in text
    assert "dataset SHA-256" in text


def test_eval_is_reproducible_across_runs(tmp_path: Path) -> None:
    """Two runs of the same code over the same data must agree on every number."""
    docs = build_mini_set()
    first = build_pairs(docs, B1OcrHeuristic().extract)
    second = build_pairs(docs, B1OcrHeuristic().extract)
    assert score_line_items(first) == score_line_items(second)

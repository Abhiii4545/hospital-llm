"""Serving layer: API behaviour, export variants and the latency table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reckon.serve.api import align_for_display, create_app
from reckon.serve.bench_latency import LatencyResult, measure, render_table
from reckon.serve.export_onnx import VARIANTS, export
from reckon.schema import RawDocument, RawPatient, RawTotals

DOCUMENT = {
    "patient": {"ward_type": "Deluxe Room"},
    "line_items": [
        {"description": "Room Rent - Deluxe", "quantity": "1",
         "unit_rate": "10000.00", "amount": "10000.00", "category": "Room Rent"},
        {"description": "Nursing Charges", "amount": "2000.00", "category": "Nursing"},
        {"description": "Attendant Charges", "amount": "800.00",
         "category": "Non-Medical"},
        {"description": "Surgeon Fee", "amount": "50000.00", "category": "Surgery"},
    ],
    "totals": {"gross_amount": "62800.00", "net_amount": "62800.00"},
}


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("RECKON_CORRECTIONS", str(tmp_path / "corrections.jsonl"))
    import importlib

    import reckon.serve.api as api

    importlib.reload(api)
    return TestClient(api.create_app())


# --------------------------------------------------------------------------
# the half that works with no model
# --------------------------------------------------------------------------

def test_health_is_honest_about_what_is_available() -> None:
    payload = TestClient(create_app()).get("/health").json()
    assert payload["adjudication_available"] is True
    assert payload["extraction_available"] is False   # no checkpoint configured


def test_adjudicate_runs_without_a_model_or_gpu() -> None:
    response = TestClient(create_app()).post(
        "/adjudicate", json={"document": DOCUMENT, "co_pay_percent": 0}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["payable"] == "31000.00"
    assert payload["total_deducted"] == "31800.00"


def test_every_deduction_returned_is_traceable() -> None:
    payload = TestClient(create_app()).post(
        "/adjudicate", json={"document": DOCUMENT}
    ).json()
    assert payload["deductions"]
    for deduction in payload["deductions"]:
        assert deduction["rule_id"] and deduction["clause"] and deduction["reason"]
    assert "IRDAI List I" in payload["audit_trail"]


def test_policy_overrides_change_the_answer() -> None:
    client = TestClient(create_app())
    strict = client.post("/adjudicate",
                         json={"document": DOCUMENT, "co_pay_percent": 30}).json()
    lenient = client.post("/adjudicate",
                          json={"document": DOCUMENT, "co_pay_percent": 0}).json()
    assert float(strict["payable"]) < float(lenient["payable"])


def test_malformed_document_gets_422_not_500() -> None:
    response = TestClient(create_app()).post(
        "/adjudicate", json={"document": {"line_items": "not a list"}}
    )
    assert response.status_code == 422


def test_extract_without_a_checkpoint_says_so_clearly() -> None:
    response = TestClient(create_app()).post(
        "/extract", files={"files": ("page.png", b"\x89PNG")}
    )
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].casefold()
    assert "/adjudicate" in response.json()["detail"]


def test_review_ui_is_served() -> None:
    response = TestClient(create_app()).get("/review")
    assert response.status_code == 200
    assert "RECKON" in response.text


# --------------------------------------------------------------------------
# human-in-the-loop corrections
# --------------------------------------------------------------------------

def test_corrections_are_logged_as_future_training_data(client, tmp_path) -> None:
    response = client.post("/corrections", json={
        "page_id": "syn_000001_p00",
        "field": "totals.net_amount",
        "predicted": "1,25,678.20",
        "corrected": "1,25,687.20",
    })
    assert response.status_code == 200

    path = Path(response.json()["path"])
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["field"] == "totals.net_amount"
    assert record["corrected"] == "1,25,687.20"
    assert record["logged_at"]
    assert "git_sha" in record         # provenance travels with the label


def test_corrections_append_rather_than_overwrite(client) -> None:
    for index in range(3):
        client.post("/corrections", json={
            "page_id": f"p{index}", "field": "patient.name", "corrected": "X"})
    path = Path(client.post("/corrections", json={
        "page_id": "p9", "field": "patient.name", "corrected": "Y"}).json()["path"])
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 4


# --------------------------------------------------------------------------
# display-only alignment
# --------------------------------------------------------------------------

def test_alignment_matches_fields_to_ocr_boxes() -> None:
    document = RawDocument(
        patient=RawPatient(name="Ramesh Kumar"),
        totals=RawTotals(net_amount="1,25,678.20"),
    )
    ocr = [
        {"text": "Patient Name : Ramesh Kumar", "box": [10, 20, 300, 40]},
        {"text": "Net Amount 1,25,678.20", "box": [400, 700, 700, 720]},
        {"text": "irrelevant", "box": [0, 0, 5, 5]},
    ]
    aligned = align_for_display(document, ocr)
    fields = {a["field"]: a for a in aligned}
    assert fields["patient.name"]["box"] == [10, 20, 300, 40]
    assert fields["totals.net_amount"]["box"] == [400, 700, 700, 720]


def test_alignment_drops_weak_matches() -> None:
    document = RawDocument(patient=RawPatient(name="Ramesh Kumar"))
    assert align_for_display(document, [{"text": "zzzz", "box": [0, 0, 1, 1]}]) == []


def test_alignment_is_documented_as_presentation_only() -> None:
    """The brief asks for this to be said in a comment where it happens."""
    source = Path("reckon/serve/api.py").read_text(encoding="utf-8")
    assert "PRESENTATION ONLY" in source
    assert "never used to compute a metric" in source


# --------------------------------------------------------------------------
# export variants and the trade-off table
# --------------------------------------------------------------------------

def test_three_variants_are_declared() -> None:
    names = {v.name for v in VARIANTS}
    assert names == {"onnx-fp32", "onnx-fp16", "onnx-int8-dynamic"}
    for variant in VARIANTS:
        assert variant.accuracy_fields    # accuracy slot is mandatory


def test_export_without_a_checkpoint_fails_usefully(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Train Head B"):
        export(tmp_path / "nope", tmp_path / "out", VARIANTS[0])


def test_latency_percentiles_discard_warmup() -> None:
    calls: list[int] = []

    def run_one(_):
        calls.append(1)

    p50, p95, mean = measure(run_one, list(range(20)), warmup=3)
    assert len(calls) == 23          # 3 warm-up + 20 measured
    assert p50 >= 0 and p95 >= p50 and mean >= 0


def test_table_refuses_to_hide_a_missing_accuracy_column() -> None:
    """A speed number without its accuracy cost gets quoted as though free."""
    table = render_table([
        LatencyResult("onnx-int8-dynamic", "cpu", 20, 120.0, 180.0, 130.0, 210.0),
    ])
    assert "**not measured**" in table
    assert "not a result" in table


def test_table_renders_measured_accuracy_when_present() -> None:
    table = render_table([
        LatencyResult("onnx-fp32", "cpu", 20, 400.0, 520.0, 430.0, 810.0,
                      accuracy={"field_exact_normalized": 0.91,
                                "line_item_f1": 0.88,
                                "median_rupee_error": 12.5}),
    ])
    assert "91.0%" in table and "88.0%" in table and "Rs 12.50" in table
    assert "not measured" not in table

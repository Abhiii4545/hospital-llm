"""FastAPI application.

Three things are served:

* ``/adjudicate`` - the deterministic rules engine. Runs with **no model and no
  GPU**, so the adjudication half of the system is demonstrable today.
* ``/extract`` - the Donut heads. Returns 503 with a clear message when no
  checkpoint is configured, rather than pretending.
* ``/review`` - the human-in-the-loop correction UI, and ``/corrections`` which
  logs every correction as future training data.

The serving layer may not import ``reckon.training`` or ``reckon.data`` - the
import contracts enforce it - so inference loads a checkpoint directory directly
and shares the target format through the top-level ``serialize`` leaf.

Bounding boxes: Donut produces none. When a review UI needs to highlight a field
on the page, a cheap OCR pass supplies coordinates and predicted strings are
fuzzy-aligned back to them. **That alignment is presentation-only. It never feeds
the model and never touches a metric.** See ``align_for_display`` below.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from reckon.adjudicate.engine import adjudicate, load_policy, load_rules
from reckon.models.assemble import PageFragment, assemble
from reckon.normalize import normalize_document
from reckon.provenance import run_metadata
from reckon.schema import Document, RawDocument
from reckon.serialize import parse_head_a, parse_head_b

__all__ = ["app", "create_app", "Extractor", "align_for_display"]

CHECKPOINT_ENV = "RECKON_CHECKPOINT"
CORRECTIONS_PATH = Path(os.environ.get("RECKON_CORRECTIONS", "data/corrections.jsonl"))


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

@dataclass
class Extractor:
    """Lazily-loaded Donut heads. Absent checkpoints are reported, not faked."""

    head_a_dir: Path | None = None
    head_b_dir: Path | None = None
    _loaded: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._loaded = {}

    @property
    def available(self) -> bool:
        return bool(self.head_b_dir and Path(self.head_b_dir).exists())

    def _load(self, key: str, directory: Path):
        if key in self._loaded:
            return self._loaded[key]
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(directory)
        model = VisionEncoderDecoderModel.from_pretrained(directory).eval()
        self._loaded[key] = (processor, model)
        return self._loaded[key]

    def run_page(self, image, head: str) -> str:
        import torch

        directory = self.head_b_dir if head == "b" else self.head_a_dir
        if not directory:
            raise HTTPException(503, f"no checkpoint configured for head {head}")
        processor, model = self._load(head, Path(directory))

        pixel_values = processor(image, return_tensors="pt").pixel_values
        start = processor.tokenizer.convert_tokens_to_ids(
            "<s_line_items>" if head == "b" else "<s_hospital>"
        )
        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                max_length=model.config.decoder.max_length,
                decoder_start_token_id=start,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        return processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]


def align_for_display(
    document: RawDocument, ocr_lines: list[dict]
) -> list[dict]:
    """Fuzzy-align predicted strings to OCR boxes so a UI can highlight them.

    PRESENTATION ONLY. Donut emits no bounding boxes, so these coordinates come
    from a separate cheap OCR pass. They are never fed back into the model and
    are never used to compute a metric - if they were, the evaluation would be
    silently measuring OCR quality instead of the model's.
    """
    from rapidfuzz import fuzz

    out: list[dict] = []
    payload = document.model_dump()
    for block, fields in payload.items():
        if not isinstance(fields, dict):
            continue
        for name, value in fields.items():
            if not value:
                continue
            best, score = None, 0.0
            for line in ocr_lines:
                ratio = fuzz.partial_ratio(str(value).casefold(),
                                           str(line.get("text", "")).casefold())
                if ratio > score:
                    best, score = line, ratio
            if best and score >= 75:
                out.append({"field": f"{block}.{name}", "value": value,
                            "box": best.get("box"), "confidence": score / 100})
    return out


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------

class AdjudicateRequest(BaseModel):
    document: dict
    sum_insured: float | None = None
    co_pay_percent: float | None = None
    deductible: float | None = None


class CorrectionRequest(BaseModel):
    page_id: str
    field: str
    predicted: str | None = None
    corrected: str
    note: str | None = None


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

def create_app(extractor: Extractor | None = None) -> FastAPI:
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    extractor = extractor or Extractor(
        head_b_dir=Path(checkpoint) if checkpoint else None
    )
    application = FastAPI(
        title="RECKON v2",
        description="Document understanding and IRDAI adjudication for Indian "
                    "hospital bills.",
        version="2.0.0.dev0",
    )

    @application.get("/health")
    def health() -> dict:
        meta = run_metadata()
        return {
            "status": "ok",
            "git_sha": meta["git_sha"],
            "extraction_available": extractor.available,
            "adjudication_available": True,
            "note": (
                "adjudication runs without a model or GPU; extraction requires a "
                f"checkpoint via ${CHECKPOINT_ENV}"
            ),
        }

    @application.post("/adjudicate")
    def adjudicate_endpoint(request: AdjudicateRequest) -> dict:
        """Deterministic. Works today, with no model."""
        try:
            raw = RawDocument.model_validate(request.document)
            typed: Document = normalize_document(raw)
        except Exception as error:                          # noqa: BLE001
            raise HTTPException(422, f"could not parse document: {error}") from error

        overrides = {
            key: value for key, value in (
                ("sum_insured", request.sum_insured),
                ("co_pay_percent", request.co_pay_percent),
                ("deductible", request.deductible),
            ) if value is not None
        }
        result = adjudicate(typed, load_rules(), load_policy(**overrides))
        return {
            "gross": str(result.gross),
            "payable": str(result.payable),
            "total_deducted": str(result.total_deducted),
            "deductions": [
                {
                    "rule_id": d.rule_id, "clause": d.clause, "reason": d.reason,
                    "amount": str(d.amount), "line_index": d.line_index,
                    "line_description": d.line_description,
                }
                for d in result.deductions
            ],
            "by_rule": {k: str(v) for k, v in result.by_rule().items()},
            "notes": result.notes,
            "audit_trail": result.explain(),
        }

    @application.post("/extract")
    async def extract(files: list[UploadFile] = File(...)) -> dict:
        if not extractor.available:
            raise HTTPException(
                503,
                "No extraction checkpoint is configured. Train Head B and set "
                f"${CHECKPOINT_ENV} to its directory. /adjudicate works without one.",
            )
        from PIL import Image

        started = time.perf_counter()
        fragments: list[PageFragment] = []
        for index, upload in enumerate(sorted(files, key=lambda f: f.filename or "")):
            image = Image.open(await_bytes(await upload.read())).convert("RGB")
            head_b = parse_head_b(extractor.run_page(image, "b"))
            head_a = parse_head_a(extractor.run_page(image, "a"))
            fragments.append(PageFragment(
                page_index=index,
                hospital=head_a.hospital, patient=head_a.patient,
                insurance=head_a.insurance, totals=head_a.totals,
                line_items=head_b,
            ))

        assembled = assemble(fragments)
        elapsed = time.perf_counter() - started
        return {
            "document": assembled.document.model_dump(),
            "reconciliation": {
                "balanced": assembled.report.balanced,
                "complete": assembled.report.complete,
                "flags": assembled.report.flags,
                "duplicates_removed": assembled.report.duplicates_removed,
            },
            "pages": len(fragments),
            "latency_seconds": round(elapsed, 3),
        }

    @application.post("/corrections")
    def log_correction(request: CorrectionRequest) -> dict:
        """Log a human correction as future training data.

        This is the loop that makes the review UI worth building: every field a
        reviewer fixes is a labelled example from the real distribution, which is
        the distribution the synthetic corpus is only approximating.
        """
        CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **request.model_dump(),
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": run_metadata()["git_sha"],
        }
        with open(CORRECTIONS_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"logged": True, "path": str(CORRECTIONS_PATH)}

    @application.get("/review", response_class=HTMLResponse)
    def review() -> str:
        return REVIEW_HTML

    @application.get("/")
    def root() -> JSONResponse:
        return JSONResponse({
            "service": "RECKON v2",
            "endpoints": ["/health", "/adjudicate", "/extract", "/review",
                          "/corrections", "/docs"],
        })

    return application


def await_bytes(data: bytes):
    from io import BytesIO

    return BytesIO(data)


REVIEW_HTML = """<!doctype html>
<meta charset="utf-8"><title>RECKON review</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#12151b;color:#e8ecf1}
 header{padding:14px 20px;border-bottom:1px solid #2a2f3a;background:#171b22}
 h1{font-size:16px;margin:0}
 main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
 section{background:#171b22;border:1px solid #2a2f3a;border-radius:8px;padding:14px}
 h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#8fa2bd;margin:0 0 10px}
 textarea{width:100%;height:260px;background:#0e1116;color:#dfe6ee;border:1px solid #2a2f3a;
   border-radius:6px;padding:10px;font-family:ui-monospace,monospace;font-size:12px}
 button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:8px 14px;
   font-size:13px;cursor:pointer}
 button:hover{background:#2c7bd0}
 pre{white-space:pre-wrap;background:#0e1116;border:1px solid #2a2f3a;border-radius:6px;
   padding:10px;font-size:12px;max-height:420px;overflow:auto}
 .row{display:flex;gap:8px;align-items:center;margin-top:8px}
 input{flex:1;background:#0e1116;color:#dfe6ee;border:1px solid #2a2f3a;border-radius:6px;padding:7px}
 .muted{color:#8fa2bd;font-size:12px}
</style>
<header><h1>RECKON v2 &mdash; review &amp; adjudication</h1>
<div class="muted">Adjudication runs with no model and no GPU. Extraction needs a checkpoint.</div>
</header>
<main>
 <section>
  <h2>Document JSON</h2>
  <textarea id="doc">{
  "patient": {"ward_type": "Deluxe Room"},
  "line_items": [
    {"description": "Room Rent - Deluxe", "quantity": "1", "unit_rate": "10000.00", "amount": "10000.00", "category": "Room Rent"},
    {"description": "Nursing Charges", "amount": "2000.00", "category": "Nursing"},
    {"description": "Attendant Charges", "amount": "800.00", "category": "Non-Medical"},
    {"description": "Surgeon Fee", "amount": "50000.00", "category": "Surgery"}
  ],
  "totals": {"gross_amount": "62800.00", "net_amount": "62800.00"}
 }</textarea>
  <div class="row"><button onclick="run()">Adjudicate</button>
   <span class="muted" id="status"></span></div>
 </section>
 <section>
  <h2>Result</h2>
  <pre id="out">Press Adjudicate.</pre>
 </section>
 <section>
  <h2>Log a correction (future training data)</h2>
  <div class="row"><input id="page" placeholder="page id"></div>
  <div class="row"><input id="field" placeholder="field path e.g. totals.net_amount"></div>
  <div class="row"><input id="pred" placeholder="predicted"></div>
  <div class="row"><input id="corr" placeholder="corrected"></div>
  <div class="row"><button onclick="logCorrection()">Log correction</button>
   <span class="muted" id="cstatus"></span></div>
 </section>
 <section>
  <h2>Why corrections are logged</h2>
  <p class="muted">Every field a reviewer fixes is a labelled example drawn from
  the real distribution &mdash; the one the synthetic corpus is only
  approximating. These land in <code>data/corrections.jsonl</code> and are the
  cheapest source of real training data this system has.</p>
 </section>
</main>
<script>
async function run(){
  const s=document.getElementById('status'); s.textContent='working...';
  try{
    const r=await fetch('/adjudicate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({document:JSON.parse(document.getElementById('doc').value)})});
    const j=await r.json();
    document.getElementById('out').textContent = j.audit_trail || JSON.stringify(j,null,2);
    s.textContent = r.ok ? '' : 'error';
  }catch(e){ document.getElementById('out').textContent=e; s.textContent='error'; }
}
async function logCorrection(){
  const s=document.getElementById('cstatus');
  const r=await fetch('/corrections',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({page_id:page.value,field:field.value,predicted:pred.value,corrected:corr.value})});
  s.textContent = r.ok ? 'logged' : 'failed';
}
</script>
"""

app = create_app()

"""Render a generated document to page images via Jinja2 + Playwright.

WeasyPrint is the brief's first choice but needs GTK/Pango natives that are not
present on this machine, so Playwright (Apache-2.0) drives headless Chromium
instead - the brief's stated alternative. Chromium also gives correct shaping for
the Telugu and Devanagari headers in the government and bilingual layouts, which
matters because those scripts are part of the layout variance being modelled.

One browser is launched per Renderer and reused across pages. Launching per page
costs about a second each, which at 10,000 pages is three wasted hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from reckon.data.generators.document import GeneratedDocument, Page
from reckon.data.layouts import LayoutSpec
from reckon.normalize import normalize_amount

__all__ = ["Renderer", "PAGE_WIDTH", "PAGE_HEIGHT", "column_labels"]

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: A4 at ~96dpi. Downscaled to the model's input resolution later; rendering
#: larger and downscaling gives cleaner glyph edges than rendering small.
PAGE_WIDTH = 794
PAGE_HEIGHT = 1123

_FONTS = {
    "serif": "Georgia, 'Times New Roman', 'Nirmala UI', serif",
    "sans": "'Segoe UI', Arial, 'Nirmala UI', sans-serif",
    "mono": "'Courier New', 'Nirmala UI', monospace",
    "condensed": "'Arial Narrow', 'Liberation Sans Narrow', 'Nirmala UI', sans-serif",
}

_BASE_SIZE = {"corporate": 12, "grouped": 12, "nursing_home": 13,
              "government": 12, "diagnostic": 12, "pharmacy": 11, "discharge": 12}

#: Column header wording differs by archetype. A model that memorised one
#: spelling of "Description" would not survive a real corpus.
_LABELS: dict[str, dict[str, str]] = {
    "corporate": {"serial_no": "S.No", "description": "Description", "service_date": "Date",
                  "category": "Category", "quantity": "Qty", "unit_rate": "Rate",
                  "amount": "Amount", "hsn_code": "HSN", "batch": "Batch", "expiry": "Exp"},
    "grouped": {"serial_no": "Sr", "description": "Particulars", "service_date": "Date",
                "category": "Head", "quantity": "Qty", "unit_rate": "Rate",
                "amount": "Amount", "hsn_code": "HSN", "batch": "Batch", "expiry": "Exp"},
    "nursing_home": {"serial_no": "#", "description": "Item", "service_date": "Date",
                     "category": "Type", "quantity": "Qty", "unit_rate": "Rate",
                     "amount": "Amount", "hsn_code": "HSN", "batch": "Batch", "expiry": "Exp"},
    "government": {"serial_no": "క్ర.సం / S.No", "description": "వివరాలు / Particulars",
                   "service_date": "తేదీ / Date", "category": "Category",
                   "quantity": "సంఖ్య / Qty", "unit_rate": "ధర / Rate",
                   "amount": "మొత్తం / Amount", "hsn_code": "HSN",
                   "batch": "Batch", "expiry": "Exp"},
    "diagnostic": {"serial_no": "#", "description": "Test / Panel", "service_date": "Date",
                   "category": "Section", "quantity": "Qty", "unit_rate": "Rate",
                   "amount": "Amount", "hsn_code": "HSN", "batch": "Batch", "expiry": "Exp"},
    "pharmacy": {"serial_no": "S.No", "description": "Drug Name", "service_date": "Date",
                 "category": "Type", "quantity": "Qty", "unit_rate": "MRP",
                 "amount": "Amount", "hsn_code": "HSN", "batch": "Batch No", "expiry": "Exp."},
    "discharge": {"serial_no": "S.No", "description": "Charge Head", "service_date": "Date",
                  "category": "Category", "quantity": "Qty", "unit_rate": "Rate",
                  "amount": "Amount", "hsn_code": "HSN", "batch": "Batch", "expiry": "Exp"},
}

#: Bilingual header strings. Telugu for the southern layouts, Devanagari Hindi
#: for the rest, chosen per document so both scripts appear in the corpus.
_BILINGUAL = {
    "telugu": {
        "govt": "ప్రభుత్వ ఆసుపత్రి / GOVERNMENT HOSPITAL",
        "patient_details": "రోగి వివరాలు / PATIENT DETAILS",
        "name": "పేరు / Name", "age": "వయస్సు / Age", "sex": "లింగం / Sex",
        "ward": "వార్డు / Ward", "admission": "చేరిన తేదీ / Admission",
        "discharge": "డిశ్చార్జి / Discharge", "details": "వివరాలు / DETAILS",
        "discharge_summary": "డిశ్చార్జి సారాంశం / DISCHARGE SUMMARY",
    },
    "hindi": {
        "govt": "राजकीय चिकित्सालय / GOVERNMENT HOSPITAL",
        "patient_details": "रोगी विवरण / PATIENT DETAILS",
        "name": "नाम / Name", "age": "आयु / Age", "sex": "लिंग / Sex",
        "ward": "वार्ड / Ward", "admission": "भर्ती / Admission",
        "discharge": "छुट्टी / Discharge", "details": "विवरण / DETAILS",
        "discharge_summary": "डिस्चार्ज सारांश / DISCHARGE SUMMARY",
    },
}

_DIAGNOSES = (
    "Acute gastroenteritis with moderate dehydration",
    "Community acquired pneumonia, right lower lobe",
    "Type 2 diabetes mellitus with diabetic foot ulcer",
    "Acute appendicitis, post appendicectomy",
    "Cholelithiasis, post laparoscopic cholecystectomy",
    "Unstable angina, post coronary angiography",
    "Lower respiratory tract infection with bronchospasm",
    "Dengue fever with thrombocytopenia",
)


def column_labels(archetype: str) -> dict[str, str]:
    return _LABELS[archetype]


def _display_rows(spec: LayoutSpec, page: Page, formatter: Any) -> list[dict[str, str]]:
    """Rows as PRINTED, including any group headings and subtotals.

    Group and subtotal rows are display-only decoration. They are inserted into a
    copy, never into ``page.items``, so the per-page ground truth stays exactly
    the set of real line items - a subtotal row scored as a line item would be a
    label bug affecting every grouped layout.
    """
    if not spec.grouped_subtotals:
        return page.rows

    ordered: list[dict[str, str]] = []
    seen: dict[str, list[dict[str, str]]] = {}
    for row in page.rows:
        seen.setdefault(row.get("category", "Other"), []).append(row)

    for category, rows in seen.items():
        ordered.append({"__group__": category})
        ordered.extend(rows)
        total = sum(
            (normalize_amount(r["amount"]) or 0) for r in rows
        )
        ordered.append({"__subtotal__": f"Subtotal - {category}",
                        "amount": formatter.amount(total)})
    return ordered


@dataclass
class Renderer:
    """Jinja + headless Chromium. Use as a context manager."""

    scale: float = 1.0

    def __enter__(self) -> "Renderer":
        from playwright.sync_api import sync_playwright

        self._env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            undefined=StrictUndefined,
            autoescape=True,
        )
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(args=["--force-color-profile=srgb"])
        self._page = self._browser.new_page(
            viewport={"width": PAGE_WIDTH, "height": PAGE_HEIGHT},
            device_scale_factor=self.scale,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._page.close()
        self._browser.close()
        self._pw.stop()

    def html(self, doc: GeneratedDocument, spec: LayoutSpec, page: Page) -> str:
        # The script follows the hospital's actual state, decided at generation
        # time, so a bilingual header is never in the wrong language for its city.
        script = doc.context.get("script", "hindi")
        template = self._env.get_template(spec.template)
        return template.render(
            ctx=doc.context,
            spec=spec,
            page=page,
            rows=_display_rows(spec, page, doc.formatter),
            labels=column_labels(spec.archetype),
            fontstack=_FONTS[spec.font],
            base_size=_BASE_SIZE[spec.archetype],
            table_style=spec.table_style,
            bi=_BILINGUAL[script],
            diagnosis=_DIAGNOSES[hash(doc.doc_id) % len(_DIAGNOSES)],
            stamped=spec.archetype in {"nursing_home", "government"},
        )

    def render_page(
        self, doc: GeneratedDocument, spec: LayoutSpec, page: Page, out_path: Path
    ) -> Path:
        self._page.set_content(self.html(doc, spec, page), wait_until="load")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._page.screenshot(path=str(out_path), full_page=False)
        return out_path

    def render_document(
        self, doc: GeneratedDocument, spec: LayoutSpec, out_dir: Path
    ) -> Iterator[tuple[Page, Path]]:
        for page in doc.pages:
            path = out_dir / f"{doc.doc_id}_p{page.index:02d}.png"
            yield page, self.render_page(doc, spec, page, path)

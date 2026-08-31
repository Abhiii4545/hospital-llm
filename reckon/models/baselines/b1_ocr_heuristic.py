"""B1 - OCR plus positional heuristics.

What a competent engineer builds in a week without machine learning, and the
baseline that actually has to be beaten. Where B0 hard-codes column ORDER, B1
infers it:

* header fields are found by fuzzy-matching a label against known aliases, so
  "Patient Name", "Name of Patient" and "Pt. Name" all resolve without a new
  entry being added by hand;
* the line-item table is located by finding a header row that mentions several
  known column names, and the cell-to-column mapping is read off THAT row, so a
  hospital that puts rate before quantity is handled;
* when no header row exists, it falls back to geometry - the rightmost numeric
  token on a row is the amount, the one before it the rate, and so on.

This is a genuinely stronger design than B0, and it is still fundamentally a pile
of heuristics. The interesting question the project has to answer is how much
better a trained model is than THIS, not than B0.
"""

from __future__ import annotations

import re
from typing import Sequence

from rapidfuzz import fuzz

from reckon.models.baselines.ocr import OcrBackend, OcrLine, OcrPage, PlainTextBackend
from reckon.schema import (
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = ["B1OcrHeuristic", "extract"]

#: Field path -> label aliases. Fuzzy-matched, so near spellings also resolve.
_ALIASES: dict[str, tuple[str, ...]] = {
    "hospital.gstin": ("gstin", "gst no", "gst number"),
    "patient.name": ("patient name", "name of patient", "patient", "name", "pt name"),
    "patient.age": ("age", "age / sex", "age/sex", "vayassu / age"),
    "patient.sex": ("sex", "gender", "lingam / sex"),
    "patient.uhid": ("uhid", "uhid no", "mr no", "hospital no", "reg no"),
    "patient.ip_number": ("ip no", "ip number", "ipd no", "admission no"),
    "patient.admission_date": ("admission", "date of admission", "doa", "adm", "admitted"),
    "patient.discharge_date": ("discharge", "date of discharge", "dod", "dis", "discharged"),
    "patient.ward_type": ("ward type", "ward", "room category", "bed type", "room type"),
    "insurance.insurer_name": ("insurer", "insurance company", "insurance"),
    "insurance.tpa_name": ("tpa", "tpa name"),
    "insurance.policy_number": ("policy no", "policy number", "policy"),
    "insurance.claim_number": ("claim no", "claim number", "claim"),
    "insurance.employee_id": ("employee id", "emp id", "emp", "employee no"),
    "totals.gross_amount": ("gross amount", "gross", "total amount", "sub total"),
    "totals.discount": ("discount", "concession"),
    "totals.cgst": ("cgst", "c gst"),
    "totals.sgst": ("sgst", "s gst"),
    "totals.net_amount": ("net amount", "net payable", "grand total", "net"),
    "totals.advance_paid": ("advance paid", "advance", "paid"),
    "totals.balance_due": ("balance due", "balance", "due"),
    "totals.amount_in_words": ("amount in words", "in words"),
}

#: Column header vocabulary. Used both to FIND the table and to map its cells.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "serial_no": ("s.no", "sno", "sr", "sr no", "#", "si no", "serial"),
    "description": ("description", "particulars", "test / panel", "test", "panel",
                    "item", "service", "details", "narration"),
    "service_date": ("date", "service date", "dos"),
    "quantity": ("qty", "quantity", "nos", "units"),
    "unit_rate": ("rate", "unit rate", "price", "mrp"),
    "amount": ("amount", "value", "total"),
}

_NUMBER = re.compile(r"^(?:Rs\.?|INR|₹)?\s*-?[\d,]+(?:\.\d{1,2})?(?:/-)?$", re.IGNORECASE)
_DATE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_TOTAL_WORDS = re.compile(
    r"^\s*(gross|net|total|discount|cgst|sgst|advance|balance|grand)", re.IGNORECASE
)
_FUZZ_FLOOR = 82

#: Column headers get a lower floor than field labels. They are short, so a
#: single dropped character costs a lot of ratio - real OCR read "Qty" as "Qt",
#: which scores 80 and was being rejected, losing the quantity column on every
#: page. Short strings are also less prone to false matches, so the looser floor
#: is safe here in a way it would not be for a long field label.
_COLUMN_FUZZ_FLOOR = 74


def _split_cells(text: str) -> list[str]:
    """Split a row into cells on pipes, or on runs of two or more spaces."""
    if "|" in text:
        return [cell.strip() for cell in text.split("|")]
    return [cell.strip() for cell in re.split(r"\s{2,}", text.strip()) if cell.strip()]


#: Aliases shorter than this are matched whole-string only. A 3-character alias
#: like "adm" or "doa" will partial-match almost anything.
_MIN_PARTIAL_ALIAS = 5


def _best_alias(token: str, aliases: Sequence[str]) -> int:
    """Best fuzzy score of *token* against any alias.

    Whole-string ratio alone fails on a real page. OCR reads the printed label
    "Admission Date", which scores 78 against the alias "admission" purely
    because of the extra word - so the field fell through to a worse match and
    came out as the literal string "Date".

    Partial ratio is therefore also considered, but only for aliases long enough
    that a substring hit means something. Short aliases stay whole-string.
    """
    token = token.strip().strip(":").casefold()
    if not token:
        return 0
    best = 0
    for alias in aliases:
        best = max(best, fuzz.ratio(token, alias))
        if len(alias) >= _MIN_PARTIAL_ALIAS:
            best = max(best, fuzz.partial_ratio(token, alias))
    return best


def _looks_numeric(cell: str) -> bool:
    return bool(_NUMBER.match(cell.strip()))


class B1OcrHeuristic:
    """OCR plus positional heuristics."""

    name = "B1 (OCR + heuristics)"

    def __init__(self, backend: OcrBackend | None = None) -> None:
        self.backend = backend or PlainTextBackend()

    def extract(self, source: object) -> RawDocument:
        page: OcrPage = self.backend.read(source)
        lines = page.lines
        values = self._header_fields(lines)

        return RawDocument(
            hospital=RawHospital(
                name=self._hospital_name(lines),
                address=None, city=None, state=None,
                gstin=self._gstin(lines, values.get("hospital.gstin")),
                hospital_type=None,
            ),
            patient=RawPatient(
                name=values.get("patient.name"),
                age=values.get("patient.age"),
                sex=self._sex(values),
                uhid=values.get("patient.uhid"),
                ip_number=values.get("patient.ip_number"),
                admission_date=values.get("patient.admission_date"),
                discharge_date=values.get("patient.discharge_date"),
                ward_type=values.get("patient.ward_type"),
            ),
            insurance=RawInsurance(
                insurer_name=values.get("insurance.insurer_name"),
                tpa_name=values.get("insurance.tpa_name"),
                policy_number=values.get("insurance.policy_number"),
                claim_number=values.get("insurance.claim_number"),
                employee_id=values.get("insurance.employee_id"),
            ),
            line_items=self._rows(lines),
            totals=RawTotals(
                **{
                    path.split(".", 1)[1]: values.get(path)
                    for path in _ALIASES
                    if path.startswith("totals.")
                }
            ),
        )

    # -- header fields ---------------------------------------------------

    def _header_fields(self, lines: Sequence[OcrLine]) -> dict[str, str]:
        """Label/value pairs, keeping the BEST-scoring candidate per field.

        First-match-wins was wrong. Two passes generate candidates - adjacent
        column chunks, and label/value inside one chunk - and the second pass
        splits "Patient Name" into ("Patient", "Name"), which scores 100 against
        the alias "patient". On a real page that produced `age` = "Patient Name"
        and `gross_amount` = "Date": a weak candidate arriving first beat the
        right one arriving second.

        A candidate whose VALUE is itself a strong label is also rejected -
        "Date" is a column header, not a total.
        """
        best: dict[str, tuple[int, str]] = {}
        for line in lines:
            for label, value in self._pairs(line.text):
                cleaned = value.strip().strip(":").strip()
                if not cleaned or cleaned in {"-", "--", "N/A"}:
                    continue
                for path, aliases in _ALIASES.items():
                    score = _best_alias(label, aliases)
                    if score < _FUZZ_FLOOR:
                        continue
                    if self._looks_like_a_label(cleaned):
                        continue
                    if path not in best or score > best[path][0]:
                        best[path] = (score, cleaned)
        return {path: value for path, (_, value) in best.items()}

    @staticmethod
    def _looks_like_a_label(value: str) -> bool:
        """True when a candidate VALUE is really another field's label.

        Guards against reading the next column header as the value, which is how
        `gross_amount` became the string "Date".
        """
        for aliases in _ALIASES.values():
            if _best_alias(value, aliases) >= 92:
                return True
        for aliases in _COLUMN_ALIASES.values():
            if _best_alias(value, aliases) >= 92:
                return True
        return False

    def _pairs(self, text: str) -> list[tuple[str, str]]:
        """Every label/value pair on a line.

        Two-column headers put several pairs on one line, so the line is first
        cut on runs of whitespace that precede a new label, then on the colon.
        """
        pairs: list[tuple[str, str]] = []
        if ":" in text:
            segments = re.split(r"\s{2,}(?=[^\s:]+(?:\s[^\s:]+)*\s*:)", text)
            for segment in segments:
                if ":" not in segment:
                    continue
                label, _, value = segment.partition(":")
                pairs.append((label, value))
            return pairs

        # No colons. Two different layouts produce this, and both must work.
        chunks = [c for c in re.split(r"\s{2,}", text.strip()) if c]

        # (a) Column-aligned: the label and its value are in ADJACENT chunks,
        #     e.g. "Patient Name | Baby of Divya | UHID | UH253950". Real OCR
        #     drops the colon and this is what a bill looks like afterwards.
        #     Without this, "Patient Name" pairs with itself and patient.name
        #     comes out as the literal string "Name".
        for left, right in zip(chunks, chunks[1:]):
            pairs.append((left, right))

        # (b) Label and value inside ONE chunk, space separated
        #     (the government layout: "Ward Deluxe Room").
        for chunk in chunks:
            tokens = chunk.split()
            for split_at in (1, 2, 3):
                if split_at < len(tokens):
                    pairs.append((" ".join(tokens[:split_at]), " ".join(tokens[split_at:])))
        return pairs

    def _hospital_name(self, lines: Sequence[OcrLine]) -> str | None:
        """The first substantial line that is not a bilingual banner or a label."""
        for line in lines[:4]:
            text = line.text.strip()
            if len(text) < 4 or ":" in text:
                continue
            latin = [part.strip() for part in text.split("/") if re.search(r"[A-Za-z]", part)]
            if latin:
                candidate = latin[-1].strip()
                if candidate and not candidate.isupper() or len(candidate) > 8:
                    return candidate
        return None

    def _gstin(self, lines: Sequence[OcrLine], labelled: str | None) -> str | None:
        if labelled:
            return labelled.split()[0]
        for line in lines:
            match = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", line.text)
            if match:
                return match.group(0)
        return None

    def _sex(self, values: dict[str, str]) -> str | None:
        direct = values.get("patient.sex")
        if direct:
            return direct.split()[0]
        combined = values.get("patient.age")
        if combined and "/" in combined:
            return combined.rsplit("/", 1)[1].strip()
        return None

    # -- line items ------------------------------------------------------

    def _column_map(self, cells: Sequence[str]) -> dict[int, str] | None:
        """Map cell positions to schema attributes using a header row."""
        mapping: dict[int, str] = {}
        for index, cell in enumerate(cells):
            best_field, best_score = None, 0
            for attribute, aliases in _COLUMN_ALIASES.items():
                score = _best_alias(cell, aliases)
                if score > best_score:
                    best_field, best_score = attribute, score
            if best_field and best_score >= _COLUMN_FUZZ_FLOOR:
                mapping[index] = best_field
        # Require a real table header, not two coincidental words.
        return mapping if len(set(mapping.values())) >= 3 else None

    def _rows(self, lines: Sequence[OcrLine]) -> list[RawLineItem]:
        header_index, column_map = None, None
        for line in lines:
            cells = _split_cells(line.text)
            if len(cells) < 3:
                continue
            mapping = self._column_map(cells)
            if mapping:
                header_index, column_map = line.top, mapping
                break

        items: list[RawLineItem] = []
        for line in lines:
            if header_index is not None and line.top <= header_index:
                continue
            text = line.text.strip()
            if not text or set(text) <= set("-=_ ") or _TOTAL_WORDS.match(text):
                continue

            item = (
                self._row_by_columns(text, column_map)
                if column_map
                else self._row_by_geometry(text)
            )
            if item is not None:
                items.append(item)
        return items

    def _row_by_columns(self, text: str, column_map: dict[int, str]) -> RawLineItem | None:
        cells = _split_cells(text)
        if len(cells) < 2:
            return None
        values: dict[str, str] = {}
        for index, cell in enumerate(cells):
            attribute = column_map.get(index)
            if attribute and cell:
                values[attribute] = cell
        if not values.get("description") or not values.get("amount"):
            return None
        return self._make_item(values)

    def _row_by_geometry(self, text: str) -> RawLineItem | None:
        """No header row: read the numbers off the right-hand side.

        Handles layouts like `3. Oxygen Charges  x2  @ 1,500.00  = 3,000.00`,
        where the trailing numerics are rate and amount regardless of decoration.
        """
        tokens = [t for t in re.split(r"\s+", text.strip()) if t not in {"@", "=", "x", "|"}]
        tokens = [re.sub(r"^[x@=]", "", t) for t in tokens]

        # The serial number has to be consumed BEFORE looking for numerics.
        # Otherwise it is itself the first numeric token, the description slice
        # runs from index 1 to index 0, and every row in a layout without a
        # header (the government one) is silently dropped.
        serial = None
        start = 0
        if tokens and re.fullmatch(r"\d{1,3}[.)]?", tokens[0]):
            serial = tokens[0].rstrip(".)")
            start = 1

        numeric = [i for i in range(start, len(tokens)) if _looks_numeric(tokens[i])]
        if len(numeric) < 2:
            return None

        amount = tokens[numeric[-1]]
        unit_rate = tokens[numeric[-2]]
        quantity = tokens[numeric[-3]] if len(numeric) >= 3 else None

        description_tokens = [
            t for t in tokens[start:numeric[0]] if not _DATE.match(t)
        ]
        description = " ".join(description_tokens).strip()
        if not description:
            return None

        service_date = next((t for t in tokens if _DATE.match(t)), None)
        return self._make_item({
            "serial_no": serial or "",
            "description": description,
            "service_date": service_date or "",
            "quantity": quantity or "",
            "unit_rate": unit_rate or "",
            "amount": amount,
        })

    def _make_item(self, values: dict[str, str]) -> RawLineItem:
        description = values.get("description", "")
        non_payable = re.search(
            r"attendant|telephone|toiletr|food|registration|record",
            description, re.IGNORECASE,
        )
        return RawLineItem(
            serial_no=values.get("serial_no") or None,
            description=description or None,
            service_date=values.get("service_date") or None,
            category=None,   # B1 does not classify; the schema allows abstention
            quantity=values.get("quantity") or None,
            unit_rate=values.get("unit_rate") or None,
            amount=values.get("amount") or None,
            hsn_code=None,
            is_payable="N" if non_payable else "Y",
            deduction_reason=(
                "IRDAI List I - non-medical expense" if non_payable else None
            ),
        )


def extract(source: object, backend: OcrBackend | None = None) -> RawDocument:
    return B1OcrHeuristic(backend).extract(source)

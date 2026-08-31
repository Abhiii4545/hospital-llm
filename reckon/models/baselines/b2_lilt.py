"""B2 - LiLT token classification over OCR tokens.

`SCUT-DLVCLab/lilt-roberta-en-base` (MIT). The strong non-generative baseline,
and the one that actually has to be beaten: B0 is a regex engine and B1 is a
week of heuristics, but B2 is a layout-aware transformer doing the same job by a
different route. **If Donut cannot beat B2, that is a finding worth reporting,
not a failure to hide.**

LayoutLMv3 would be the obvious comparison and is forbidden here - it is
CC-BY-NC-SA. LiLT is the MIT-licensed equivalent.

Structure of the task, and where it differs from Donut:

* LiLT reads OCR tokens with bounding boxes and assigns each a BIO tag. It cannot
  invent text, so it never hallucinates a value - but it also cannot recover a
  value OCR failed to read, and it cannot normalise. Donut can do both.
* Line items are the hard part for a token classifier. A row is a *group* of
  spans, and BIO tagging has no native notion of "these five spans are one row".
  Row grouping is done geometrically afterwards, by vertical band, which is the
  honest weakness of this approach and is documented rather than hidden.

The BIO scheme and the span decoder are pure Python and are tested. Training
requires torch, transformers and an OCR pass over the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from reckon.models.baselines.ocr import OcrBackend, PlainTextBackend
from reckon.schema import (
    FIELD_PATHS,
    LINE_ITEM_FIELDS,
    RawDocument,
    RawHospital,
    RawInsurance,
    RawLineItem,
    RawPatient,
    RawTotals,
)

__all__ = [
    "LABELS",
    "label_id",
    "OcrToken",
    "assign_bio_labels",
    "decode_spans",
    "group_rows_by_band",
    "B2LiltBaseline",
    "MODEL_NAME",
]

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"     # MIT

#: Tagged fields: every scalar field path, plus line-item attributes prefixed so
#: they cannot collide with a header field of the same name.
_TAGGABLE: tuple[str, ...] = (
    *FIELD_PATHS,
    *(f"item.{name}" for name in LINE_ITEM_FIELDS),
)

#: BIO scheme. "O" first so id 0 is the majority class, which keeps a confusion
#: matrix readable and matches the convention every token-classification recipe
#: assumes.
LABELS: tuple[str, ...] = ("O", *[
    f"{prefix}-{path}" for path in _TAGGABLE for prefix in ("B", "I")
])

_LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def label_id(label: str) -> int:
    return _LABEL_TO_ID[label]


@dataclass
class OcrToken:
    """One OCR word with a normalised 0-1000 box, as LiLT expects."""

    text: str
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 in 0..1000
    line: int = 0

    @property
    def y_centre(self) -> float:
        return (self.box[1] + self.box[3]) / 2


def _tokenise(value: str) -> list[str]:
    return [t for t in re.split(r"\s+", value.strip()) if t]


def assign_bio_labels(
    tokens: Sequence[OcrToken],
    field_values: dict[str, str | None],
) -> list[str]:
    """Tag OCR tokens by matching ground-truth field values against them.

    This is *distant supervision*: labels come from string matching, not from a
    human pointing at the page. Two consequences worth stating plainly, because
    they cap what B2 can score:

    * a value OCR misread cannot be matched, so it becomes an unlabelled token
      and the model is trained to call it "O" - a systematic recall ceiling;
    * a short value that appears twice (a "1" that is both a serial number and a
      quantity) matches the first occurrence, which may be the wrong one.

    Longest values are matched first so a specific value claims its tokens before
    a short one can steal them.
    """
    labels = ["O"] * len(tokens)
    ordered = sorted(
        ((path, value) for path, value in field_values.items() if value),
        key=lambda pair: len(str(pair[1])),
        reverse=True,
    )

    for path, value in ordered:
        wanted = _tokenise(str(value))
        if not wanted:
            continue
        span = _find_span(tokens, wanted, labels)
        if span is None:
            continue
        start, end = span
        labels[start] = f"B-{path}"
        for index in range(start + 1, end):
            labels[index] = f"I-{path}"
    return labels


def _find_span(
    tokens: Sequence[OcrToken], wanted: Sequence[str], labels: Sequence[str]
) -> tuple[int, int] | None:
    """First unclaimed run of tokens matching *wanted*, case-insensitively."""
    target = [w.casefold().strip(".,:;") for w in wanted]
    size = len(target)
    for start in range(len(tokens) - size + 1):
        if any(labels[start + offset] != "O" for offset in range(size)):
            continue
        window = [tokens[start + offset].text.casefold().strip(".,:;")
                  for offset in range(size)]
        if window == target:
            return start, start + size
    return None


def decode_spans(
    tokens: Sequence[OcrToken], labels: Sequence[str]
) -> dict[str, list[tuple[str, list[OcrToken]]]]:
    """Turn a BIO tag sequence back into (field path, text) spans.

    An ``I-`` tag with no preceding ``B-`` starts a new span rather than being
    dropped. Real model output does produce that, and discarding it would throw
    away a correct value over a tagging technicality.
    """
    spans: dict[str, list[tuple[str, list[OcrToken]]]] = {}
    current_path: str | None = None
    current: list[OcrToken] = []

    def flush() -> None:
        nonlocal current_path, current
        if current_path and current:
            text = " ".join(t.text for t in current)
            spans.setdefault(current_path, []).append((text, list(current)))
        current_path, current = None, []

    for token, label in zip(tokens, labels):
        if label == "O":
            flush()
            continue
        prefix, _, path = label.partition("-")
        if prefix == "B" or path != current_path:
            flush()
            current_path = path
        current.append(token)
    flush()
    return spans


def group_rows_by_band(
    item_spans: dict[str, list[tuple[str, list[OcrToken]]]],
    tolerance: float = 12.0,
) -> list[dict[str, str]]:
    """Group line-item spans into rows by vertical position.

    **This is the honest weakness of a token-classification approach to line
    items.** BIO tagging has no notion of "these spans belong to the same row";
    the grouping is geometric, so a row that wraps onto two visual lines, or a
    table with tight row spacing, will be grouped wrongly no matter how good the
    tagging is. Donut has no equivalent failure because it emits rows directly.
    """
    entries: list[tuple[float, str, str]] = []
    for path, spans in item_spans.items():
        if not path.startswith("item."):
            continue
        attribute = path.split(".", 1)[1]
        for text, tokens in spans:
            if tokens:
                centre = sum(t.y_centre for t in tokens) / len(tokens)
                entries.append((centre, attribute, text))

    entries.sort(key=lambda e: e[0])
    rows: list[dict[str, str]] = []
    band_centre: float | None = None
    current: dict[str, str] = {}

    for centre, attribute, text in entries:
        if band_centre is None or abs(centre - band_centre) <= tolerance:
            band_centre = centre if band_centre is None else band_centre
            # A repeated attribute in one band means the band spans two rows.
            if attribute in current:
                rows.append(current)
                current = {}
                band_centre = centre
            current[attribute] = text
        else:
            if current:
                rows.append(current)
            current = {attribute: text}
            band_centre = centre
    if current:
        rows.append(current)
    return rows


def spans_to_document(
    spans: dict[str, list[tuple[str, list[OcrToken]]]]
) -> RawDocument:
    """Assemble decoded spans into a document."""
    blocks: dict[str, dict[str, str]] = {
        "hospital": {}, "patient": {}, "insurance": {}, "totals": {}
    }
    for path, found in spans.items():
        if path.startswith("item.") or "." not in path:
            continue
        block, name = path.split(".", 1)
        if block in blocks and found:
            blocks[block][name] = found[0][0]

    return RawDocument(
        hospital=RawHospital(**blocks["hospital"]),
        patient=RawPatient(**blocks["patient"]),
        insurance=RawInsurance(**blocks["insurance"]),
        totals=RawTotals(**blocks["totals"]),
        line_items=[RawLineItem(**row) for row in group_rows_by_band(spans)],
    )


@dataclass
class B2LiltBaseline:
    """Inference wrapper. Training lives in reckon/training/train_lilt.py."""

    name: str = "B2 (LiLT)"
    checkpoint: str | None = None
    backend: OcrBackend = field(default_factory=PlainTextBackend)
    _model: Any = None
    _tokenizer: Any = None

    @property
    def available(self) -> bool:
        return self.checkpoint is not None

    def extract(self, source: object) -> RawDocument:
        if not self.available:
            raise NotImplementedError(
                "B2 needs a fine-tuned LiLT checkpoint and an OCR pass over the "
                "corpus. Neither exists yet; see docs/MODEL_CARD.md. B2 is the "
                "baseline that actually has to be beaten, so its absence is a "
                "gap in the comparison, not a detail."
            )
        raise NotImplementedError("LiLT inference path not yet wired")

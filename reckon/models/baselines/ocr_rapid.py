"""Real OCR backend: PaddleOCR's models under ONNXRuntime (RapidOCR, Apache-2.0).

The brief specifies PaddleOCR. Installing `paddleocr` on this machine fails -
one of its transitive dependencies needs MSVC build tools that are not present -
so this uses RapidOCR, which runs **the same PaddleOCR detection and recognition
models** exported to ONNX. Same models, same licence (Apache-2.0), no compiler.
The substitution is recorded rather than passed off as the original.

The interesting work here is not calling the OCR engine; it is turning boxes back
into LINES. B1's heuristics read column structure out of horizontal whitespace,
so a bag of boxes is useless to it. This reconstructs a text layout by grouping
boxes into rows and placing each at a character column derived from its x
coordinate - which preserves the column gaps B1 keys off, and is exactly the
information a naive `" ".join(texts)` would destroy.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from reckon.models.baselines.ocr import OcrLine, OcrPage

__all__ = ["RapidOcrBackend", "boxes_to_page"]


def _bounds(box) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def boxes_to_page(
    results: list[tuple[Any, str, float]],
    row_tolerance: float | None = None,
    min_confidence: float = 0.30,
) -> OcrPage:
    """Group OCR boxes into text lines with column positions preserved.

    Row grouping is by vertical overlap rather than by a fixed y tolerance,
    because a skewed page (every page in the `heavy` bucket is skewed) puts the
    left and right ends of one printed line at noticeably different y.
    """
    entries = []
    for box, text, score in results:
        if not text or score < min_confidence:
            continue
        x0, y0, x1, y1 = _bounds(box)
        entries.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "text": text.strip(), "h": y1 - y0})
    if not entries:
        return OcrPage(lines=[])

    median_height = statistics.median(e["h"] for e in entries) or 12.0
    tolerance = row_tolerance if row_tolerance is not None else median_height * 0.6

    entries.sort(key=lambda e: (e["y0"], e["x0"]))
    rows: list[list[dict]] = []
    for entry in entries:
        placed = False
        for row in rows:
            centre = statistics.fmean((r["y0"] + r["y1"]) / 2 for r in row)
            if abs((entry["y0"] + entry["y1"]) / 2 - centre) <= tolerance:
                row.append(entry)
                placed = True
                break
        if not placed:
            rows.append([entry])

    # Character width from the recognised text itself, so the reconstruction
    # tracks the page's actual font size rather than an assumed one.
    widths = [(e["x1"] - e["x0"]) / max(1, len(e["text"])) for e in entries
              if e["text"]]
    char_width = statistics.median(widths) or 6.0

    lines: list[OcrLine] = []
    rows.sort(key=lambda r: min(e["y0"] for e in r))
    for index, row in enumerate(rows):
        row.sort(key=lambda e: e["x0"])
        rendered = ""
        for entry in row:
            column = int(entry["x0"] / char_width)
            if column > len(rendered):
                rendered += " " * (column - len(rendered))
            elif rendered:
                # Boxes that would overlap still need a separator, and it must be
                # two spaces: a single space would merge two columns into one
                # cell and silently destroy the table structure.
                rendered += "  "
            rendered += entry["text"]
        left = int(min(e["x0"] for e in row) / char_width)
        lines.append(OcrLine(text=rendered.rstrip(), top=index, left=left))
    return OcrPage(lines=lines)


@dataclass
class RapidOcrBackend:
    """PaddleOCR models via ONNXRuntime. Engine is created once and reused."""

    min_confidence: float = 0.30
    _engine: Any = field(default=None, repr=False)

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, source: object) -> OcrPage:
        import numpy as np
        from PIL import Image

        engine = self._get_engine()
        if isinstance(source, (str, bytes)):
            image = source
        elif isinstance(source, Image.Image):
            image = np.array(source.convert("RGB"))
        else:
            image = source

        results, _ = engine(image)
        return boxes_to_page(results or [], min_confidence=self.min_confidence)

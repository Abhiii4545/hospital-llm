"""OCR seam for the baselines.

B1 is defined as "PaddleOCR plus positional heuristics". PaddleOCR (Apache-2.0)
plus paddlepaddle is close to a gigabyte and there are no page images before
Phase 3, so the heuristics are written against this interface instead and the
engine is swapped in behind it when images exist.

The Phase 2 backend hands over text that is already perfect. That must be stated
wherever B1's mini-set numbers are quoted: the OCR stage is not being exercised,
so the score is an upper bound B1 will not reach on a scanned page.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

__all__ = ["OcrLine", "OcrPage", "OcrBackend", "PlainTextBackend", "PaddleOcrBackend"]


class OcrLine(NamedTuple):
    """One line of recognised text with enough geometry for column heuristics."""

    text: str
    top: int    # line index from the top of the page
    left: int   # column offset of the first non-space character


class OcrPage(NamedTuple):
    lines: list[OcrLine]

    @property
    def raw_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@runtime_checkable
class OcrBackend(Protocol):
    def read(self, source: object) -> OcrPage: ...


class PlainTextBackend:
    """Phase 2 backend: the text is given, no recognition happens."""

    def read(self, source: object) -> OcrPage:
        text = source if isinstance(source, str) else str(source)
        lines = []
        for index, raw in enumerate(text.splitlines()):
            stripped = raw.lstrip()
            lines.append(
                OcrLine(text=raw.rstrip(), top=index, left=len(raw) - len(stripped))
            )
        return OcrPage(lines=lines)


class PaddleOcrBackend:
    """Real OCR. Lands in Phase 3 once page images exist.

    Kept as an explicit stub rather than an import so that the absence of a
    ~1GB dependency is a clear message instead of an ImportError halfway
    through an evaluation run.
    """

    def read(self, source: object) -> OcrPage:  # pragma: no cover - phase 3
        raise NotImplementedError(
            "PaddleOcrBackend lands in Phase 3, when page images exist. "
            "Install with: uv add paddleocr paddlepaddle  (both Apache-2.0)"
        )

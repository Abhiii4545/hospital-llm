"""B3 - off-the-shelf Donut, zero-shot. The "just use a HuggingFace model" arm.

`naver-clova-ix/donut-base-finetuned-cord-v2` (MIT) is Donut fine-tuned on CORD,
a receipt dataset. A receipt line-item table is the closest public analogue to a
hospital bill table, so this is the strongest zero-shot option that exists -
there is no pretrained model for Indian hospital bills.

**This baseline exists to answer a specific question honestly**: can you skip the
fine-tuning and just run a pretrained document model? The brief does not ask for
it. It is here because "why not just use something off HuggingFace" is the first
thing anyone sensible asks, and the answer should be a measurement rather than an
assertion.

CORD's schema is mapped onto ours where the fields genuinely correspond:

    menu.nm        -> line_item.description
    menu.cnt       -> line_item.quantity
    menu.unitprice -> line_item.unit_rate
    menu.price     -> line_item.amount
    total.total_price     -> totals.net_amount
    sub_total.subtotal_price -> totals.gross_amount

Everything else in our schema - patient, insurance, GSTIN, ward type, dates,
payability - has **no CORD equivalent at all**, so this arm is structurally
incapable of producing them. That is not a flaw in the mapping; it is the point.
A receipt model has no concept of a policy number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from reckon.schema import RawDocument, RawLineItem, RawTotals

__all__ = ["MODEL_NAME", "CORD_TO_SCHEMA", "parse_cord_sequence", "B3DonutCord"]

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"     # MIT
TASK_PROMPT = "<s_cord-v2>"

#: CORD tag -> our field. Only genuine correspondences; nothing is forced.
CORD_TO_SCHEMA: dict[str, str] = {
    "nm": "description",
    "cnt": "quantity",
    "unitprice": "unit_rate",
    "price": "amount",
}

_TOTAL_TAGS = {
    "total_price": "net_amount",
    "subtotal_price": "gross_amount",
}

#: A degenerate decoder loops on one phrase. Detecting it matters: without this
#: the repeated text is scored as if it were an extraction attempt, which
#: flatters nothing but does hide *why* the arm failed.
_REPEAT = re.compile(r"\b(\w[\w ]{3,30}?)\b(?:\s+\1\b){3,}", re.IGNORECASE)


def _tag_pairs(text: str) -> list[tuple[str, str]]:
    """Every ``<s_tag>value</s_tag>`` pair, in order, tolerant of bad nesting.

    Zero-shot output is frequently malformed - an opening tag closed by a
    different tag's closer. Values are taken up to the next tag of any kind
    rather than requiring a matching close, or almost everything would be lost.
    """
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"<s_([a-z_0-9-]+)>(.*?)(?=<)", text, re.DOTALL):
        tag, value = match.group(1), match.group(2).strip()
        if value:
            pairs.append((tag, value))
    return pairs


def has_repetition_loop(text: str) -> bool:
    return bool(_REPEAT.search(text))


def parse_cord_sequence(text: str) -> RawDocument:
    """Map a raw CORD output sequence onto our schema."""
    document = RawDocument()
    items: list[dict[str, str]] = []
    current: dict[str, str] = {}
    totals: dict[str, str] = {}

    for tag, value in _tag_pairs(text):
        if tag in _TOTAL_TAGS:
            totals.setdefault(_TOTAL_TAGS[tag], value)
            continue
        field_name = CORD_TO_SCHEMA.get(tag)
        if field_name is None:
            continue
        # A repeated key means the previous row ended.
        if field_name in current:
            items.append(current)
            current = {}
        current[field_name] = value

    if current:
        items.append(current)

    document.line_items = [
        RawLineItem(**row) for row in items if row.get("description")
    ]
    if totals:
        document.totals = RawTotals(**totals)
    return document


@dataclass
class B3DonutCord:
    """Zero-shot inference. torch and transformers are imported lazily."""

    name: str = "B3 (Donut-CORD zero-shot)"
    model_name: str = MODEL_NAME
    max_length: int = 512
    _processor: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    def load(self) -> None:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        if self._model is None:
            self._processor = DonutProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name
            ).eval()

    def run_raw(self, image) -> str:
        import torch

        self.load()
        pixel_values = self._processor(image, return_tensors="pt").pixel_values
        prompt = self._processor.tokenizer(
            TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        with torch.no_grad():
            output = self._model.generate(
                pixel_values,
                decoder_input_ids=prompt,
                max_length=self.max_length,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                num_beams=1,
                use_cache=True,
                return_dict_in_generate=True,
            )
        return self._processor.batch_decode(output.sequences)[0]

    def extract(self, image) -> RawDocument:
        return parse_cord_sequence(self.run_raw(image))

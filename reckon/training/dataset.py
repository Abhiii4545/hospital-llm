"""Page dataset for the two Donut heads.

Reads the corpus manifest and yields (image, target string) pairs. Head B is the
default because it is trained first: it is the harder task and the one that
decides whether the project works.

Torch and transformers are imported lazily inside the methods that need them, so
this module can be imported - and its target construction tested - in an
environment with neither installed. That matters because the target logic is the
part most likely to be wrong, and it should not require a 2GB dependency to
check.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from reckon.schema import RawDocument, RawLineItem
from reckon.serialize import render_head_a, render_head_b

__all__ = ["ManifestRow", "PageSample", "load_manifest", "build_target",
           "PageDataset", "Head"]

Head = Literal["a", "b"]


@dataclass(frozen=True)
class ManifestRow:
    page_id: str
    image: Path
    targets: Path
    split: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class PageSample:
    page_id: str
    image_path: Path
    target: str
    meta: dict[str, Any]


def load_manifest(
    manifest: Path | str, split: str | None = None
) -> list[ManifestRow]:
    path = Path(manifest)
    root = path.parent
    rows: list[ManifestRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if split and payload.get("split") != split:
            continue
        rows.append(ManifestRow(
            page_id=payload["page_id"],
            image=root / payload["image"],
            targets=root / payload["targets"],
            split=payload.get("split", "train"),
            meta=payload,
        ))
    return rows


def build_target(targets_payload: dict, head: Head) -> str:
    """Target string for one page and one head.

    Head A on a continuation page yields the empty string, which is correct and
    is deliberately KEPT in the dataset rather than filtered out. The model has
    to learn that a page can carry no header at all; training only on pages that
    do have one would guarantee it hallucinates a header for every continuation
    page it ever sees.
    """
    if head == "b":
        items = [RawLineItem(**row) for row in
                 targets_payload["head_b"]["line_items"]]
        return render_head_b(items)

    head_a = targets_payload.get("head_a", {})
    document = RawDocument()
    for block in ("hospital", "patient", "insurance", "totals"):
        if block in head_a:
            model = type(getattr(document, block))
            setattr(document, block, model(**head_a[block]))
    return render_head_a(
        document,
        include_header=any(b in head_a for b in ("hospital", "patient", "insurance")),
        include_totals="totals" in head_a,
    )


def iter_samples(
    manifest: Path | str, head: Head, split: str | None = None
) -> Iterator[PageSample]:
    for row in load_manifest(manifest, split):
        payload = json.loads(row.targets.read_text(encoding="utf-8"))
        yield PageSample(
            page_id=row.page_id,
            image_path=row.image,
            target=build_target(payload, head),
            meta=row.meta,
        )


class PageDataset:
    """torch Dataset over corpus pages. torch is imported lazily."""

    def __init__(
        self,
        manifest: Path | str,
        head: Head = "b",
        split: str = "train",
        processor: Any = None,
        image_size: tuple[int, int] = (1280, 960),   # (height, width)
        max_length: int = 1024,
        augment_order: bool = False,
        seed: int = 1337,
    ) -> None:
        self.samples: list[PageSample] = list(iter_samples(manifest, head, split))
        self.head = head
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        self.augment_order = augment_order
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def target_lengths(self, tokenizer: Any) -> list[int]:
        """Token length of every target. Run this BEFORE choosing max_length.

        Silently truncating targets is the failure mode that destroys line-item
        recall without producing any error, so the length distribution is
        measured rather than assumed.
        """
        return [len(tokenizer(s.target).input_ids) for s in self.samples]

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        pixel_values = self.processor(
            image, random_padding=self.head == "b", return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample.target,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # -100 is ignored by the loss; padding must not be learned as content.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "target": sample.target,
            "page_id": sample.page_id,
        }


def split_counts(manifest: Path | str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in load_manifest(manifest):
        counts[row.split] = counts.get(row.split, 0) + 1
    return counts

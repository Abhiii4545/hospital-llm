"""Contact sheet for human review of the corpus.

The Phase 3 gate is a person looking at 100 random pages. That is not ceremony:
every corpus bug found so far in this project was found by looking - a Telugu
header on a Tamil Nadu hospital, "Ms. Rohit Kumar / Sex: F", and a `heavy`
augmentation bucket that rendered pages no human could read. None of those would
have shown up in a class-balance table.

The sheet is sampled STRATIFIED by layout and quality rather than uniformly, so
the rare-but-important combinations (a heavy-quality bilingual government page)
actually appear in the 100 a person will look at.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

__all__ = ["build_contact_sheet", "sample_stratified"]

THUMB_W = 260
COLUMNS = 10
LABEL_H = 26
PADDING = 6


def sample_stratified(
    rows: Sequence[dict], n: int, seed: int = 1337
) -> list[dict]:
    """Pick *n* pages spread across (layout, quality) cells.

    Uniform sampling of 100 pages from 10,000 would, at these proportions, miss
    several layout/quality cells entirely - exactly the cells most likely to be
    broken.
    """
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(str(row.get("layout", "?")), str(row.get("quality", "?")))].append(row)

    order = sorted(cells)
    rng.shuffle(order)
    picked: list[dict] = []
    round_index = 0
    while len(picked) < n and any(cells[key] for key in order):
        for key in order:
            bucket = cells[key]
            if not bucket:
                continue
            picked.append(bucket.pop(rng.randrange(len(bucket))))
            if len(picked) >= n:
                break
        round_index += 1
        if round_index > n:
            break
    return picked


def build_contact_sheet(
    manifest: Path | str,
    out_path: Path | str = "reports/contact_sheet.png",
    n: int = 100,
    seed: int = 1337,
) -> Path:
    manifest_path = Path(manifest)
    root = manifest_path.parent
    rows = [json.loads(line) for line in
            manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = sample_stratified(rows, n, seed)

    thumbs: list[tuple[Image.Image, str]] = []
    for row in picked:
        image = Image.open(root / row["image"]).convert("RGB")
        ratio = THUMB_W / image.width
        thumb = image.resize((THUMB_W, max(1, int(image.height * ratio))),
                             Image.LANCZOS)
        label = (f"{row.get('layout','?')} | {row.get('quality','?')} | "
                 f"p{row.get('page','?')}/{row.get('total_pages','?')} | "
                 f"{row.get('messiness','none')}")
        thumbs.append((thumb, label))

    if not thumbs:
        raise ValueError("no pages sampled")

    cell_h = max(t.height for t, _ in thumbs) + LABEL_H
    cell_w = THUMB_W
    rows_count = (len(thumbs) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (COLUMNS * (cell_w + PADDING) + PADDING,
         rows_count * (cell_h + PADDING) + PADDING + 30),
        "#20242b",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((PADDING, 8),
              f"RECKON synthetic corpus - {len(thumbs)} pages, stratified by "
              f"layout x quality (of {len(rows)} total)",
              fill="#e8e8e8")

    for index, (thumb, label) in enumerate(thumbs):
        col, row_i = index % COLUMNS, index // COLUMNS
        x = PADDING + col * (cell_w + PADDING)
        y = 30 + PADDING + row_i * (cell_h + PADDING)
        sheet.paste(thumb, (x, y))
        # Two lines so the layout id stays readable at thumbnail width.
        head, _, tail = label.partition(" | ")
        draw.text((x + 2, y + thumb.height + 2), head, fill="#9fd0ff")
        draw.text((x + 2, y + thumb.height + 13), tail, fill="#c9c9c9")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/synthetic/manifest.jsonl")
    parser.add_argument("--out", default="reports/contact_sheet.png")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    path = build_contact_sheet(args.manifest, args.out, args.n, args.seed)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the stratified synthetic corpus.

Produces page images plus per-page targets for both heads, and a manifest with a
SHA-256 per file. Section 4.3: every training run records the manifest hash, so
if the data changes the hash changes and old results are marked stale
automatically.

Stratification is by LAYOUT, assigned round-robin before any sampling, so every
one of the 21 layouts appears in train, val and synthetic-test in proportion.
Random assignment would leave some layout absent from the test split at these
sizes, and "the model fails on layout 14" would then be unmeasurable.

Usage:
    python -m reckon.data.build_corpus --documents 2000 --out data/synthetic
    python -m reckon.data.build_corpus --documents 2000 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image

from reckon.data.augment import (
    QUALITY_BUCKETS,
    augment_page,
    edge_energy,
    ink_contrast,
    is_legible,
    sample_quality,
)
from reckon.data.generators.document import GeneratedDocument, generate_document
from reckon.data.layouts import LAYOUTS, by_id
from reckon.data.render import Renderer
from reckon.provenance import run_metadata

__all__ = ["build", "SPLITS", "Manifest", "corpus_hash"]

#: train / val / synthetic-test. The real splits are separate and locked.
SPLITS: dict[str, float] = {"train": 0.80, "val": 0.10, "synth_test": 0.10}

DEFAULT_OUT = Path("data/synthetic")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_for(index: int) -> str:
    """Deterministic split assignment, stratified within each layout.

    Uses the document's position WITHIN its layout, so each layout is divided in
    the same proportions rather than by a global random draw.
    """
    position = (index // len(LAYOUTS)) % 10
    if position < 8:
        return "train"
    return "val" if position == 8 else "synth_test"


@dataclass
class Manifest:
    rows: list[dict]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def read(path: Path) -> "Manifest":
        with open(path, encoding="utf-8") as handle:
            return Manifest([json.loads(line) for line in handle if line.strip()])


def corpus_hash(rows: Iterable[dict]) -> str:
    """Order-independent hash over every page's content hash."""
    digest = hashlib.sha256()
    for sha in sorted(row["sha256"] for row in rows):
        digest.update(sha.encode())
    return digest.hexdigest()


def _page_targets(doc: GeneratedDocument, page_index: int) -> dict:
    """Per-page targets for the two heads.

    Head A sees header/totals only on the pages that actually carry them; Head B
    sees the line items on THIS page. A duplicated row reprinted across a page
    break is deliberately NOT in the target - the assembly layer has to notice
    it, and rewarding the model for emitting it twice would defeat that.
    """
    page = doc.pages[page_index]
    head_a: dict = {}
    if page.show_header:
        head_a["hospital"] = doc.truth.hospital.model_dump()
        head_a["patient"] = doc.truth.patient.model_dump()
        head_a["insurance"] = doc.truth.insurance.model_dump()
    if page.show_totals:
        head_a["totals"] = doc.truth.totals.model_dump()
    return {
        "head_a": head_a,
        "head_b": {"line_items": [i.model_dump() for i in page.items]},
        "has_header": page.show_header,
        "has_totals": page.show_totals,
        "duplicate_row_reprinted": page.duplicate_of_previous_last,
    }


def _iter_documents(
    n_documents: int, seed: int, shard: int, workers: int
) -> Iterator[tuple[int, GeneratedDocument, object]]:
    for index in range(n_documents):
        if index % workers != shard:
            continue
        spec = LAYOUTS[index % len(LAYOUTS)]
        # Per-document RNG so a document is reproducible regardless of which
        # worker or shard produced it.
        rng = random.Random(f"{seed}:{index}")
        doc = generate_document(
            rng, f"syn_{index:06d}", spec.id, spec.rows_per_page,
            hospital_type=spec.hospital_type, bilingual=spec.bilingual,
        )
        yield index, doc, spec


def build(
    n_documents: int = 1200,
    out_dir: Path | str = DEFAULT_OUT,
    seed: int = 1337,
    shard: int = 0,
    workers: int = 1,
    max_pages: int | None = None,
    progress_every: int = 50,
) -> Path:
    """Generate, render, augment and manifest one shard of the corpus."""
    out = Path(out_dir)
    images = out / "pages"
    images.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: Counter[str] = Counter()
    illegible = 0
    pages_done = 0
    started = time.time()

    with Renderer() as renderer:
        for index, doc, spec in _iter_documents(n_documents, seed, shard, workers):
            split = _split_for(index)
            aug_rng = random.Random(f"{seed}:aug:{index}")
            quality = sample_quality(aug_rng)

            for page in doc.pages:
                name = f"{doc.doc_id}_p{page.index:02d}"
                png = images / f"{name}.png"
                renderer.render_page(doc, spec, page, png)

                image = np.array(Image.open(png).convert("RGB"))
                augmented, step_failures = augment_page(image, quality, aug_rng)
                failures.update(step_failures)

                legible = is_legible(augmented)
                if not legible:
                    # Keep the clean render rather than a page nothing can read.
                    # Silently writing an unreadable page would be label noise.
                    illegible += 1
                    augmented = image

                Image.fromarray(augmented.astype("uint8")).save(png, optimize=True)

                targets = _page_targets(doc, page.index)
                (images / f"{name}.json").write_text(
                    json.dumps({"doc_id": doc.doc_id, "page": page.index,
                                **targets}, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )

                rows.append({
                    "page_id": name,
                    "doc_id": doc.doc_id,
                    "page": page.index,
                    "total_pages": page.total_pages,
                    "image": str(png.relative_to(out)).replace("\\", "/"),
                    "targets": f"pages/{name}.json",
                    "sha256": _sha256(png),
                    "split": split,
                    "quality": quality,
                    "legible": legible,
                    "ink_contrast": round(ink_contrast(augmented), 1),
                    "edge_energy": round(edge_energy(augmented), 1),
                    "has_header": targets["has_header"],
                    "has_totals": targets["has_totals"],
                    "n_line_items": len(page.items),
                    **{k: v for k, v in doc.meta.items()},
                })

                pages_done += 1
                if progress_every and pages_done % progress_every == 0:
                    rate = pages_done / max(1e-6, time.time() - started)
                    print(f"[shard {shard}] {pages_done} pages  {rate:.1f}/s",
                          file=sys.stderr, flush=True)
                if max_pages and pages_done >= max_pages:
                    break
            if max_pages and pages_done >= max_pages:
                break

    suffix = "" if workers == 1 else f".{shard}"
    manifest_path = out / f"manifest{suffix}.jsonl"
    Manifest(rows).write(manifest_path)

    stats = {
        "pages": len(rows),
        "documents": len({r["doc_id"] for r in rows}),
        "illegible_reverted": illegible,
        "augmentation_failures": dict(failures),
        "corpus_sha256": corpus_hash(rows),
        "seed": seed,
        "provenance": run_metadata(seed=seed),
    }
    (out / f"stats{suffix}.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    print(f"[shard {shard}] wrote {len(rows)} pages -> {manifest_path}",
          file=sys.stderr, flush=True)
    return manifest_path


def merge_shards(out_dir: Path | str, workers: int) -> Path:
    out = Path(out_dir)
    rows: list[dict] = []
    for shard in range(workers):
        path = out / f"manifest.{shard}.jsonl"
        if path.exists():
            rows.extend(Manifest.read(path).rows)
    rows.sort(key=lambda r: r["page_id"])
    merged = Manifest(rows).write(out / "manifest.jsonl")
    (out / "corpus_hash.txt").write_text(corpus_hash(rows), encoding="utf-8")
    return merged


def class_balance(rows: list[dict]) -> dict[str, Counter]:
    """The table the Phase 3 gate asks for."""
    keys = ("split", "layout", "quality", "page_bucket", "language",
            "hospital_type", "ward", "messiness", "region")
    table: dict[str, Counter] = {}
    for key in keys:
        if any(key in row for row in rows):
            table[key] = Counter(str(row.get(key, "?")) for row in rows)
    table["has_header"] = Counter(str(r["has_header"]) for r in rows)
    table["has_totals"] = Counter(str(r["has_totals"]) for r in rows)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=1200)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=int, default=None,
                        help="internal: render only this shard")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args(argv)

    if args.shard is not None:
        build(args.documents, args.out, args.seed, args.shard, args.workers,
              args.max_pages)
        return 0

    if args.workers <= 1:
        build(args.documents, args.out, args.seed, 0, 1, args.max_pages)
    else:
        import subprocess

        procs = [
            subprocess.Popen([
                sys.executable, "-m", "reckon.data.build_corpus",
                "--documents", str(args.documents), "--out", args.out,
                "--seed", str(args.seed), "--workers", str(args.workers),
                "--shard", str(shard),
                *(["--max-pages", str(args.max_pages)] if args.max_pages else []),
            ], env={**os.environ})
            for shard in range(args.workers)
        ]
        for proc in procs:
            if proc.wait() != 0:
                print(f"shard failed with {proc.returncode}", file=sys.stderr)
                return 1
        merge_shards(args.out, args.workers)

    rows = Manifest.read(Path(args.out) / "manifest.jsonl").rows
    print(f"\ncorpus: {len(rows)} pages, sha256 {corpus_hash(rows)[:16]}")
    for key, counts in class_balance(rows).items():
        top = ", ".join(f"{k}={v}" for k, v in counts.most_common(6))
        print(f"  {key:14s} {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

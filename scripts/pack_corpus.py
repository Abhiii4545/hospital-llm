"""Pack the corpus for upload: PNG -> JPEG, then a single archive.

A 10,000-page corpus of augmented PNGs is ~7.7 GB. PNG is lossless, and lossless
compression of *scanner noise* is close to useless - the noise is exactly the
high-entropy content PNG cannot squeeze. That size is a real obstacle: it is half
of a free Google Drive, a slow upload on home broadband, and session time spent
downloading rather than training.

JPEG at quality 90 brings it to roughly 1.2 GB with no meaningful loss, and the
loss there is is of a kind already present: the augmentation pipeline applies
JPEG compression itself, so these pages have been through a JPEG round trip
regardless. The model reads them at 960x1280 either way.

The manifest is rewritten with new paths and new SHA-256 digests, so the packed
corpus is a properly versioned artefact rather than the same corpus with
different bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

__all__ = ["convert", "main"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2


def convert(
    corpus: Path,
    quality: int = 90,
    keep_png: bool = False,
) -> dict:
    """Rewrite every page as JPEG and update the manifest in place."""
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}; is the build finished?")

    before = _dir_size_mb(corpus)
    rows = [json.loads(line) for line in
            manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    converted = 0
    for row in rows:
        source = corpus / row["image"]
        if source.suffix.lower() != ".png" or not source.exists():
            continue
        target = source.with_suffix(".jpg")
        with Image.open(source) as image:
            image.convert("RGB").save(
                target, "JPEG", quality=quality, optimize=True, progressive=True
            )
        if not keep_png:
            source.unlink()
        row["image"] = str(target.relative_to(corpus)).replace("\\", "/")
        row["sha256"] = _sha256(target)
        converted += 1

    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    digest = hashlib.sha256()
    for sha in sorted(r["sha256"] for r in rows):
        digest.update(sha.encode())
    (corpus / "corpus_hash.txt").write_text(digest.hexdigest(), encoding="utf-8")

    after = _dir_size_mb(corpus)
    return {
        "pages_converted": converted,
        "size_before_mb": round(before, 1),
        "size_after_mb": round(after, 1),
        "reduction": f"{before / after:.1f}x" if after else "?",
        "corpus_sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/synthetic")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--keep-png", action="store_true")
    parser.add_argument("--archive", default=None,
                        help="also write a .zip at this path")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    stats = convert(corpus, args.quality, args.keep_png)
    print(json.dumps(stats, indent=2))

    if args.archive:
        archive = Path(args.archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(archive.with_suffix("")), "zip", str(corpus))
        size = archive.with_suffix(".zip").stat().st_size / 1024**2
        print(f"archive: {archive.with_suffix('.zip')}  ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

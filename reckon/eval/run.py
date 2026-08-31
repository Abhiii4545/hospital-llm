"""``make eval`` entrypoint.

Runs every available system over a dataset and writes
``reports/eval_<git-sha>.md``. In Phase 2 the systems are B0 and B1 and the
dataset is the 20-document mini-set. No model exists yet, and that is correct.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable, Sequence

from reckon.data.mini_set import MiniDoc, load_mini_set
from reckon.eval.metrics import DocumentPair
from reckon.eval.report import evaluate_system, write_report
from reckon.models.baselines.b0_rules import B0RulesEngine
from reckon.models.baselines.b1_ocr_heuristic import B1OcrHeuristic
from reckon.normalize import normalize_document
from reckon.provenance import set_global_seed
from reckon.schema import FIELD_PATHS, RawDocument

__all__ = ["run", "build_pairs", "ensemble_confidence", "MINI_SET_NOTES"]

#: Caveats printed at the top of every mini-set report. They are part of the
#: result, not a disclaimer to be dropped when the numbers look good.
MINI_SET_NOTES = (
    "**The mini-set is a harness smoke-test, not a benchmark.** It exists to prove "
    "the metrics run end to end before any corpus exists.",
    "Its text is noise-free, so **B1's OCR stage is never exercised** - B1's score "
    "here is an upper bound it will not reach on a scanned page.",
    "Five layouts is not layout variance, and layout variance is what broke v1.",
    "The renderer and both parsers were written by the same person on the same day, "
    "which is the classic way a baseline gets accidentally flattered.",
    "Quote no number from this report as a result for RECKON. Real numbers begin at "
    "Phase 3 (synthetic corpus) and Phase 6 (locked real-test).",
)


def build_pairs(
    docs: Sequence[MiniDoc], extractor: Callable[[str], RawDocument]
) -> list[DocumentPair]:
    return [
        DocumentPair(
            doc_id=doc.doc_id,
            pred=extractor(doc.text),
            truth=doc.truth,
            meta=doc.meta,
        )
        for doc in docs
    ]


def ensemble_confidence(
    pairs_by_system: dict[str, Sequence[DocumentPair]],
) -> dict[str, dict[str, float]]:
    """Per-field confidence from ensemble disagreement.

    The brief allows either a decoder sequence log-prob or an ensemble
    disagreement proxy. Rule engines have no log-prob, so this uses the proxy:
    a field where every system independently produced the same normalized value
    is treated as high confidence, one where they disagree as low, and one that
    nobody produced as zero.

    It is a genuine signal - independent extractors agreeing is evidence - but it
    is weaker than a calibrated model confidence, and the Phase 4 model must
    replace it rather than inherit it.
    """
    systems = list(pairs_by_system)
    if not systems:
        return {}

    by_doc: dict[str, dict[str, list[object]]] = {}
    for system in systems:
        for pair in pairs_by_system[system]:
            normalized = normalize_document(pair.pred)
            slot = by_doc.setdefault(pair.doc_id, {path: [] for path in FIELD_PATHS})
            for path in FIELD_PATHS:
                block, name = path.split(".", 1)
                slot[path].append(getattr(getattr(normalized, block), name))

    confidences: dict[str, dict[str, float]] = {}
    for doc_id, fields in by_doc.items():
        scores: dict[str, float] = {}
        for path, values in fields.items():
            present = [v for v in values if v is not None]
            if not present:
                scores[path] = 0.0
            elif len(set(map(str, present))) == 1 and len(present) == len(values):
                scores[path] = 1.0          # unanimous, and nobody abstained
            elif len(set(map(str, present))) == 1:
                scores[path] = 0.7          # agreed, but someone abstained
            else:
                scores[path] = 0.3          # genuine disagreement
        confidences[doc_id] = scores
    return confidences


def _dataset_hash(docs: Sequence[MiniDoc]) -> str:
    digest = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        digest.update(doc.doc_id.encode())
        digest.update(doc.text.encode())
        digest.update(doc.truth.model_dump_json().encode())
    return digest.hexdigest()


def run(
    data_dir: Path | str | None = None,
    reports_dir: Path | str = "reports",
    seed: int = 1337,
) -> Path:
    set_global_seed(seed)
    docs = load_mini_set(data_dir) if data_dir else load_mini_set()

    systems: dict[str, Callable[[str], RawDocument]] = {
        B0RulesEngine.name: B0RulesEngine().extract,
        B1OcrHeuristic.name: B1OcrHeuristic().extract,
    }
    pairs_by_system = {
        name: build_pairs(docs, extractor) for name, extractor in systems.items()
    }
    confidences = ensemble_confidence(pairs_by_system)

    results = [
        evaluate_system(name, pairs, confidences)
        for name, pairs in pairs_by_system.items()
    ]

    return write_report(
        results,
        dataset_name=f"phase-2 mini-set ({len(docs)} documents)",
        dataset_hash=_dataset_hash(docs),
        notes=MINI_SET_NOTES,
        reports_dir=reports_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    path = run(args.data_dir, args.reports_dir, args.seed)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

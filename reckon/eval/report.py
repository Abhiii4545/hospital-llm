"""Markdown evaluation report.

``make eval`` writes ``reports/eval_<git-sha>.md`` containing every table the
brief requires, for every system evaluated, plus the provenance block that makes
the numbers reproducible.

Two rules are enforced structurally rather than by good intentions:

* no headline average is emitted without the per-field breakdown beside it, and
* the report never presents one system's number without the baselines' numbers
  in the same table.

A report that flatters the model over the baselines is a bug in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from reckon.eval.metrics import (
    BusinessScore,
    CoveragePoint,
    DocumentPair,
    DocumentScore,
    EvalCache,
    FieldScore,
    LineItemScore,
    auto_processing_rate,
    coverage_curve,
    deployment_sentence,
    score_business,
    score_documents,
    score_fields,
    score_line_items,
)
from reckon.eval.slices import SliceSummary, slice_report
from reckon.provenance import run_metadata
from reckon.schema import LINE_ITEM_FIELDS

__all__ = ["SystemResult", "evaluate_system", "build_report", "write_report"]


@dataclass(frozen=True)
class SystemResult:
    name: str
    n: int
    fields: dict[str, FieldScore]
    line_items: LineItemScore
    documents: DocumentScore
    business: BusinessScore
    slices: dict[str, list[SliceSummary]]
    coverage: list[CoveragePoint]


def evaluate_system(
    name: str,
    pairs: Sequence[DocumentPair],
    confidences: Mapping[str, Mapping[str, float]] | None = None,
) -> SystemResult:
    """Run every metric family over one system's predictions.

    A single cache spans the whole system: every family and every slice would
    otherwise re-normalize and re-tree-diff the same documents.
    """
    cache = EvalCache()
    return SystemResult(
        name=name,
        n=len(pairs),
        fields=score_fields(pairs, cache),
        line_items=score_line_items(pairs, cache=cache),
        documents=score_documents(pairs, cache),
        business=score_business(pairs, cache=cache),
        slices=slice_report(pairs, cache),
        coverage=coverage_curve(pairs, confidences, cache=cache) if confidences else [],
    )


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rupees(value: Decimal) -> str:
    return f"Rs {value:,.2f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _ascii_curve(points: Sequence[CoveragePoint], height: int = 9) -> str:
    """Coverage-vs-accuracy plotted without a charting dependency."""
    usable = [p for p in points if p.field_coverage > 0]
    if not usable:
        return "_no coverage points with non-zero coverage_\n"

    grid = [[" " for _ in usable] for _ in range(height)]
    for x, point in enumerate(usable):
        y = height - 1 - round(point.field_accuracy * (height - 1))
        grid[max(0, min(height - 1, y))][x] = "*"

    lines = ["```", "accuracy"]
    for row_index, row in enumerate(grid):
        label = 1.0 - row_index / (height - 1)
        # rstrip: trailing whitespace would be stripped by the pre-commit
        # hook, making a regenerated report differ from the committed one.
        lines.append((f"{label:4.2f} |" + "".join(row)).rstrip())
    lines.append("     +" + "-" * len(usable))
    lines.append("      coverage: " + " -> ".join(_pct(p.field_coverage) for p in usable[:6]))
    lines.append("```")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------

def _headline_table(results: Sequence[SystemResult]) -> str:
    rows = []
    for r in results:
        n_fields = len(r.fields) or 1
        rows.append([
            r.name,
            str(r.n),
            _pct(sum(f.exact for f in r.fields.values()) / n_fields),
            _pct(sum(f.normalized_exact for f in r.fields.values()) / n_fields),
            _pct(r.line_items.f1),
            _pct(r.documents.ted_accuracy),
            _pct(r.documents.strict_exact_match),
            _rupees(r.business.median_error),
        ])
    return _table(
        ["system", "n", "field exact (raw)", "field exact (norm)",
         "line-item F1", "TED acc", "strict doc match", "median Rs error"],
        rows,
    )


def _per_field_table(results: Sequence[SystemResult]) -> str:
    if not results:
        return ""
    paths = list(results[0].fields)
    headers = ["field"]
    for r in results:
        headers += [f"{r.name} raw", f"{r.name} norm", f"{r.name} CER"]

    rows = []
    for path in paths:
        row = [f"`{path}`"]
        for r in results:
            score = r.fields[path]
            row += [_pct(score.exact), _pct(score.normalized_exact), f"{score.cer:.3f}"]
        rows.append(row)
    return _table(headers, rows)


def _line_item_table(results: Sequence[SystemResult]) -> str:
    rows = []
    for r in results:
        li = r.line_items
        rows.append([
            r.name, str(li.n_true), str(li.n_pred), str(li.n_matched),
            _pct(li.precision), _pct(li.recall), _pct(li.f1),
            str(li.insertions), str(li.deletions),
        ])
    return _table(
        ["system", "true rows", "pred rows", "matched",
         "precision", "recall", "F1", "insertions", "deletions"],
        rows,
    )


def _attribute_table(results: Sequence[SystemResult]) -> str:
    """Accuracy within matched rows, with each system's own support.

    Support differs per system because it is counted over MATCHED rows, and the
    systems match different numbers of rows. A single shared support column
    would silently attribute one system's denominator to all of them.
    """
    headers = ["attribute"]
    for r in results:
        headers += [r.name, f"{r.name} n"]
    rows = []
    for attribute in LINE_ITEM_FIELDS:
        row = [f"`{attribute}`"]
        for r in results:
            row.append(_pct(r.line_items.attribute_accuracy.get(attribute, 0.0)))
            row.append(str(r.line_items.attribute_support.get(attribute, 0)))
        rows.append(row)
    return _table(headers, rows)


def _business_table(results: Sequence[SystemResult]) -> str:
    rows = []
    for r in results:
        b = r.business
        rows.append([
            r.name, str(b.n_scored), _rupees(b.median_error), _rupees(b.p95_error),
            _pct(b.pct_error_over_threshold), _pct(b.pct_deduction_total_exact),
            _rupees(b.rupees_per_1000_claims),
        ])
    return _table(
        ["system", "n scored", "median abs err", "p95 abs err",
         "% err > Rs 100", "% deduction exact", "Rs per 1,000 claims"],
        rows,
    )


def _slice_sections(result: SystemResult) -> str:
    if not result.slices:
        return "_no slice metadata supplied_\n"
    out = []
    for key, summaries in result.slices.items():
        out.append(f"**by `{key}`**\n")
        rows = []
        for s in summaries:
            rows.append([
                s.value + (" _(underpowered)_" if s.underpowered else ""),
                str(s.n),
                _pct(s.field_normalized_macro),
                _pct(s.line_item_f1),
                _pct(s.ted_accuracy),
                _rupees(s.median_business_error),
            ])
        out.append(_table(
            ["value", "n", "field exact (norm)", "line-item F1", "TED acc", "median Rs error"],
            rows,
        ))
    return "\n".join(out)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_report(
    results: Sequence[SystemResult],
    *,
    dataset_name: str,
    dataset_hash: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    meta = run_metadata()
    lines: list[str] = []
    add = lines.append

    add(f"# RECKON evaluation - {dataset_name}\n")
    add(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    add("## Provenance\n")
    add(_table(
        ["key", "value"],
        [
            ["git SHA", f"`{meta['git_sha'] or 'unknown'}`"],
            ["working tree", "dirty" if meta["git_dirty"] else "clean"],
            ["dataset", dataset_name],
            ["dataset SHA-256", f"`{dataset_hash or 'n/a'}`"],
            ["python", meta["python"]],
            ["platform", meta["platform"]],
            ["packages", ", ".join(f"{k}=={v}" for k, v in sorted(meta["packages"].items()))],
        ],
    ))
    if meta["git_dirty"]:
        add(
            "> **Working tree is dirty.** The git SHA above does not fully describe "
            "the code that produced these numbers, so this run is NOT reproducible "
            "and must not go in the results table.\n"
        )

    if notes:
        add("## Read this before quoting any number\n")
        for note in notes:
            add(f"- {note}")
        add("")

    add("## Headline\n")
    add("Averages across fields are shown for orientation only. The per-field table")
    add("below is the real result: an average hides a collapsed field.\n")
    add(_headline_table(results))

    add("## Field level, per field\n")
    add(_per_field_table(results))

    add("## Line items\n")
    add(_line_item_table(results))
    add("\nInsertions (hallucinated rows) and deletions (dropped rows) are listed")
    add("separately: they cost an insurer different things.\n")
    add("### Per-attribute accuracy within matched rows\n")
    add(_attribute_table(results))

    add("## Document level\n")
    add(_table(
        ["system", "TED accuracy (norm)", "TED accuracy (raw)", "strict exact match"],
        [[r.name, _pct(r.documents.ted_accuracy), _pct(r.documents.ted_accuracy_raw),
          _pct(r.documents.strict_exact_match)] for r in results],
    ))
    add("\nStrict full-document exact match is expected to be brutally low. It is")
    add("reported because it is the only number that reflects a document needing no")
    add("human correction at all.\n")

    add("## Business impact\n")
    add(_business_table(results))

    add("## Coverage and abstention\n")
    for r in results:
        if not r.coverage:
            add(f"**{r.name}**: no confidence scores supplied, curve not computed.\n")
            continue
        add(f"**{r.name}**\n")
        add(_ascii_curve(r.coverage))
        add("")
        add(deployment_sentence(r.coverage, 0.95) + "\n")
        point = auto_processing_rate(r.coverage, 0.95)
        if point is None:
            add("_No threshold reaches 95% field accuracy on this set._\n")

    add("## Slices\n")
    for r in results:
        add(f"### {r.name}\n")
        add(_slice_sections(r))

    return "\n".join(lines)


def write_report(
    results: Sequence[SystemResult],
    *,
    dataset_name: str,
    dataset_hash: str | None = None,
    notes: Sequence[str] = (),
    reports_dir: Path | str = "reports",
) -> Path:
    """Write ``reports/eval_<git-sha>.md`` and return the path."""
    meta = run_metadata()
    sha = (meta["git_sha"] or "nogit")[:12]
    suffix = "-dirty" if meta["git_dirty"] else ""
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval_{sha}{suffix}.md"
    path.write_text(
        build_report(results, dataset_name=dataset_name,
                     dataset_hash=dataset_hash, notes=notes),
        encoding="utf-8",
    )
    return path

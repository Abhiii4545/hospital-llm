"""Per-page latency benchmark: p50/p95, per variant, on CPU and GPU.

The deliverable is a TRADE-OFF table, so this refuses to emit a latency row
without a slot for the accuracy that variant costs. A speed number on its own
reliably gets quoted as though it were free.

Warm-up runs are discarded. The first inference pays for lazy CUDA context
creation and ONNX graph optimisation, and including it inflates p50 by a factor
that has nothing to do with serving.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = ["LatencyResult", "measure", "benchmark", "render_table", "main"]

WARMUP = 3


@dataclass
class LatencyResult:
    variant: str
    device: str
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    size_mb: float | None = None
    #: Populated from an eval run against this variant. Deliberately not
    #: optional in the rendered table.
    accuracy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant, "device": self.device, "n": self.n,
            "p50_ms": round(self.p50_ms, 2), "p95_ms": round(self.p95_ms, 2),
            "mean_ms": round(self.mean_ms, 2), "size_mb": self.size_mb,
            "accuracy": self.accuracy or None,
        }


def measure(
    run_one: Callable[[Any], Any],
    inputs: Sequence[Any],
    warmup: int = WARMUP,
) -> tuple[float, float, float]:
    """Return (p50, p95, mean) in milliseconds, warm-up discarded."""
    for sample in inputs[:warmup]:
        run_one(sample)

    timings: list[float] = []
    for sample in inputs:
        started = time.perf_counter()
        run_one(sample)
        timings.append((time.perf_counter() - started) * 1000)

    timings.sort()
    index = max(0, min(len(timings) - 1, int(round(0.95 * len(timings))) - 1))
    return (
        statistics.median(timings),
        timings[index],
        statistics.fmean(timings),
    )


def _dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 1)


def benchmark(
    variants_dir: Path | str,
    images: Sequence[Path],
    devices: Sequence[str] = ("cpu",),
    accuracy_by_variant: dict[str, dict] | None = None,
) -> list[LatencyResult]:
    """Benchmark every exported variant found under *variants_dir*."""
    from PIL import Image

    variants_dir = Path(variants_dir)
    accuracy_by_variant = accuracy_by_variant or {}
    loaded = [Image.open(p).convert("RGB") for p in images]
    if not loaded:
        raise SystemExit("no images to benchmark")

    results: list[LatencyResult] = []
    for variant_dir in sorted(p for p in variants_dir.iterdir() if p.is_dir()):
        for device in devices:
            runner = _build_runner(variant_dir, device)
            if runner is None:
                continue
            p50, p95, mean = measure(runner, loaded)
            results.append(LatencyResult(
                variant=variant_dir.name, device=device, n=len(loaded),
                p50_ms=p50, p95_ms=p95, mean_ms=mean,
                size_mb=_dir_size_mb(variant_dir),
                accuracy=accuracy_by_variant.get(variant_dir.name, {}),
            ))
    return results


def _build_runner(variant_dir: Path, device: str):
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import DonutProcessor
    except ImportError:
        print(f"[bench] skipping {variant_dir.name}: optimum/onnxruntime absent")
        return None

    provider = "CUDAExecutionProvider" if device == "gpu" else "CPUExecutionProvider"
    try:
        model = ORTModelForVision2Seq.from_pretrained(variant_dir, provider=provider)
        processor = DonutProcessor.from_pretrained(variant_dir)
    except Exception as error:                              # noqa: BLE001
        print(f"[bench] skipping {variant_dir.name} on {device}: {error}")
        return None

    def run_one(image):
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return model.generate(pixel_values, max_length=768, num_beams=1)

    return run_one


def render_table(results: Sequence[LatencyResult]) -> str:
    """Markdown trade-off table. Missing accuracy is stated, not omitted."""
    if not results:
        return "_no variants benchmarked_\n"

    lines = ["| variant | device | size (MB) | p50 (ms) | p95 (ms) | "
             "field exact (norm) | line-item F1 | median Rs error |",
             "|---|---|---|---|---|---|---|---|"]
    for result in results:
        accuracy = result.accuracy or {}
        cells = [
            result.variant, result.device,
            f"{result.size_mb:.1f}" if result.size_mb else "?",
            f"{result.p50_ms:.0f}", f"{result.p95_ms:.0f}",
            _fmt(accuracy.get("field_exact_normalized")),
            _fmt(accuracy.get("line_item_f1")),
            _fmt(accuracy.get("median_rupee_error"), rupees=True),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if any(not r.accuracy for r in results):
        lines.append("")
        lines.append(
            "> **Accuracy columns are not yet measured for every variant.** A "
            "latency number without the accuracy it costs is not a result; run "
            "`make eval` against each exported variant before quoting this table."
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, rupees: bool = False) -> str:
    if value is None:
        return "**not measured**"
    if rupees:
        return f"Rs {float(value):,.2f}"
    return f"{float(value) * 100:.1f}%" if float(value) <= 1 else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-dir", default="outputs/onnx")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--accuracy-json", default=None,
                        help="per-variant accuracy measured by make eval")
    parser.add_argument("--out", default="reports/latency.md")
    args = parser.parse_args(argv)

    accuracy = (
        json.loads(Path(args.accuracy_json).read_text(encoding="utf-8"))
        if args.accuracy_json else {}
    )
    results = benchmark([*[Path(args.variants_dir)]][0],
                        [Path(p) for p in args.images],
                        args.devices, accuracy)
    table = render_table(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table, encoding="utf-8")
    print(table)
    print(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ONNX export with fp16 and dynamic-int8 variants.

Section 2.3 wants a trade-off table, not a single "we exported to ONNX" claim:
p50/p95 latency per page on CPU and GPU for every variant, **alongside the
accuracy each variant loses**. A quantised model that is 3x faster and 8 points
worse on `totals.net_amount` is not an improvement, and a latency table without
the matching accuracy column invites exactly that mistake.

So this module exports the variants, and `bench_latency.py` measures them, but
neither reports a speed number without a slot for the accuracy it cost.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Variant", "VARIANTS", "export", "export_all", "main"]


@dataclass(frozen=True)
class Variant:
    name: str
    precision: str          # fp32 | fp16 | int8
    description: str
    #: Filled in by evaluation, never by export. Present here so the trade-off
    #: table cannot be assembled with the accuracy column missing.
    accuracy_fields: tuple[str, ...] = (
        "field_exact_normalized", "line_item_f1", "median_rupee_error",
    )


VARIANTS: tuple[Variant, ...] = (
    Variant("onnx-fp32", "fp32", "ONNX baseline, no precision loss"),
    Variant("onnx-fp16", "fp16", "half precision; GPU-oriented"),
    Variant("onnx-int8-dynamic", "int8",
            "dynamic per-channel weight quantisation; CPU-oriented"),
)


def export(
    checkpoint: Path | str,
    out_dir: Path | str,
    variant: Variant,
    opset: int = 17,
) -> Path:
    """Export one variant. Raises with a useful message if deps are absent."""
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint}. Train Head B first: "
            "python -m reckon.training.train --config reckon/training/configs/head_b.yaml"
        )

    target = out_dir / variant.name
    target.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
    except ImportError as error:                            # pragma: no cover
        raise SystemExit(
            "ONNX export needs optimum and onnxruntime (both Apache-2.0):\n"
            "  uv add --optional onnx 'optimum[onnxruntime]' onnxruntime"
        ) from error

    model = ORTModelForVision2Seq.from_pretrained(
        checkpoint, export=True, use_cache=True
    )
    model.save_pretrained(target)

    for extra in ("preprocessor_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json"):
        source = checkpoint / extra
        if source.exists():
            shutil.copy2(source, target / extra)

    if variant.precision == "fp16":
        _convert_fp16(target)
    elif variant.precision == "int8":
        _quantize_int8(target)

    (target / "variant.json").write_text(
        json.dumps({
            "name": variant.name,
            "precision": variant.precision,
            "description": variant.description,
            "source_checkpoint": str(checkpoint),
            "size_bytes": _dir_size(target),
            "accuracy": {k: None for k in variant.accuracy_fields},
            "accuracy_note": (
                "null until measured by `make eval` against this variant. A "
                "latency number published without these is meaningless."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    return target


def _convert_fp16(directory: Path) -> None:
    from onnxconverter_common import float16
    import onnx

    for path in directory.glob("*.onnx"):
        model = onnx.load(str(path))
        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True),
                  str(path))


def _quantize_int8(directory: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for path in directory.glob("*.onnx"):
        quantize_dynamic(
            model_input=str(path),
            model_output=str(path),
            weight_type=QuantType.QInt8,
            # The encoder is a vision transformer; per-channel keeps more of the
            # accuracy that dynamic quantisation would otherwise throw away.
            per_channel=True,
            reduce_range=True,
        )


def _dir_size(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def export_all(checkpoint: Path | str, out_dir: Path | str) -> list[Path]:
    return [export(checkpoint, out_dir, variant) for variant in VARIANTS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/onnx")
    parser.add_argument("--variant", choices=[v.name for v in VARIANTS] + ["all"],
                        default="all")
    args = parser.parse_args(argv)

    if args.variant == "all":
        paths = export_all(args.checkpoint, args.out)
    else:
        chosen = next(v for v in VARIANTS if v.name == args.variant)
        paths = [export(args.checkpoint, args.out, chosen)]
    for path in paths:
        print(f"exported {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

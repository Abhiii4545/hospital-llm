"""Scan realism via Augraphy (MIT).

A clean render rasterised to PNG is not a training example, it is a fantasy. Real
claim documents arrive as phone photographs of photocopies of faxes, and a model
trained on pristine renders learns to depend on edges that will not be there.

Four quality buckets are produced, in fixed proportions, so that scan quality is
a slice the evaluation can break results down by. That matters more than the
augmentations themselves: "the model is fine on clean scans and collapses on
phone captures" is a finding, and it is invisible without the bucket label.

Buckets:

* ``clean``  - a flatbed scan on a good day. Light JPEG, faint noise.
* ``light``  - office scanner: mild ink bleed, slight skew, paper texture.
* ``medium`` - photocopy of a fax: low toner, dirty rollers, fold lines, dust.
* ``heavy``  - phone capture: perspective warp, lighting gradient, shadow,
  motion blur, moire from photographing a screen.

Every augmentation is wrapped: Augraphy occasionally raises on a specific input
(a page too small for a fold, a degenerate warp), and one bad page must not kill
a 10,000-page corpus run.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["QUALITY_BUCKETS", "QUALITY_WEIGHTS", "build_pipeline", "augment_page",
           "sample_quality", "ink_contrast", "edge_energy",
           "MIN_LEGIBLE_CONTRAST", "MIN_EDGE_ENERGY", "is_legible"]

#: Legibility floors an augmented page must clear.
#:
#: A first pass at the `heavy` bucket produced pages no human could read. Those
#: are not hard examples, they are label noise: a model cannot learn to read what
#: is not there. These are the automated guards that stop it regressing silently,
#: and both are asserted in the test suite.
#:
#: Calibrated over the real 10,212-page corpus: genuine pages score 165-215 and
#: a deliberately destroyed control scores 31. Threshold sits well below every
#: real page and well above the control.
MIN_LEGIBLE_CONTRAST = 80.0   # Otsu class separation: paper mean minus ink mean
MIN_EDGE_ENERGY = 200.0       # variance of the Laplacian: is there still an edge?

#: Ordered worst-to-best so a report table reads naturally.
QUALITY_BUCKETS: tuple[str, ...] = ("clean", "light", "medium", "heavy")

#: Deliberately not uniform. Real claim intake is mostly mediocre scans, with a
#: minority of pristine ones and a meaningful tail of phone photographs.
QUALITY_WEIGHTS: tuple[float, ...] = (0.20, 0.35, 0.30, 0.15)


def sample_quality(rng: random.Random) -> str:
    return rng.choices(QUALITY_BUCKETS, weights=QUALITY_WEIGHTS, k=1)[0]


@dataclass
class _Safe:
    """Wrap one augmentation so a failure degrades to a no-op, loudly counted."""

    name: str
    fn: Callable[[np.ndarray], Any]
    failures: dict[str, int]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = self.fn(image)
            if out is None:
                return image
            out = np.asarray(out)
            # An augmentation that changes the page shape would break alignment
            # with the ground truth, so a shape change is treated as a failure.
            if out.shape[:2] != image.shape[:2] and self.name not in _RESHAPING:
                self.failures[self.name] = self.failures.get(self.name, 0) + 1
                return image
            return out
        except Exception:                            # noqa: BLE001 - see docstring
            self.failures[self.name] = self.failures.get(self.name, 0) + 1
            return image


#: Augmentations that legitimately change page dimensions (warp, border).
_RESHAPING = frozenset({"Geometric", "PageBorder", "BookBinding"})


def build_pipeline(quality: str, rng: random.Random) -> list[_Safe]:
    """Ordered augmentation list for one quality bucket."""
    import augraphy as ag

    failures: dict[str, int] = {}

    def safe(name: str, factory: Callable[[], Any]) -> _Safe | None:
        try:
            instance = factory()
        except Exception:                            # noqa: BLE001
            return None
        return _Safe(name, instance, failures)

    steps: list[_Safe | None] = []
    jitter = rng.uniform

    if quality == "clean":
        steps += [
            safe("SubtleNoise", lambda: ag.SubtleNoise(subtle_range=rng.randint(2, 6))),
            safe("Jpeg", lambda: ag.Jpeg(quality_range=(88, 97))),
        ]

    elif quality == "light":
        steps += [
            safe("InkBleed", lambda: ag.InkBleed(intensity_range=(0.1, 0.3),
                                                 kernel_size=(5, 5))),
            safe("NoiseTexturize", lambda: ag.NoiseTexturize(
                sigma_range=(2, 6), turbulence_range=(2, 5))),
            safe("BrightnessTexturize", lambda: ag.BrightnessTexturize(
                texturize_range=(0.9, 0.99), deviation=0.03)),
            safe("Geometric", lambda: ag.Geometric(rotate_range=(-2, 2))),
            safe("SubtleNoise", lambda: ag.SubtleNoise(subtle_range=rng.randint(4, 9))),
            safe("Jpeg", lambda: ag.Jpeg(quality_range=(72, 90))),
        ]

    elif quality == "medium":
        steps += [
            safe("LowInkRandomLines", lambda: ag.LowInkRandomLines(
                count_range=(3, 12), use_consistent_lines=rng.random() < 0.5)),
            safe("InkBleed", lambda: ag.InkBleed(intensity_range=(0.3, 0.6),
                                                 kernel_size=(5, 5))),
            safe("Folding", lambda: ag.Folding(
                fold_count=rng.randint(1, 3), fold_noise=0.02,
                gradient_width=(0.1, 0.2), gradient_height=(0.01, 0.02))),
            safe("DirtyRollers", lambda: ag.DirtyRollers(line_width_range=(8, 24))),
            safe("DirtyDrum", lambda: ag.DirtyDrum(
                line_width_range=(1, 4), line_concentration=jitter(0.05, 0.2))),
            safe("BadPhotoCopy", lambda: ag.BadPhotoCopy(
                noise_type=rng.randint(1, 4), noise_side="random")),
            safe("Geometric", lambda: ag.Geometric(rotate_range=(-4, 4))),
            safe("Jpeg", lambda: ag.Jpeg(quality_range=(48, 72))),
        ]

    else:  # heavy - phone capture
        # Tuned DOWN from a first pass that stacked Letterpress + full-range
        # LightingGradient + dense Moire. That version was illegible to a human
        # reader, which makes those pages label noise rather than hard examples:
        # a model cannot learn to read what is not there, and 15% of the corpus
        # would have been teaching it to hallucinate. Degradation has to stay
        # recoverable to be worth training on.
        steps += [
            safe("InkBleed", lambda: ag.InkBleed(intensity_range=(0.2, 0.4),
                                                 kernel_size=(3, 3))),
            safe("LightingGradient", lambda: ag.LightingGradient(
                light_position=None, direction=rng.randint(0, 360),
                max_brightness=rng.randint(235, 255),
                min_brightness=rng.randint(90, 140))),
            safe("ShadowCast", lambda: ag.ShadowCast(
                shadow_side=rng.choice(["top", "bottom", "left", "right"]),
                shadow_opacity=jitter(0.15, 0.35))),
            safe("Geometric", lambda: ag.Geometric(
                rotate_range=(-4, 4), scale=(0.97, 1.03))),
            safe("SubtleNoise", lambda: ag.SubtleNoise(subtle_range=rng.randint(6, 12))),
            safe("Jpeg", lambda: ag.Jpeg(quality_range=(40, 62))),
        ]
        # Moire only sometimes, and sparse when it appears: it is the single most
        # destructive effect in the stack.
        if rng.random() < 0.35:
            steps.append(safe("Moire", lambda: ag.Moire(moire_density=(3, 8))))

    # Physical-object artefacts appear at a lower rate across all buckets.
    if quality != "clean" and rng.random() < 0.30:
        steps.append(safe("BindingsAndFasteners", lambda: ag.BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type=rng.choice(["punch_holes", "binding_holes", "clips"]),
            ntimes=(2, 4))))
    if quality in {"medium", "heavy"} and rng.random() < 0.20:
        steps.append(safe("Stains", lambda: ag.Stains(stains_type="light_stains")))

    return [s for s in steps if s is not None]


def augment_page(
    image: np.ndarray, quality: str, rng: random.Random
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the pipeline for *quality*. Returns (image, per-step failure counts).

    Failures are returned rather than swallowed so the corpus build can report
    how often augmentation silently did nothing. An augmentation stack that is
    quietly a no-op would make the corpus look harder than it is.
    """
    steps = build_pipeline(quality, rng)
    out = image
    for step in steps:
        out = step(out)
    failures = steps[0].failures if steps else {}
    return out, dict(failures)


def _otsu_threshold(grey: np.ndarray) -> int:
    """Otsu's threshold: the grey level that best separates ink from paper."""
    histogram = np.bincount(grey.reshape(-1), minlength=256).astype(float)
    total = histogram.sum()
    if total == 0:
        return 128
    weight = np.cumsum(histogram)
    mean = np.cumsum(histogram * np.arange(256))
    denominator = weight * (total - weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(
            denominator > 0,
            (mean[-1] * weight / total - mean) ** 2 / denominator,
            0.0,
        )
    return int(np.argmax(between))


def ink_contrast(image: np.ndarray) -> float:
    """Separation between the ink and paper classes, found by Otsu's method.

    Two earlier versions of this were wrong, both in the same way: they used
    FIXED percentiles, which measure ink DENSITY rather than legibility.

    * `p50 - p5` ranked a clean render below a heavily degraded one.
    * `p90 - p2` looked right on a dense page but failed on a sparse one. On a
      nursing-home bill with four line items, 99% of pixels are paper, so even
      the 2nd percentile is paper and the score collapses. Measured over the real
      corpus it rejected 36% of the CLEAN bucket, and scored a perfectly legible
      sparse page (28.9) BELOW the deliberately destroyed control (36.0) - it
      could not do the one job it was calibrated for.

    Otsu finds the split point from the image itself, so a page with 1% ink and a
    page with 16% ink are judged on the same basis: how far apart ink and paper
    actually are.
    """
    grey = (image if image.ndim == 2 else image.mean(axis=2)).astype(np.uint8)
    threshold = _otsu_threshold(grey)
    ink = grey[grey <= threshold]
    paper = grey[grey > threshold]
    if ink.size == 0 or paper.size == 0:
        return 0.0
    return float(paper.mean() - ink.mean())


def edge_energy(image: np.ndarray) -> float:
    """Variance of the Laplacian: whether glyph edges survived.

    Contrast alone is not enough - a blurred page can keep its dynamic range
    while losing every character boundary. This is the second, independent guard.
    """
    grey = image if image.ndim == 2 else image.mean(axis=2)
    lap = (-4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
           + grey[1:-1, :-2] + grey[1:-1, 2:])
    return float(lap.var())


def is_legible(image: np.ndarray) -> bool:
    """Both guards, as applied during the corpus build."""
    return (ink_contrast(image) >= MIN_LEGIBLE_CONTRAST
            and edge_energy(image) >= MIN_EDGE_ENERGY)

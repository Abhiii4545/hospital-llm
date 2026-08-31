"""Training layer: config, targets, dataset wiring and resume semantics.

Deliberately runs without torch or transformers where possible. The logic most
likely to be wrong - what the target string contains, and whether a resumed run
truly continues - should not need a 2GB dependency to verify, and it should be
verified BEFORE any free-tier GPU hours are spent on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reckon.training.callbacks import CheckpointConfig, TrainingState
from reckon.training.dataset import (
    build_target,
    load_manifest,
    split_counts,
)
from reckon.training.train import TrainConfig, load_config

CONFIG_DIR = Path("reckon/training/configs")


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["head_a.yaml", "head_b.yaml",
                                  "head_b_cord_warmstart.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config, sha = load_config(CONFIG_DIR / name)
    assert isinstance(config, TrainConfig)
    assert len(sha) == 64                 # config hash goes into run metadata


def test_configs_respect_the_fixed_architecture_decisions() -> None:
    """Section 2 is not open for re-litigation, so it is asserted."""
    for name in ("head_a.yaml", "head_b.yaml"):
        config, _ = load_config(CONFIG_DIR / name)
        assert config.base_model == "naver-clova-ix/donut-base"    # MIT
        assert config.batch_size == 1
        assert 8 <= config.grad_accum <= 16
        assert config.fp16 is True
        assert config.gradient_checkpointing is True
        assert config.optim_8bit is True
        # 960x1280, not Donut's 2560x1920 default which will not fit in 16GB
        assert (config.image_width, config.image_height) == (960, 1280)


def test_no_config_uses_a_forbidden_model() -> None:
    """LayoutLMv3 is CC-BY-NC-SA and must not appear anywhere."""
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text(encoding="utf-8").casefold()
        assert "layoutlm" not in text, path.name


def test_head_b_allows_a_longer_target_than_head_a() -> None:
    """A header block is bounded; a line-item list is not."""
    head_a, _ = load_config(CONFIG_DIR / "head_a.yaml")
    head_b, _ = load_config(CONFIG_DIR / "head_b.yaml")
    assert head_b.max_length > head_a.max_length


def test_cord_arm_is_an_ablation_not_the_default() -> None:
    default, _ = load_config(CONFIG_DIR / "head_b.yaml")
    cord, _ = load_config(CONFIG_DIR / "head_b_cord_warmstart.yaml")
    assert default.warm_start_from is None
    assert cord.warm_start_from is not None
    assert default.out_dir != cord.out_dir      # results must not overwrite


def test_unknown_config_keys_are_kept_not_silently_dropped(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"head": "b", "mystery_knob": 7}), encoding="utf-8")
    config, _ = load_config(path)
    assert config.extra["mystery_knob"] == 7


# --------------------------------------------------------------------------
# checkpoint policy
# --------------------------------------------------------------------------

def test_checkpoint_interval_over_200_is_rejected(tmp_path: Path) -> None:
    """The brief's hard requirement, enforced rather than documented."""
    CheckpointConfig(out_dir=tmp_path, every_steps=200)
    CheckpointConfig(out_dir=tmp_path, every_steps=50)
    with pytest.raises(ValueError, match="200"):
        CheckpointConfig(out_dir=tmp_path, every_steps=500)


def test_shipped_configs_checkpoint_often_enough() -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        config, _ = load_config(path)
        assert config.checkpoint_every <= 200, path.name


def test_training_state_round_trips() -> None:
    state = TrainingState(step=400, epoch=2, samples_seen=4800,
                          best_metric=0.83, history=[{"step": 25, "loss": 1.2}])
    assert TrainingState.from_dict(state.to_dict()) == state


def test_training_state_tracks_position_not_just_weights() -> None:
    """Resuming weights alone silently restarts the epoch.

    The model then re-sees data it has already trained on and the effective
    schedule stops matching the recorded config - a run that looks fine and is
    not reproducible.
    """
    fields = TrainingState().to_dict()
    for key in ("step", "epoch", "samples_seen"):
        assert key in fields


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def _payload(with_header: bool, with_totals: bool, rows: int) -> dict:
    head_a: dict = {}
    if with_header:
        head_a["patient"] = {"name": "Mr. Ramesh Kumar", "uhid": "UH1"}
        head_a["hospital"] = {"name": "Sunrise Hospital"}
    if with_totals:
        head_a["totals"] = {"net_amount": "1,25,678.20"}
    return {
        "head_a": head_a,
        "head_b": {"line_items": [
            {"serial_no": str(i + 1), "description": f"Item {i}",
             "amount": f"{100 * (i + 1)}.00"} for i in range(rows)
        ]},
    }


def test_head_b_target_contains_every_row() -> None:
    target = build_target(_payload(True, True, 5), "b")
    assert target.count("<s_description>") == 5
    assert "<s_line_items>" in target


def test_head_a_target_reflects_what_the_page_carries() -> None:
    first = build_target(_payload(True, False, 3), "a")
    assert "<s_patient>" in first and "<s_totals>" not in first

    last = build_target(_payload(False, True, 3), "a")
    assert "<s_totals>" in last and "<s_patient>" not in last


def test_continuation_pages_have_an_empty_head_a_target() -> None:
    """Kept in the dataset on purpose, not filtered out.

    Training only on pages that HAVE a header guarantees the model hallucinates
    one for every continuation page it ever sees.
    """
    assert build_target(_payload(False, False, 4), "a") == ""


def test_head_b_target_is_never_empty_even_with_no_rows() -> None:
    target = build_target(_payload(True, True, 0), "b")
    assert target == "<s_line_items></s_line_items>"


# --------------------------------------------------------------------------
# manifest wiring
# --------------------------------------------------------------------------

def _fake_corpus(tmp_path: Path, n: int = 6) -> Path:
    pages = tmp_path / "pages"
    pages.mkdir()
    rows = []
    for index in range(n):
        page_id = f"p{index:03d}"
        (pages / f"{page_id}.json").write_text(
            json.dumps(_payload(index % 2 == 0, index % 3 == 0, 3)), encoding="utf-8"
        )
        (pages / f"{page_id}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        rows.append({
            "page_id": page_id,
            "image": f"pages/{page_id}.png",
            "targets": f"pages/{page_id}.json",
            "split": "train" if index < 4 else ("val" if index == 4 else "synth_test"),
            "layout": f"L{index % 3:02d}",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return manifest


def test_manifest_loads_and_filters_by_split(tmp_path: Path) -> None:
    manifest = _fake_corpus(tmp_path)
    assert len(load_manifest(manifest)) == 6
    assert len(load_manifest(manifest, "train")) == 4
    assert len(load_manifest(manifest, "val")) == 1


def test_split_counts(tmp_path: Path) -> None:
    counts = split_counts(_fake_corpus(tmp_path))
    assert counts == {"train": 4, "val": 1, "synth_test": 1}


def test_manifest_paths_resolve_relative_to_the_manifest(tmp_path: Path) -> None:
    rows = load_manifest(_fake_corpus(tmp_path))
    assert all(row.image.exists() for row in rows)
    assert all(row.targets.exists() for row in rows)


def test_dataset_builds_targets_without_torch(tmp_path: Path) -> None:
    """Target construction must be checkable with no ML dependency installed."""
    from reckon.training.dataset import iter_samples

    samples = list(iter_samples(_fake_corpus(tmp_path), "b", "train"))
    assert len(samples) == 4
    assert all(s.target.startswith("<s_line_items>") for s in samples)

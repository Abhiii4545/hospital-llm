"""Resumable Donut fine-tuning, sized for a free-tier T4.

Run:
    python -m reckon.training.train --config reckon/training/configs/head_b.yaml

Memory budget is the binding constraint, not accuracy. Section 2.1 fixes the
configuration: fp16, gradient checkpointing, batch size 1, gradient accumulation
8-16, 8-bit AdamW, and 960x1280 input rather than Donut's 2560x1920 default,
which does not fit in 16GB.

Assume the session dies. Checkpoints go to Drive or the Hub every <=200 steps and
resume restores optimiser, scheduler, scaler and RNG state, not just weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reckon.provenance import file_sha256, run_metadata, set_global_seed
from reckon.schema import donut_special_tokens
from reckon.training.callbacks import (
    CheckpointConfig,
    CheckpointManager,
    TrainingState,
    skip_batches,
)
from reckon.training.dataset import PageDataset

__all__ = ["TrainConfig", "load_config", "train", "main"]


@dataclass
class TrainConfig:
    head: str = "b"
    base_model: str = "naver-clova-ix/donut-base"      # MIT
    manifest: str = "data/synthetic/manifest.jsonl"
    out_dir: str = "outputs/head_b"

    image_height: int = 1280
    image_width: int = 960
    max_length: int = 1024

    epochs: float = 3.0
    batch_size: int = 1
    grad_accum: int = 12
    lr: float = 3e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    fp16: bool = True
    gradient_checkpointing: bool = True
    optim_8bit: bool = True

    seed: int = 1337
    checkpoint_every: int = 200
    keep_last: int = 2
    drive_dir: str | None = None
    hub_repo: str | None = None
    push_to_hub: bool = False

    limit_train: int | None = None
    limit_val: int | None = None
    log_every: int = 25

    #: Optional CORD warm-start. Section 4.1 asks for this to be ABLATED, not
    #: assumed helpful - it may or may not transfer, and either result is worth
    #: reporting.
    warm_start_from: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> tuple[TrainConfig, str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f for f in TrainConfig.__dataclass_fields__}
    config = TrainConfig(**{k: v for k, v in payload.items() if k in known})
    config.extra = {k: v for k, v in payload.items() if k not in known}
    return config, file_sha256(path)


def _build_processor(config: TrainConfig):
    from transformers import DonutProcessor

    processor = DonutProcessor.from_pretrained(config.base_model)
    # Section 2.1: schema keys become single tokens BEFORE training. Skipping
    # this makes the target encoding strictly worse than plain JSON.
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": donut_special_tokens()}
    )
    processor.image_processor.size = {
        "height": config.image_height, "width": config.image_width
    }
    processor.image_processor.do_align_long_axis = False
    return processor


def _build_model(config: TrainConfig, processor):
    from transformers import VisionEncoderDecoderModel

    source = config.warm_start_from or config.base_model
    model = VisionEncoderDecoderModel.from_pretrained(source)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.encoder.image_size = [config.image_height, config.image_width]
    model.config.decoder.max_length = config.max_length
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<s_line_items>" if config.head == "b" else "<s_hospital>"
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def _build_optimizer(config: TrainConfig, model):
    import torch

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if any(k in name for k in ("bias", "norm")) else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if config.optim_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(groups, lr=config.lr)
        except ImportError:
            # Falling back is fine, but it must be visible: 8-bit AdamW is part
            # of what makes this fit in 16GB, and a silent fallback would show up
            # later as an unexplained OOM.
            print("[train] bitsandbytes unavailable; using fp32 AdamW "
                  "(expect higher memory use)", file=sys.stderr)
    return torch.optim.AdamW(groups, lr=config.lr)


def train(config: TrainConfig, config_sha: str = "") -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    set_global_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = _build_processor(config)
    model = _build_model(config, processor).to(device)

    common = dict(
        manifest=config.manifest, head=config.head, processor=processor,
        image_size=(config.image_height, config.image_width),
        max_length=config.max_length, seed=config.seed,
    )
    train_set = PageDataset(split="train", **common)
    val_set = PageDataset(split="val", **common)
    if config.limit_train:
        train_set.samples = train_set.samples[: config.limit_train]
    if config.limit_val:
        val_set.samples = val_set.samples[: config.limit_val]

    if not train_set.samples:
        raise SystemExit(f"no training pages in {config.manifest}")

    loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=device == "cuda",
        collate_fn=_collate,
    )

    steps_per_epoch = math.ceil(len(loader) / config.grad_accum)
    total_steps = max(1, int(steps_per_epoch * config.epochs))

    optimizer = _build_optimizer(config, model)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16 and device == "cuda")

    meta = run_metadata(seed=config.seed)
    meta["config_sha256"] = config_sha
    meta["head"] = config.head
    meta["total_steps"] = total_steps
    meta["train_pages"] = len(train_set)
    meta["val_pages"] = len(val_set)

    manager = CheckpointManager(
        CheckpointConfig(
            out_dir=Path(config.out_dir),
            every_steps=config.checkpoint_every,
            keep_last=config.keep_last,
            push_to_hub=config.push_to_hub,
            hub_repo=config.hub_repo,
            drive_dir=Path(config.drive_dir) if config.drive_dir else None,
        ),
        run_metadata=meta,
    )
    state = manager.resume(model, optimizer, scheduler, scaler)

    print(json.dumps({"device": device, "total_steps": total_steps,
                      "resumed_at": state.step, **meta}, indent=2, default=str))

    model.train()
    accumulated = 0
    iterator = skip_batches(loader, state.samples_seen % max(1, len(loader)))

    while state.step < total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            state.epoch += 1
            iterator = iter(loader)
            continue

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=config.fp16 and device == "cuda"):
            loss = model(pixel_values=pixel_values, labels=labels).loss / config.grad_accum

        scaler.scale(loss).backward()
        accumulated += 1
        state.samples_seen += pixel_values.size(0)

        if accumulated == config.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulated = 0
            state.step += 1

            if state.step % config.log_every == 0:
                entry = {
                    "step": state.step,
                    "loss": float(loss.item() * config.grad_accum),
                    "lr": scheduler.get_last_lr()[0],
                }
                state.history.append(entry)
                print(json.dumps(entry), flush=True)

            if manager.should_save(state.step):
                manager.save(state.step, state, model, processor, optimizer,
                             scheduler, scaler)

    final = manager.save(state.step, state, model, processor, optimizer,
                         scheduler, scaler)
    print(f"[train] finished at step {state.step}; final checkpoint {final}")
    return final


def _collate(batch: list[dict]) -> dict:
    import torch

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "page_id": [b["page_id"] for b in batch],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    args = parser.parse_args(argv)

    config, sha = load_config(args.config)
    if args.limit_train is not None:
        config.limit_train = args.limit_train
    if args.epochs is not None:
        config.epochs = args.epochs
    train(config, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

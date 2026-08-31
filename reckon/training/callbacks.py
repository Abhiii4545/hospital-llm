"""Checkpointing and resume.

Section 1 of the brief: assume any run will be killed mid-epoch. Colab
disconnects and Kaggle enforces a session limit, so a training loop that only
checkpoints at epoch boundaries will lose everything on a 3-hour run that dies at
2h50m.

Two rules this module exists to enforce:

* checkpoint at least every 200 optimiser steps, to Hub or Drive, not to the
  ephemeral local disk;
* resume must restore the optimiser, scheduler, scaler, RNG state and the exact
  dataloader position - not just the weights. Restoring weights alone silently
  restarts the epoch and re-trains on data the model has already seen, which
  looks like it worked and quietly changes the effective schedule.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CheckpointConfig", "CheckpointManager", "TrainingState"]


@dataclass
class CheckpointConfig:
    out_dir: Path
    every_steps: int = 200
    keep_last: int = 2
    push_to_hub: bool = False
    hub_repo: str | None = None
    drive_dir: Path | None = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if self.every_steps > 200:
            raise ValueError(
                "checkpoint interval must be <= 200 steps: a free-tier session "
                "can die at any moment and anything longer loses real work"
            )


@dataclass
class TrainingState:
    """Everything needed to continue as if nothing happened."""

    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    best_metric: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step, "epoch": self.epoch,
            "samples_seen": self.samples_seen,
            "best_metric": self.best_metric, "history": self.history,
        }

    @staticmethod
    def from_dict(payload: dict) -> "TrainingState":
        return TrainingState(
            step=payload.get("step", 0),
            epoch=payload.get("epoch", 0),
            samples_seen=payload.get("samples_seen", 0),
            best_metric=payload.get("best_metric"),
            history=payload.get("history", []),
        )


class CheckpointManager:
    """Save/restore a full training state, with copies off the ephemeral disk."""

    def __init__(self, config: CheckpointConfig, run_metadata: dict | None = None):
        self.config = config
        self.run_metadata = run_metadata or {}
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def _dir(self, step: int) -> Path:
        return self.config.out_dir / f"checkpoint-{step:07d}"

    def latest(self) -> Path | None:
        candidates = sorted(self.config.out_dir.glob("checkpoint-*"))
        return candidates[-1] if candidates else None

    # -- save ------------------------------------------------------------

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.config.every_steps == 0

    def save(
        self,
        step: int,
        state: TrainingState,
        model: Any = None,
        processor: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> Path:
        import torch

        target = self._dir(step)
        target.mkdir(parents=True, exist_ok=True)

        if model is not None:
            model.save_pretrained(target)
        if processor is not None:
            processor.save_pretrained(target)

        torch.save(
            {
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                # Without RNG state, resuming changes the augmentation and
                # shuffling stream, so the run is no longer reproducible.
                "rng_python": random.getstate(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            target / "trainer_state.pt",
        )

        (target / "state.json").write_text(
            json.dumps({**state.to_dict(), "run": self.run_metadata}, indent=2,
                       default=str),
            encoding="utf-8",
        )

        self._mirror(target)
        self._prune()
        return target

    def _mirror(self, target: Path) -> None:
        """Copy off the ephemeral disk. This is the point of the exercise."""
        if self.config.drive_dir:
            destination = Path(self.config.drive_dir) / target.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, destination, dirs_exist_ok=True)

        if self.config.push_to_hub and self.config.hub_repo:
            try:
                from huggingface_hub import HfApi

                HfApi().upload_folder(
                    folder_path=str(target),
                    repo_id=self.config.hub_repo,
                    path_in_repo=target.name,
                )
            except Exception as error:                      # noqa: BLE001
                # A failed upload must not kill a training run that is otherwise
                # fine; the local copy still exists.
                print(f"[checkpoint] hub upload failed: {error}")

    def _prune(self) -> None:
        checkpoints = sorted(self.config.out_dir.glob("checkpoint-*"))
        for stale in checkpoints[: max(0, len(checkpoints) - self.config.keep_last)]:
            shutil.rmtree(stale, ignore_errors=True)

    # -- resume ----------------------------------------------------------

    def resume(
        self,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        path: Path | None = None,
    ) -> TrainingState:
        """Restore everything. Returns a fresh state if there is nothing to resume."""
        import torch

        source = path or self.latest()
        if source is None:
            return TrainingState()

        state = TrainingState.from_dict(
            json.loads((source / "state.json").read_text(encoding="utf-8"))
        )

        blob_path = source / "trainer_state.pt"
        if blob_path.exists():
            blob = torch.load(blob_path, map_location="cpu", weights_only=False)
            if optimizer and blob.get("optimizer"):
                optimizer.load_state_dict(blob["optimizer"])
            if scheduler and blob.get("scheduler"):
                scheduler.load_state_dict(blob["scheduler"])
            if scaler and blob.get("scaler"):
                scaler.load_state_dict(blob["scaler"])
            if blob.get("rng_python"):
                random.setstate(blob["rng_python"])
            if blob.get("rng_torch") is not None:
                torch.set_rng_state(blob["rng_torch"])
            if blob.get("rng_cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(blob["rng_cuda"])

        print(f"[checkpoint] resumed from {source} at step {state.step}")
        return state


def skip_batches(loader: Any, n: int):
    """Fast-forward a dataloader to the position a resumed run left off at.

    Naive resumption restarts the epoch, so the model re-sees data and the
    effective learning-rate schedule no longer matches the recorded config. The
    run looks fine and is not reproducible.
    """
    iterator = iter(loader)
    for _ in range(n):
        try:
            next(iterator)
        except StopIteration:
            return iter(loader)
    return iterator

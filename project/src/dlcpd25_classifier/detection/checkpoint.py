"""Complete J2/J3 joint training checkpoint with deterministic loader state."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

SCHEMA_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_joint_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    global_step: int,
    next_task: str,
    classification_generator: torch.Generator,
    detection_generator: torch.Generator,
    classification_batches_into_cycle: int,
    detection_batches_into_cycle: int,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    if classification_batches_into_cycle != 0 or detection_batches_into_cycle != 0:
        raise ValueError("joint checkpoints may only be saved at loader-cycle boundaries")
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite joint checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "joint-resnet50-fasterrcnn",
        "classification_classes": 203,
        "detection_classes": 96,
        "image_size": 224,
        "global_step": global_step,
        "next_task": next_task,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "classification_loader_generator_state": classification_generator.get_state(),
        "detection_loader_generator_state": detection_generator.get_state(),
        "loader_cycle_position": {
            "classification_batches": classification_batches_into_cycle,
            "detection_batches": detection_batches_into_cycle,
        },
        "rng_state": capture_rng_state(),
        "metadata": metadata,
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def load_joint_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    classification_generator: torch.Generator | None = None,
    detection_generator: torch.Generator | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "joint-resnet50-fasterrcnn",
        "classification_classes": 203,
        "detection_classes": 96,
        "image_size": 224,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported joint checkpoint contract")
    if payload.get("loader_cycle_position") != {
        "classification_batches": 0,
        "detection_batches": 0,
    }:
        raise ValueError("joint checkpoint is not at a loader-cycle boundary")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if classification_generator is not None:
        classification_generator.set_state(
            payload["classification_loader_generator_state"].cpu()
        )
    if detection_generator is not None:
        detection_generator.set_state(payload["detection_loader_generator_state"].cpu())
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload

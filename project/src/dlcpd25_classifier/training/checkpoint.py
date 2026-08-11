"""Versioned model checkpoint save and load helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    *,
    architecture: str,
    num_classes: int,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": architecture,
        "num_classes": num_classes,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
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
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    expected_architecture: str | None = None,
    expected_num_classes: int | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if expected_architecture is not None and payload.get("architecture") != expected_architecture:
        raise ValueError("checkpoint architecture mismatch")
    if expected_num_classes is not None and payload.get("num_classes") != expected_num_classes:
        raise ValueError("checkpoint class count mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = payload.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return payload

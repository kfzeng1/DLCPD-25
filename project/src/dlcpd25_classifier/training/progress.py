"""Persist epoch-level training history as an append-only CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PROGRESS_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_images_per_second",
    "train_duration_seconds",
    "val_loss",
    "val_accuracy",
    "val_top5_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
    "val_images_per_second",
    "val_duration_seconds",
)


def progress_row(record: dict[str, Any]) -> dict[str, float | int]:
    train = record["train"]
    val = record["val"]
    return {
        "epoch": int(record["epoch"]),
        "learning_rate": float(record["learning_rate"]),
        "train_loss": float(train["loss"]),
        "train_accuracy": float(train["accuracy"]),
        "train_images_per_second": float(train["images_per_second"]),
        "train_duration_seconds": float(train["duration_seconds"]),
        "val_loss": float(val["loss"]),
        "val_accuracy": float(val["accuracy"]),
        "val_top5_accuracy": float(val["top5_accuracy"]),
        "val_macro_f1": float(val["macro_f1"]),
        "val_balanced_accuracy": float(val["balanced_accuracy"]),
        "val_images_per_second": float(val["images_per_second"]),
        "val_duration_seconds": float(val["duration_seconds"]),
    }


def append_progress(csv_path: Path, record: dict[str, Any]) -> bool:
    """Append one unseen epoch; return False when it already exists."""
    epoch = int(record["epoch"])
    existing_epochs: set[int] = set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != PROGRESS_FIELDS:
                raise ValueError(f"unexpected progress CSV schema: {reader.fieldnames}")
            existing_epochs = {int(row["epoch"]) for row in reader}
    if epoch in existing_epochs:
        return False
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS)
        if not existing_epochs:
            writer.writeheader()
        writer.writerow(progress_row(record))
        stream.flush()
    return True

"""Streaming metrics for the fixed 203-class classification task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ClassificationMetrics:
    num_classes: int = 203
    confusion: torch.Tensor = field(init=False)
    top5_correct: int = 0
    samples: int = 0

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(f"expected [N, {self.num_classes}] logits")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must be a one-dimensional batch")
        predictions = logits.argmax(dim=1)
        cpu_targets = targets.detach().to(device="cpu", dtype=torch.int64)
        cpu_predictions = predictions.detach().to(device="cpu", dtype=torch.int64)
        flat = cpu_targets * self.num_classes + cpu_predictions
        self.confusion += torch.bincount(
            flat, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        top_k = min(5, self.num_classes)
        top5 = logits.topk(top_k, dim=1).indices
        self.top5_correct += int((top5 == targets.unsqueeze(1)).any(dim=1).sum())
        self.samples += targets.numel()

    def compute(self) -> tuple[dict[str, float], list[dict[str, float | int]]]:
        if self.samples == 0:
            raise ValueError("cannot compute metrics without samples")
        matrix = self.confusion.to(torch.float64)
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
        precision = torch.where(
            predicted > 0, true_positive / predicted, torch.zeros_like(predicted)
        )
        denominator = precision + recall
        f1 = torch.where(
            denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator)
        )
        present = support > 0
        summary = {
            "accuracy": float(true_positive.sum() / self.samples),
            "top5_accuracy": self.top5_correct / self.samples,
            "macro_f1": float(f1[present].mean()),
            "balanced_accuracy": float(recall[present].mean()),
        }
        per_class: list[dict[str, float | int]] = []
        for class_id in range(self.num_classes):
            per_class.append(
                {
                    "class_id": class_id,
                    "support": int(support[class_id]),
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1": float(f1[class_id]),
                }
            )
        return summary, per_class

    def as_serializable_confusion(self) -> list[list[int]]:
        return self.confusion.tolist()


def validate_metric_payload(payload: dict[str, Any]) -> None:
    required = {"accuracy", "top5_accuracy", "macro_f1", "balanced_accuracy"}
    if not required.issubset(payload):
        raise ValueError(f"missing classification metrics: {sorted(required - payload.keys())}")
    if not all(0.0 <= float(payload[name]) <= 1.0 for name in required):
        raise ValueError("classification metrics must be within [0, 1]")


def epoch_duration_seconds(history: list[dict[str, Any]]) -> float:
    """Sum train and validation time across all recorded epochs."""
    return sum(
        float(record["train"]["duration_seconds"])
        + float(record["val"]["duration_seconds"])
        for record in history
    )

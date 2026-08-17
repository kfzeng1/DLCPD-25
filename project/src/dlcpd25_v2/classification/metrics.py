"""Classification metrics for long-tail DLCPD-25 evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ClassificationMetrics:
    top1: float
    top5: float
    macro_f1: float
    balanced_accuracy: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "top1": self.top1,
            "top5": self.top5,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "support": self.support,
        }


def accuracy_from_logits(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    if logits.numel() == 0:
        return [0.0 for _ in topk]
    max_k = max(topk)
    batch_size = targets.shape[0]
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    return [float(correct[:k].reshape(-1).float().sum(0) / max(batch_size, 1) * 100.0) for k in topk]


def _one_vs_rest_f1(confusion: Tensor, num_classes: int) -> float:
    tp = confusion.diagonal()
    true_sum = confusion.sum(dim=1)
    pred_sum = confusion.sum(dim=0)
    supported = true_sum > 0
    if not bool(supported.any()):
        return 0.0
    f1 = torch.zeros(num_classes, dtype=torch.float64, device=confusion.device)
    denominator = true_sum + pred_sum
    valid = supported & (denominator > 0)
    f1[valid] = 2.0 * tp[valid].to(torch.float64) / denominator[valid].to(torch.float64)
    return float(f1[supported].mean().item() * 100.0)


def macro_f1_from_confusion(confusion: Tensor) -> float:
    return _one_vs_rest_f1(confusion, confusion.shape[0])


def balanced_accuracy_from_confusion(confusion: Tensor) -> float:
    true_sum = confusion.sum(dim=1)
    supported = true_sum > 0
    if not bool(supported.any()):
        return 0.0
    recalls = torch.zeros(confusion.shape[0], dtype=torch.float64, device=confusion.device)
    recalls[supported] = confusion.diagonal()[supported].to(torch.float64) / true_sum[supported].to(torch.float64)
    return float(recalls[supported].mean().item() * 100.0)


def confusion_matrix_from_pairs(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    indices = targets * num_classes + predictions
    flat = torch.bincount(indices, minlength=num_classes * num_classes)
    return flat.reshape(num_classes, num_classes)

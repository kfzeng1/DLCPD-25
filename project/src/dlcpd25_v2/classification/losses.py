"""Losses for long-tail DLCPD-25 classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def focal_loss_with_smoothing(
    logits: Tensor,
    targets: Tensor,
    alpha: Tensor | None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Multi-class focal loss with optional uniform label smoothing.

    ``alpha`` is an optional per-class weight vector of shape ``[C]`` on the
    same device as ``logits``.
    """
    num_classes = logits.shape[1]
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()

    if label_smoothing > 0.0:
        epsilon = label_smoothing
        smooth = epsilon / max(num_classes - 1, 1)
        one_hot = torch.full_like(prob, smooth)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)
    else:
        one_hot = F.one_hot(targets, num_classes=num_classes).to(dtype=prob.dtype)

    true_prob = (one_hot * prob).sum(dim=1)
    focal_weight = (1.0 - true_prob).pow(gamma)
    per_sample = -(one_hot * log_prob).sum(dim=1) * focal_weight
    if alpha is not None:
        per_sample = per_sample * alpha[targets]
    return per_sample.mean()


def inverse_sqrt_class_weights(class_counts: dict[int, int], num_classes: int) -> Tensor:
    values = torch.ones(num_classes, dtype=torch.float32)
    for class_id, count in class_counts.items():
        if count > 0:
            values[class_id] = 1.0 / float(count) ** 0.5
    values = values / values.mean()
    return values


class ClassificationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_counts: dict[int, int],
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        host_weight: float = 0.2,
        category_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "alpha", inverse_sqrt_class_weights(class_counts, num_classes), persistent=True
        )
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.host_weight = host_weight
        self.category_weight = category_weight
        self.host_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.category_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(
        self,
        logits: Tensor,
        host_logits: Tensor,
        category_logits: Tensor,
        targets: Tensor,
        host_targets: Tensor,
        category_targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        main_loss = focal_loss_with_smoothing(
            logits,
            targets,
            self.alpha,
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
        )
        host_loss = self.host_ce(host_logits, host_targets)
        category_loss = self.category_ce(category_logits, category_targets)
        total = main_loss + self.host_weight * host_loss + self.category_weight * category_loss
        return total, {
            "loss": total.detach(),
            "loss_main": main_loss.detach(),
            "loss_host": host_loss.detach(),
            "loss_category": category_loss.detach(),
        }

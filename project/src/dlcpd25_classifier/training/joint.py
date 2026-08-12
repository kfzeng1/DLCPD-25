"""Shared mechanics for alternating classification and detection training."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor, nn


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


def collate_detection(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def set_task_trainability(model: nn.Module, task: str) -> None:
    """Enable the shared body and only the head used by the current step."""
    if task not in {"classification", "detection"}:
        raise ValueError(f"unknown joint-training task: {task}")
    for parameter in model.shared_body.parameters():
        parameter.requires_grad = True
    for parameter in model.classification_head.parameters():
        parameter.requires_grad = task == "classification"
    for module in model.detection_head:
        for parameter in module.parameters():
            parameter.requires_grad = task == "detection"


def build_joint_optimizer(
    model: nn.Module,
    *,
    backbone_learning_rate: float,
    classification_head_learning_rate: float,
    detection_head_learning_rate: float,
    weight_decay: float,
    fused: bool,
) -> torch.optim.AdamW:
    rates = {
        "backbone": backbone_learning_rate,
        "classification_head": classification_head_learning_rate,
        "detection_head": detection_head_learning_rate,
    }
    if any(rate <= 0 for rate in rates.values()):
        raise ValueError("joint optimizer learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("joint optimizer weight decay must be non-negative")
    detection_parameters = [
        parameter for module in model.detection_head for parameter in module.parameters()
    ]
    return torch.optim.AdamW(
        [
            {
                "params": list(model.shared_body.parameters()),
                "lr": backbone_learning_rate,
                "name": "backbone",
            },
            {
                "params": list(model.classification_head.parameters()),
                "lr": classification_head_learning_rate,
                "name": "classification_head",
            },
            {
                "params": detection_parameters,
                "lr": detection_head_learning_rate,
                "name": "detection_head",
            },
        ],
        weight_decay=weight_decay,
        fused=fused,
    )


def learning_rates_by_name(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name: Any = group.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("joint optimizer parameter groups require stable names")
        if name in result:
            raise ValueError(f"duplicate joint optimizer parameter group: {name}")
        result[name] = float(group["lr"])
    return result

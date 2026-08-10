"""Construct supported 203-class image classifiers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class ModelInfo:
    architecture: str
    num_classes: int
    pretrained: bool
    weights: str | None
    parameter_count: int
    trainable_parameter_count: int


def build_classification_model(
    name: str = "resnet50",
    num_classes: int = 203,
    pretrained: bool = True,
) -> tuple[nn.Module, ModelInfo]:
    """Build a classifier whose only learned output is the fine-grained class."""
    if name != "resnet50":
        raise ValueError(f"unsupported model architecture: {name}")
    if num_classes != 203:
        raise ValueError(f"DLCPD-25 requires num_classes=203, got {num_classes}")
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, ModelInfo(
        architecture=name,
        num_classes=num_classes,
        pretrained=pretrained,
        weights=weights.name if weights is not None else None,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    )

"""ConvNeXt-Tiny classifier with hierarchical auxiliary heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

SUPPORTED_ARCHITECTURES = {"convnext_tiny"}


@dataclass(frozen=True)
class ClassificationModelInfo:
    architecture: str
    num_classes: int
    num_hosts: int
    num_categories: int
    parameter_count: int
    trainable_parameter_count: int


class ConvNextClassifier(nn.Module):
    """ConvNeXt-Tiny features followed by main/aux classification heads."""

    def __init__(
        self,
        num_classes: int,
        num_hosts: int,
        num_categories: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 203:
            raise ValueError(f"DLCPD-25 requires num_classes=203, got {num_classes}")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.classifier[0]
        self.flatten = base.classifier[1]
        feature_dim = 768
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.host_classifier = nn.Linear(feature_dim, num_hosts)
        self.category_classifier = nn.Linear(feature_dim, num_categories)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.features(images)
        features = self.avgpool(features)
        features = self.norm(features)
        features = self.flatten(features)
        return (
            self.classifier(features),
            self.host_classifier(features),
            self.category_classifier(features),
        )


def build_model(
    architecture: str = "convnext_tiny",
    num_classes: int = 203,
    num_hosts: int = 22,
    num_categories: int = 4,
    pretrained: bool = True,
) -> tuple[nn.Module, ClassificationModelInfo]:
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    model = ConvNextClassifier(num_classes, num_hosts, num_categories, pretrained=pretrained)
    info = ClassificationModelInfo(
        architecture=architecture,
        num_classes=num_classes,
        num_hosts=num_hosts,
        num_categories=num_categories,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    )
    return model, info

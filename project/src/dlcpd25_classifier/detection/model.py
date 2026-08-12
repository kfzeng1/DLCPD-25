"""Shared ResNet-50 model for 203-class classification and IP102 detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d

from dlcpd25_classifier.models import build_classification_model

from .mapping import DetectionClassMapping


@dataclass(frozen=True)
class SharedModelInfo:
    backbone: str
    classification_classes: int
    detection_classes: int
    detector_classes_with_background: int


class SharedResNet50ClassifierDetector(nn.Module):
    """Own a frozen 203-class head and an FPN detector initialized from its backbone."""

    def __init__(self, detector: FasterRCNN, classification_head: nn.Linear) -> None:
        super().__init__()
        self.detector = detector
        self.classification_head = classification_head

    def forward_detection(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Any:
        detector_targets = None
        if targets is not None:
            detector_targets = [
                {key: value for key, value in target.items() if key in {"boxes", "labels", "image_id", "area", "iscrowd"}}
                for target in targets
            ]
        return self.detector(images, detector_targets)

    def forward_classification(self, images: Tensor) -> Tensor:
        """Classify already normalized 224x224 tensors with the shared backbone."""
        features = self.detector.backbone.body(images)
        layer4 = features["3"]
        pooled = F.adaptive_avg_pool2d(layer4, (1, 1)).flatten(1)
        return self.classification_head(pooled)

    def export_detections(
        self,
        predictions: list[dict[str, Tensor]],
        mapping: DetectionClassMapping,
    ) -> list[dict[str, Tensor]]:
        exported: list[dict[str, Tensor]] = []
        for prediction in predictions:
            labels = torch.tensor(
                [mapping.from_detector(int(label)).dlcpd25_class_id for label in prediction["labels"]],
                dtype=torch.int64,
                device=prediction["labels"].device,
            )
            exported.append({**prediction, "labels": labels})
        return exported


def build_shared_detection_model(
    classification_checkpoint: str | Path,
    mapping: DetectionClassMapping,
    *,
    trainable_backbone_layers: int = 0,
) -> tuple[SharedResNet50ClassifierDetector, SharedModelInfo]:
    """Load the released classifier and reuse its ResNet-50 weights in Faster R-CNN."""
    classifier, _ = build_classification_model(pretrained=False)
    payload = torch.load(Path(classification_checkpoint), map_location="cpu", weights_only=True)
    if payload.get("architecture") != "resnet50" or payload.get("num_classes") != 203:
        raise ValueError("classification checkpoint must be the 203-class ResNet-50 release")
    classifier.load_state_dict(payload["model_state_dict"], strict=True)

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        trainable_layers=trainable_backbone_layers,
        norm_layer=FrozenBatchNorm2d,
    )
    backbone_state = {
        name: value
        for name, value in classifier.state_dict().items()
        if not name.startswith("fc.") and not name.endswith("num_batches_tracked")
    }
    backbone.body.load_state_dict(backbone_state, strict=True)
    detector = FasterRCNN(backbone, num_classes=mapping.num_detector_classes + 1)
    classification_head = classifier.fc
    for parameter in classification_head.parameters():
        parameter.requires_grad = False
    model = SharedResNet50ClassifierDetector(detector, classification_head)
    return model, SharedModelInfo(
        backbone="resnet50-fpn",
        classification_classes=203,
        detection_classes=mapping.num_detector_classes,
        detector_classes_with_background=mapping.num_detector_classes + 1,
    )

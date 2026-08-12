"""One-pass ResNet-50 model for classification and IP102 detection."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.misc import FrozenBatchNorm2d

from dlcpd25_classifier.models import build_classification_model

from .mapping import DetectionClassMapping

IMAGE_SIZE = 224


def _validate_j1_checkpoint(payload: dict[str, Any]) -> None:
    if payload.get("architecture") != "resnet50" or payload.get("num_classes") != 203:
        raise ValueError("classification checkpoint must be a 203-class ResNet-50")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("stage") != "J1":
        raise ValueError("joint model initialization requires an accepted J1 checkpoint")
    preprocessing = metadata.get("preprocessing")
    expected_preprocessing = {
        "color_mode": "RGB",
        "image_size": IMAGE_SIZE,
        "resize": [IMAGE_SIZE, IMAGE_SIZE],
        "crop": "none",
        "preserve_aspect_ratio": False,
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    if preprocessing != expected_preprocessing:
        raise ValueError("J1 checkpoint does not use the joint 224 direct-resize contract")
    if metadata.get("test_metrics_read") is not False:
        raise ValueError("J1 checkpoint violates test-isolation metadata")


@dataclass(frozen=True)
class SharedModelInfo:
    backbone: str
    classification_classes: int
    detection_classes: int
    detector_classes_with_background: int
    image_size: int
    shared_body_forwards_per_joint_call: int


class SharedResNet50ClassifierDetector(nn.Module):
    """Route one normalized tensor through one shared body and two task heads."""

    def __init__(self, detector: FasterRCNN, classification_head: nn.Linear) -> None:
        super().__init__()
        self.detector = detector
        self.classification_head = classification_head

    @property
    def shared_body(self) -> nn.Module:
        return self.detector.backbone.body

    @property
    def detection_head(self) -> tuple[nn.Module, ...]:
        return (self.detector.backbone.fpn, self.detector.rpn, self.detector.roi_heads)

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("joint model requires an [N, 3, 224, 224] tensor")
        if images.shape[1] != 3 or not torch.is_floating_point(images):
            raise ValueError("joint model requires floating-point RGB tensors")

    @staticmethod
    def _detector_targets(
        targets: list[dict[str, Tensor]] | None,
    ) -> list[dict[str, Tensor]] | None:
        if targets is None:
            return None
        return [
            {
                key: value
                for key, value in target.items()
                if key in {"boxes", "labels", "image_id", "area", "iscrowd"}
            }
            for target in targets
        ]

    def extract_body_features(self, images: Tensor) -> OrderedDict[str, Tensor]:
        self._validate_images(images)
        features = self.shared_body(images)
        if not isinstance(features, OrderedDict) or "3" not in features:
            raise RuntimeError("shared ResNet body did not return layer1-layer4 features")
        return features

    def classify_features(self, body_features: OrderedDict[str, Tensor]) -> Tensor:
        pooled = F.adaptive_avg_pool2d(body_features["3"], (1, 1)).flatten(1)
        logits = self.classification_head(pooled)
        if logits.ndim != 2 or logits.shape[1] != 203:
            raise RuntimeError("classification head did not return [N, 203] logits")
        return logits

    def detect_features(
        self,
        images: Tensor,
        body_features: OrderedDict[str, Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Any:
        self._validate_images(images)
        detector_targets = self._detector_targets(targets)
        if self.training and detector_targets is None:
            raise ValueError("detection training requires targets")
        image_list = ImageList(
            images,
            [(IMAGE_SIZE, IMAGE_SIZE)] * images.shape[0],
        )
        fpn_features = self.detector.backbone.fpn(body_features)
        proposals, proposal_losses = self.detector.rpn(
            image_list, fpn_features, detector_targets
        )
        detections, detector_losses = self.detector.roi_heads(
            fpn_features,
            proposals,
            image_list.image_sizes,
            detector_targets,
        )
        losses = {**proposal_losses, **detector_losses}
        return losses if self.training else detections

    def forward_classification(self, images: Tensor) -> Tensor:
        return self.classify_features(self.extract_body_features(images))

    def forward_detection(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Any:
        features = self.extract_body_features(images)
        return self.detect_features(images, features, targets)

    def forward_joint(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[Tensor, Any]:
        features = self.extract_body_features(images)
        logits = self.classify_features(features)
        detections = self.detect_features(images, features, targets)
        return logits, detections

    def export_detections(
        self,
        predictions: list[dict[str, Tensor]],
        mapping: DetectionClassMapping,
    ) -> list[dict[str, Tensor]]:
        exported: list[dict[str, Tensor]] = []
        for prediction in predictions:
            labels = torch.tensor(
                [
                    mapping.from_detector(int(label)).dlcpd25_class_id
                    for label in prediction["labels"]
                ],
                dtype=torch.int64,
                device=prediction["labels"].device,
            )
            exported.append({**prediction, "labels": labels})
        return exported


def build_shared_detection_model(
    classification_checkpoint: str | Path,
    mapping: DetectionClassMapping,
    *,
    trainable_backbone_layers: int = 5,
) -> tuple[SharedResNet50ClassifierDetector, SharedModelInfo]:
    """Load the J1 classifier and initialize one shared ResNet-50 body."""
    classifier, _ = build_classification_model(pretrained=False)
    payload = torch.load(Path(classification_checkpoint), map_location="cpu", weights_only=True)
    _validate_j1_checkpoint(payload)
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
    detector = FasterRCNN(
        backbone,
        num_classes=mapping.num_detector_classes + 1,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )
    classification_head = classifier.fc
    for parameter in classification_head.parameters():
        parameter.requires_grad = True
    model = SharedResNet50ClassifierDetector(detector, classification_head)
    return model, SharedModelInfo(
        backbone="resnet50-fpn",
        classification_classes=203,
        detection_classes=mapping.num_detector_classes,
        detector_classes_with_background=mapping.num_detector_classes + 1,
        image_size=IMAGE_SIZE,
        shared_body_forwards_per_joint_call=1,
    )

"""ConvNeXt-Tiny-FPN + Faster R-CNN detection model for IP102."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from dlcpd25_v2.data.classification_dataset import IMAGENET_MEAN, IMAGENET_STD

DETECTOR_FOREGROUND_CLASSES = 96
DETECTOR_CLASSES_WITH_BACKGROUND = 97
IMAGE_SIZE = 640


@dataclass(frozen=True)
class DetectionModelInfo:
    backbone: str
    num_classes: int
    image_size: int
    parameter_count: int


def build_detection_model(
    num_classes: int = DETECTOR_CLASSES_WITH_BACKGROUND,
    image_size: int = IMAGE_SIZE,
    pretrained_backbone: bool = True,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 30,
    box_batch_size_per_img: int = 128,
    rpn_batch_size_per_img: int = 256,
) -> tuple[nn.Module, DetectionModelInfo]:
    if num_classes != DETECTOR_CLASSES_WITH_BACKGROUND:
        raise ValueError(f"IP102 detector requires num_classes={DETECTOR_CLASSES_WITH_BACKGROUND}, got {num_classes}")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    base = convnext_tiny(weights=weights)

    # ConvNeXt-T feature stages at strides 4/8/16/32.
    return_layers = {"1": "0", "3": "1", "5": "2", "7": "3"}
    backbone = BackboneWithFPN(
        backbone=base.features,
        return_layers=return_layers,
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=image_size,
        max_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        box_batch_size_per_img=box_batch_size_per_img,
        box_positive_fraction=0.25,
        rpn_batch_size_per_img=rpn_batch_size_per_img,
    )
    info = DetectionModelInfo(
        backbone="convnext-tiny-fpn",
        num_classes=num_classes,
        image_size=image_size,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    return model, info


def load_classification_backbone(checkpoint_path: Path | str, model: nn.Module) -> None:
    """Initialize detection backbone from the frozen DLCPD-25 classifier EMA weights."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema_payload = payload.get("ema") or {}
    state = ema_payload.get("ema_model_state_dict") if isinstance(ema_payload, dict) else None
    if state is None:
        state = payload["model_state_dict"]
    prefix = "features."
    backbone_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.backbone.body.load_state_dict(backbone_state, strict=False)
    if not backbone_state:
        raise RuntimeError(f"no backbone weights found in {checkpoint_path}")
    print(f"[init-backbone] loaded {len(backbone_state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")

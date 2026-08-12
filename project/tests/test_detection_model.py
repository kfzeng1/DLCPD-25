from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    build_shared_detection_model,
)
from torchvision.ops.misc import FrozenBatchNorm2d

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"
CHECKPOINT = ROOT / "artifacts/training/j1-direct-resize-ed09c0f/classification-init.pt"
HISTORICAL_CHECKPOINT = ROOT / "artifacts/releases/dlcpd25-resnet50-weighted-v1/best.pt"


def test_shared_model_reuses_j1_backbone_and_classification_head() -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, info = build_shared_detection_model(CHECKPOINT, mapping)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"]
    assert info.classification_classes == 203
    assert info.detection_classes == 96
    assert info.detector_classes_with_background == 97
    assert model.detector.roi_heads.box_predictor.cls_score.out_features == 97
    assert torch.equal(model.detector.backbone.body.conv1.weight, checkpoint["conv1.weight"])
    assert torch.equal(model.classification_head.weight, checkpoint["fc.weight"])
    assert all(parameter.requires_grad for parameter in model.classification_head.parameters())
    assert any(parameter.requires_grad for parameter in model.detector.backbone.body.parameters())
    assert isinstance(model.detector.backbone.body.bn1, FrozenBatchNorm2d)


def test_shared_model_rejects_historical_pre_j1_checkpoint() -> None:
    mapping = DetectionClassMapping(MAPPING)
    with pytest.raises(ValueError, match="J1"):
        build_shared_detection_model(HISTORICAL_CHECKPOINT, mapping)

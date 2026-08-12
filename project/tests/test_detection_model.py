from __future__ import annotations

from pathlib import Path

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from dlcpd25_classifier.detection import DetectionClassMapping, build_shared_detection_model

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"
CHECKPOINT = ROOT / "artifacts/releases/dlcpd25-resnet50-weighted-v1/best.pt"


def test_shared_model_reuses_released_backbone_and_classification_head() -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, info = build_shared_detection_model(CHECKPOINT, mapping)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"]
    assert info.classification_classes == 203
    assert info.detection_classes == 96
    assert info.detector_classes_with_background == 97
    assert model.detector.roi_heads.box_predictor.cls_score.out_features == 97
    assert torch.equal(model.detector.backbone.body.conv1.weight, checkpoint["conv1.weight"])
    assert torch.equal(model.classification_head.weight, checkpoint["fc.weight"])
    assert not any(parameter.requires_grad for parameter in model.classification_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.detector.backbone.body.parameters())
    assert isinstance(model.detector.backbone.body.bn1, FrozenBatchNorm2d)

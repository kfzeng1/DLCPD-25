from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    DirectResizeDetectionTransform,
    build_shared_detection_model,
)
from dlcpd25_classifier.detection.checkpoint import (
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from dlcpd25_classifier.training.j2 import _predictions_equal, _verify_frozen_inputs
from dlcpd25_classifier.training.train import load_config

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"
CHECKPOINT = (
    ROOT
    / "artifacts/training/j1-direct-resize-ed09c0f/classification-init.pt"
)
CONFIG = ROOT / "project/configs/j2.yaml"


def test_detection_resize_scales_boxes_and_normalizes() -> None:
    image = torch.zeros((3, 100, 400), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[40.0, 10.0, 200.0, 80.0]]),
        "area": torch.tensor([11200.0]),
    }
    transformed, updated = DirectResizeDetectionTransform()(image, target)
    assert transformed.shape == (3, 224, 224)
    assert updated["boxes"].tolist()[0] == pytest.approx(
        [22.4, 22.4, 112.0, 179.2]
    )
    assert updated["area"].item() == pytest.approx(14049.28, rel=1e-4)
    assert torch.isfinite(transformed).all()


def test_j2_frozen_inputs_are_pinned_and_train_only() -> None:
    config = load_config(CONFIG)
    actual = _verify_frozen_inputs(config, ROOT / "project")
    assert actual == config["input_sha256"]


def test_j2_rejects_wrong_input_checksum_and_nontrain_split() -> None:
    config = load_config(CONFIG)
    config["input_sha256"]["detection_mapping"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch: detection_mapping"):
        _verify_frozen_inputs(config, ROOT / "project")
    config = load_config(CONFIG)
    config["detection_train_split"] = "../artifacts/data/ip102-detection-v1/val.txt"
    with pytest.raises(ValueError, match="train.txt"):
        _verify_frozen_inputs(config, ROOT / "project")


def test_joint_forward_uses_shared_body_once(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, info = build_shared_detection_model(CHECKPOINT, mapping)
    model.eval()
    count = 0
    original = model.shared_body.forward

    def counted(images: torch.Tensor):
        nonlocal count
        count += 1
        return original(images)

    monkeypatch.setattr(model.shared_body, "forward", counted)
    images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        logits, detections = model.forward_joint(images)
    assert count == 1
    assert info.shared_body_forwards_per_joint_call == 1
    assert logits.shape == (1, 203)
    assert len(detections) == 1
    assert model.detector.transform.min_size == (224,)
    assert model.detector.transform.max_size == 224
    assert model.detector.transform.image_mean == [0.0, 0.0, 0.0]
    assert model.detector.transform.image_std == [1.0, 1.0, 1.0]


def test_joint_checkpoint_restores_training_and_loader_state(tmp_path: Path) -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, _ = build_shared_detection_model(CHECKPOINT, mapping)
    optimizer = torch.optim.SGD(
        [
            {"params": list(model.shared_body.parameters()), "lr": 0.01},
            {
                "params": list(model.classification_head.parameters())
                + [parameter for module in model.detection_head for parameter in module.parameters()],
                "lr": 0.1,
            },
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    classification_generator = torch.Generator().manual_seed(1)
    detection_generator = torch.Generator().manual_seed(2)
    expected_classification_state = classification_generator.get_state().clone()
    expected_detection_state = detection_generator.get_state().clone()
    path = tmp_path / "joint.pt"
    save_joint_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        global_step=7,
        next_task="detection",
        classification_generator=classification_generator,
        detection_generator=detection_generator,
        classification_batches_into_cycle=0,
        detection_batches_into_cycle=0,
        metadata={"stage": "J2"},
    )
    fresh, _ = build_shared_detection_model(CHECKPOINT, mapping)
    fresh_optimizer = torch.optim.SGD(
        [
            {"params": list(fresh.shared_body.parameters()), "lr": 0.01},
            {
                "params": list(fresh.classification_head.parameters())
                + [parameter for module in fresh.detection_head for parameter in module.parameters()],
                "lr": 0.1,
            },
        ]
    )
    fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)
    fresh_classification_generator = torch.Generator().manual_seed(99)
    fresh_detection_generator = torch.Generator().manual_seed(98)
    payload = load_joint_checkpoint(
        path,
        fresh,
        fresh_optimizer,
        fresh_scaler,
        fresh_classification_generator,
        fresh_detection_generator,
        restore_rng=False,
    )
    assert payload["global_step"] == 7
    assert payload["next_task"] == "detection"
    assert torch.equal(
        fresh_classification_generator.get_state(), expected_classification_state
    )
    assert torch.equal(fresh_detection_generator.get_state(), expected_detection_state)
    assert all(
        torch.equal(value, fresh.state_dict()[key])
        for key, value in model.state_dict().items()
    )


def test_joint_checkpoint_rejects_midcycle_save(tmp_path: Path) -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, _ = build_shared_detection_model(CHECKPOINT, mapping)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(ValueError, match="loader-cycle boundaries"):
        save_joint_checkpoint(
            tmp_path / "joint.pt",
            model,
            optimizer,
            scaler,
            global_step=1,
            next_task="detection",
            classification_generator=torch.Generator().manual_seed(1),
            detection_generator=torch.Generator().manual_seed(2),
            classification_batches_into_cycle=1,
            detection_batches_into_cycle=0,
            metadata={"stage": "J2"},
        )


def test_joint_model_rejects_non_224_input() -> None:
    mapping = DetectionClassMapping(MAPPING)
    model, _ = build_shared_detection_model(CHECKPOINT, mapping)
    with pytest.raises(ValueError, match="224"):
        model.forward_classification(torch.randn(1, 3, 256, 256))


def test_prediction_equality_checks_all_detection_tensors() -> None:
    prediction = {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    copied = [{key: value.clone() for key, value in prediction.items()}]
    assert _predictions_equal([prediction], copied)
    copied[0]["scores"][0] = 0.8
    assert not _predictions_equal([prediction], copied)

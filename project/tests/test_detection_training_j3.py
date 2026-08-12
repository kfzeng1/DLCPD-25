from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from dlcpd25_classifier.detection.evaluation import evaluate_detection
from dlcpd25_classifier.training.j3 import (
    DetectionBatchStream,
    _json_scalar,
    _load_j3_checkpoint,
    _save_j3_checkpoint,
    _validate_config,
    checkpoint_is_better,
    checkpoint_is_eligible,
    evaluate_joint_classification,
)
from dlcpd25_classifier.training.joint import (
    build_joint_optimizer,
    capture_rng_state,
    collate_detection,
    learning_rates_by_name,
    restore_rng_state,
)
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


class TinyDetectionDataset(Dataset):
    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int):
        image = torch.full((3, 224, 224), float(index))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(index),
            "area": torch.tensor([400.0]),
            "iscrowd": torch.tensor([0]),
        }
        return image, target


class FixedDetector(nn.Module):
    def forward_detection(self, images: torch.Tensor):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=images.device),
                "labels": torch.tensor([1], device=images.device),
                "scores": torch.tensor([0.99], device=images.device),
            }
            for _ in images
        ]


class FixedJointClassifier(nn.Module):
    def forward_classification(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(images), 203), device=images.device)
        logits[:, 0] = 10.0
        return logits


class TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_body = nn.Linear(2, 2)
        self.classification_head = nn.Linear(2, 203)
        self.detection_head = (nn.Linear(2, 2),)


def test_j3_config_freezes_ratio_learning_rates_and_test_isolation() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "j3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    assert config["training"]["task_ratio"] == [1, 1]
    assert config["training"]["classification_amp"] is True
    assert config["training"]["detection_amp"] is False
    assert config["training"]["backbone_learning_rate"] == 1e-5
    assert config["training"]["classification_head_learning_rate"] == 1e-5
    assert config["training"]["detection_head_learning_rate"] == 1e-4
    assert config["evaluation"]["classification_top1_abort_threshold"] == 0.85
    assert config["evaluation"]["test_metrics_read"] is False
    assert "joint_initial_checkpoint" not in config
    assert "joint_initial_checkpoint" not in config["input_sha256"]
    assert "test" not in Path(config["classification_val_csv"]).name
    assert "test" not in Path(config["detection_val_split"]).name


def test_j3_artifact_contract_keeps_initial_classification_gate() -> None:
    source = (
        Path(__file__).parents[1]
        / "src/dlcpd25_classifier/training/j3.py"
    ).read_text(encoding="utf-8")
    assert "initial-classification-validation.json" in source
    assert "blocked_initial_classification_below_eligibility_gate" in source


def test_j3_preflight_is_explicitly_read_only() -> None:
    source = (
        Path(__file__).parents[1]
        / "src/dlcpd25_classifier/training/j3.py"
    ).read_text(encoding="utf-8")
    assert "def run_j3_preflight" in source
    assert "if args.preflight_only:" in source
    assert '"artifacts_written": False' in source
    assert '"training_steps_executed": 0' in source


def test_j3_selection_applies_classification_gate_before_map() -> None:
    threshold = 0.8878365948236992
    ineligible = {"classification": {"accuracy": threshold - 1e-5}, "detection": {"map": 0.99}}
    eligible = {"classification": {"accuracy": threshold}, "detection": {"map": 0.10}}
    stronger = {"classification": {"accuracy": 0.90}, "detection": {"map": 0.11}}
    assert not checkpoint_is_eligible(ineligible["classification"]["accuracy"], threshold)
    assert checkpoint_is_better(eligible, None, threshold)
    assert not checkpoint_is_better(ineligible, eligible, threshold)
    assert checkpoint_is_better(stronger, eligible, threshold)


def test_joint_optimizer_has_separate_classification_and_detection_rates() -> None:
    optimizer = build_joint_optimizer(
        TinyJointModel(),
        backbone_learning_rate=1e-5,
        classification_head_learning_rate=1e-5,
        detection_head_learning_rate=1e-4,
        weight_decay=1e-4,
        fused=False,
    )
    assert learning_rates_by_name(optimizer) == {
        "backbone": 1e-5,
        "classification_head": 1e-5,
        "detection_head": 1e-4,
    }


def test_joint_rng_state_round_trip() -> None:
    torch.manual_seed(123)
    expected = torch.rand(4)
    state = capture_rng_state()
    actual = torch.rand(4)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), actual)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), actual)
    assert not torch.equal(expected, actual)


def test_detection_stream_resumes_exact_next_samples() -> None:
    dataset = TinyDetectionDataset()
    stream = DetectionBatchStream(dataset, batch_size=2, workers=0, seed=41)
    stream.next()
    stream.next()
    state = stream.state_dict()
    expected = [stream.next()[0][:, 0, 0, 0].tolist() for _ in range(5)]
    resumed = DetectionBatchStream(dataset, batch_size=2, workers=0, seed=999)
    resumed.load_state_dict(state)
    actual = [resumed.next()[0][:, 0, 0, 0].tolist() for _ in range(5)]
    assert actual == expected


def test_j3_checkpoint_round_trip_restores_full_training_state(tmp_path: Path) -> None:
    model = TinyJointModel()
    optimizer = build_joint_optimizer(
        model,
        backbone_learning_rate=1e-5,
        classification_head_learning_rate=1e-5,
        detection_head_learning_rate=1e-4,
        weight_decay=1e-4,
        fused=False,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stream = DetectionBatchStream(
        TinyDetectionDataset(), batch_size=2, workers=0, seed=41
    )
    stream.next()
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_stream = stream.state_dict()
    checkpoint = tmp_path / "joint-last.pt"
    _save_j3_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
        stream,
        epoch=2,
        global_pair=19,
        history=[{"epoch": 2, "eligible": True}],
        metadata={"stage": "J3"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    stream.next()
    payload = _load_j3_checkpoint(
        checkpoint, model, optimizer, scheduler, scaler, stream
    )
    assert payload["completed_epoch"] == 2
    assert payload["global_pair"] == 19
    assert payload["metadata"] == {"stage": "J3"}
    assert all(
        torch.equal(model.state_dict()[name], expected)
        for name, expected in expected_parameters.items()
    )
    assert stream.cursor == expected_stream["cursor"]
    assert stream.permutation == expected_stream["permutation"]


def test_coco_evaluation_reports_standard_and_operating_metrics() -> None:
    loader = DataLoader(
        TinyDetectionDataset(), batch_size=2, shuffle=False, collate_fn=collate_detection
    )
    summary, per_class = evaluate_detection(
        FixedDetector(), loader, torch.device("cpu"), score_threshold=0.5,
        class_names={1: "class one"}, train_object_counts={1: 7},
    )
    assert summary["implementation"] == "pycocotools COCOeval bbox"
    assert summary["map"] == 1.0
    assert summary["ap50"] == 1.0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["object_size_support"] == {"small": 7}
    assert per_class[0]["train_objects"] == 7
    assert per_class[0]["ap"] == 1.0


def test_joint_classification_evaluation_uses_explicit_task_forward() -> None:
    images = torch.zeros((3, 3, 224, 224))
    targets = torch.zeros(3, dtype=torch.int64)
    loader = DataLoader(list(zip(images, targets)), batch_size=2)
    summary, per_class, confusion = evaluate_joint_classification(
        FixedJointClassifier(), loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert summary["accuracy"] == 1.0
    assert summary["top5_accuracy"] == 1.0
    assert summary["samples"] == 3.0
    assert per_class[0]["support"] == 3
    assert confusion[0][0] == 3


def test_coco_numpy_scalars_are_json_serializable() -> None:
    numpy = __import__("numpy")
    payload = {"label": numpy.int64(7), "score": numpy.float32(0.5)}
    encoded = json.dumps(payload, default=_json_scalar)
    assert json.loads(encoded) == {"label": 7, "score": 0.5}

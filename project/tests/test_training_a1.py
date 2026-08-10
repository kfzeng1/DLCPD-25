from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.optim import SGD

from dlcpd25_classifier.data import SplitRecord
from dlcpd25_classifier.models import build_classification_model
from dlcpd25_classifier.training.checkpoint import load_checkpoint, save_checkpoint
from dlcpd25_classifier.training.train import select_fixed_balanced_indices
from dlcpd25_classifier.training.transforms import (
    build_eval_transform,
    build_train_transform,
    preprocessing_spec,
)


def test_resnet50_outputs_finite_203_class_probabilities() -> None:
    model, info = build_classification_model(pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224))
        probabilities = logits.softmax(dim=1)
    assert info.architecture == "resnet50"
    assert info.num_classes == 203
    assert logits.shape == (2, 203)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_train_and_eval_transforms_return_finite_fixed_shape() -> None:
    image = Image.new("RGB", (360, 280), (60, 120, 180))
    for transform in (build_train_transform(), build_eval_transform()):
        tensor = transform(image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    spec = preprocessing_spec()
    assert spec["color_mode"] == "RGB"
    assert spec["image_size"] == 224
    assert len(spec["mean"]) == len(spec["std"]) == 3


def test_checkpoint_round_trip_and_contract_validation(tmp_path: Path) -> None:
    model = nn.Linear(4, 3)
    optimizer = SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        architecture="test-model",
        num_classes=3,
        epoch=2,
        metrics={"loss": 0.25},
        metadata={"stage": "A1"},
    )
    reloaded = nn.Linear(4, 3)
    payload = load_checkpoint(
        checkpoint,
        reloaded,
        expected_architecture="test-model",
        expected_num_classes=3,
    )
    assert payload["epoch"] == 2
    for expected, actual in zip(model.parameters(), reloaded.parameters(), strict=True):
        assert torch.equal(expected, actual)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            architecture="test-model",
            num_classes=3,
            epoch=3,
            metrics={},
            metadata={},
        )


class RecordDataset:
    def __init__(self) -> None:
        self.records = tuple(
            SplitRecord(
                relative_path=f"class-{class_id}/{index}.jpg",
                class_id=class_id,
                sha256="a" * 64,
                duplicate_group_id=f"group-{class_id}-{index}",
                split="train",
            )
            for class_id in range(10)
            for index in range(6)
        )


def test_fixed_subset_is_balanced_deterministic_and_bounded() -> None:
    dataset = RecordDataset()
    first = select_fixed_balanced_indices(dataset, sample_count=32, class_count=8)  # type: ignore[arg-type]
    second = select_fixed_balanced_indices(dataset, sample_count=32, class_count=8)  # type: ignore[arg-type]
    assert first == second
    assert len(first) == 32
    labels = [dataset.records[index].class_id for index in first]
    assert sorted(set(labels)) == list(range(8))
    assert all(labels.count(class_id) == 4 for class_id in range(8))
    with pytest.raises(ValueError, match="between 32 and 64"):
        select_fixed_balanced_indices(dataset, sample_count=16, class_count=8)  # type: ignore[arg-type]

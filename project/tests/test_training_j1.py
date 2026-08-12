from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dlcpd25_classifier.training.j1 import _build_dataset, _validate_config
from dlcpd25_classifier.training.transforms import (
    build_direct_resize_eval_transform,
    build_direct_resize_train_transform,
    direct_resize_preprocessing_spec,
)
from PIL import Image
from torchvision import transforms


def test_direct_resize_eval_uses_full_image_without_crop() -> None:
    transform = build_direct_resize_eval_transform(224)
    assert [type(step) for step in transform.transforms] == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    resize = transform.transforms[0]
    assert resize.size == (224, 224)
    image = Image.new("RGB", (480, 120), color=(10, 20, 30))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_direct_resize_train_has_no_crop() -> None:
    transform = build_direct_resize_train_transform(224)
    assert not any(isinstance(step, transforms.RandomResizedCrop) for step in transform.transforms)
    assert not any(isinstance(step, transforms.CenterCrop) for step in transform.transforms)
    assert isinstance(transform.transforms[0], transforms.Resize)
    assert transform.transforms[0].size == (224, 224)


def test_direct_resize_spec_is_joint_contract() -> None:
    assert direct_resize_preprocessing_spec() == {
        "color_mode": "RGB",
        "image_size": 224,
        "resize": [224, 224],
        "crop": "none",
        "preserve_aspect_ratio": False,
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def test_j1_config_freezes_selection_and_test_isolation() -> None:
    config = {
        "model": {"name": "resnet50", "num_classes": 203, "image_size": 224},
        "training": {
            "class_weighting": "inverse_frequency_clipped_at_10_then_mean_normalized"
        },
        "evaluation": {"selection_metric": "val_macro_f1", "test_metrics_read": False},
    }
    _validate_config(config)
    config["evaluation"]["selection_metric"] = "val_accuracy"
    with pytest.raises(ValueError, match="Macro-F1"):
        _validate_config(config)


def test_j1_dataset_rejects_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only permits"):
        _build_dataset(tmp_path, tmp_path / "test.csv", tmp_path / "taxonomy.json", None)

"""Shared train and deterministic image preprocessing definitions."""

from __future__ import annotations

from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocessing_spec(image_size: int = 224) -> dict[str, Any]:
    return {
        "color_mode": "RGB",
        "image_size": image_size,
        "eval_resize": int(round(image_size / 0.875)),
        "eval_crop": "center",
        "interpolation": "bicubic",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }

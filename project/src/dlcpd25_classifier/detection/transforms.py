"""Synchronized 224x224 preprocessing for IP102 images and boxes."""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from dlcpd25_classifier.training.transforms import IMAGENET_MEAN, IMAGENET_STD


class DirectResizeDetectionTransform:
    def __init__(self, image_size: int = 224) -> None:
        if image_size != 224:
            raise ValueError("joint detection input is fixed at 224")
        self.image_size = image_size

    def __call__(
        self, image: Tensor, target: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("detection transform requires a CHW RGB tensor")
        original_height, original_width = image.shape[-2:]
        if original_height <= 0 or original_width <= 0:
            raise ValueError("image dimensions must be positive")
        resized = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        normalized = F.normalize(resized, IMAGENET_MEAN, IMAGENET_STD)
        boxes = target["boxes"].clone()
        boxes[:, 0::2] *= self.image_size / original_width
        boxes[:, 1::2] *= self.image_size / original_height
        boxes[:, 0::2].clamp_(0, self.image_size)
        boxes[:, 1::2].clamp_(0, self.image_size)
        updated = {**target, "boxes": boxes}
        updated["area"] = (boxes[:, 2] - boxes[:, 0]) * (
            boxes[:, 3] - boxes[:, 1]
        )
        if not torch.isfinite(normalized).all() or not torch.isfinite(boxes).all():
            raise RuntimeError("detection preprocessing produced non-finite tensors")
        if not ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all():
            raise RuntimeError("detection preprocessing produced invalid boxes")
        return normalized, updated

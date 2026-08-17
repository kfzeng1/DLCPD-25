"""Image transforms for DLCPD-25 Plan-A classification."""

from __future__ import annotations

from torchvision import transforms

from dlcpd25_v2.data.classification_dataset import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=9,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

#!/usr/bin/env python3
"""Traverse every frozen T0 sample through the project detection Dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "project/src"))

from dlcpd25_classifier.detection import IP102DetectionDataset

DEFAULT_VOC_ROOT = REPO_ROOT / "data/raw/ip102/downloads/Detection/VOC2007"
DEFAULT_CONTRACT = REPO_ROOT / "artifacts/data/ip102-detection-v1"
EXPECTED = {"train": 12_142, "val": 3_036, "test": 3_798}


def validate_sample(sample: tuple[torch.Tensor, dict[str, torch.Tensor]]) -> dict[str, int]:
    image, target = sample
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"invalid image tensor shape or dtype: {image.shape}, {image.dtype}")
    stride = max(image.numel() // 64, 1)
    if not torch.isfinite(image.reshape(-1)[::stride]).all():
        raise ValueError("image tensor contains non-finite sampled pixels")
    boxes = target["boxes"]
    labels = target["labels"]
    public_ids = target["dlcpd25_class_ids"]
    source_ids = target["ip102_class_ids"]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
        raise ValueError("target has invalid or empty boxes")
    if not (boxes.shape[0] == labels.shape[0] == public_ids.shape[0] == source_ids.shape[0]):
        raise ValueError("target label spaces have inconsistent lengths")
    if not torch.isfinite(boxes).all() or not torch.all(boxes[:, 0] < boxes[:, 2]) or not torch.all(boxes[:, 1] < boxes[:, 3]):
        raise ValueError("target has invalid effective boxes")
    if not torch.all(labels.ge(1)) or not torch.all(labels.le(96)):
        raise ValueError("target detector labels are outside 1..96")
    if not torch.all(public_ids.ge(0)) or not torch.all(public_ids.le(202)):
        raise ValueError("target public IDs are outside 0..202")
    return {"images": 1, "boxes": int(boxes.shape[0])}


def collate(batch):
    return tuple(zip(*batch))


def traverse(voc_root: Path, contract: Path, workers: int) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1, "workers": workers, "splits": {}}
    total_images = total_boxes = 0
    for split, expected in EXPECTED.items():
        dataset = IP102DetectionDataset(voc_root, contract / f"{split}.txt", contract / "class-map.json")
        if len(dataset) != expected:
            raise ValueError(f"unexpected {split} length: {len(dataset)}")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            stats = executor.map(dataset.__getitem__, range(len(dataset)))
            image_count = box_count = 0
            for sample in stats:
                sample_stats = validate_sample(sample)
                image_count += sample_stats["images"]
                box_count += sample_stats["boxes"]
        if image_count != expected:
            raise ValueError(f"incomplete {split} traversal: {image_count}")
        result["splits"][split] = {"images": image_count, "boxes": box_count, "seconds": round(time.monotonic() - started, 3)}
        total_images += image_count
        total_boxes += box_count

    smoke: dict[str, Any] = {}
    for split in ("train", "val"):
        dataset = IP102DetectionDataset(voc_root, contract / f"{split}.txt", contract / "class-map.json")
        images, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)))
        for sample in zip(images, targets):
            validate_sample(sample)
        smoke[split] = {"batch_size": len(images), "passed": True}
    result["dataloader_smoke"] = smoke
    result["total_images"] = total_images
    result["total_valid_boxes"] = total_boxes
    if total_images != 18_976 or total_boxes != 22_283:
        raise ValueError(f"full traversal totals mismatch: {total_images}, {total_boxes}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print(json.dumps(traverse(args.voc_root.resolve(), args.contract.resolve(), args.workers), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

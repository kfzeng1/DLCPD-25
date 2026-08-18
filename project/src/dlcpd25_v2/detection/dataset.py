"""IP102 detection dataset backed by the frozen Plan-A contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms.functional import hflip, to_tensor


class IP102DetectionDataset(Dataset):
    """Reads ``artifacts/data/ip102`` and ``data/raw/ip102/VOC2007``."""

    def __init__(
        self,
        voc_root: Path | str,
        contract_dir: Path | str,
        split: str,
        train: bool,
        horizontal_flip_p: float = 0.5,
    ) -> None:
        self.voc_root = Path(voc_root)
        self.contract_dir = Path(contract_dir)
        self.split = split
        self.train = train
        self.horizontal_flip_p = horizontal_flip_p if train else 0.0

        split_path = self.contract_dir / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"split file not found: {split_path}")
        ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate ids in split {split}: {split_path}")
        self.ids = ids

        annotations_path = self.contract_dir / "annotations.jsonl"
        self.records: dict[str, dict[str, Any]] = {}
        with annotations_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                image_id = str(record["image_id"])
                if image_id in set(ids):
                    self.records[image_id] = record
        missing = [image_id for image_id in ids if image_id not in self.records]
        if missing:
            raise ValueError(f"{split} ids missing from annotations: {missing[:5]}")

        label_counts: dict[int, int] = {}
        for record in self.records.values():
            for obj in record["objects"]:
                label = int(obj["detector_label"])
                label_counts[label] = label_counts.get(label, 0) + 1
        self.label_counts = label_counts

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.ids[index]
        record = self.records[image_id]
        image_path = self.voc_root / str(record["image_path"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        boxes = torch.tensor(
            [[float(v) for v in obj["bbox"]] for obj in record["objects"]],
            dtype=torch.float32,
        )
        labels = torch.tensor([int(obj["detector_label"]) for obj in record["objects"]], dtype=torch.int64)
        public_ids = torch.tensor(
            [int(obj["dlcpd25_class_id"]) for obj in record["objects"]], dtype=torch.int64
        )
        source_ids = torch.tensor(
            [int(obj["ip102_class_id"]) for obj in record["objects"]], dtype=torch.int64
        )
        if self.train and boxes.numel() and torch.rand(1).item() < self.horizontal_flip_p:
            image = hflip(image)
            width = float(image.width)
            boxes = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
            )
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
            "dlcpd25_class_ids": public_ids,
            "ip102_class_ids": source_ids,
            "width": int(record["width"]),
            "height": int(record["height"]),
        }
        return to_tensor(image), target

    def repeat_factor_sampler(self, cap: float = 8.0) -> WeightedRandomSampler:
        """Square-root repeat-factor sampling for long-tail detection labels."""
        if not self.label_counts:
            raise ValueError("no labels in split")
        max_count = max(self.label_counts.values())
        repeat_by_label = {
            label: min(cap, float(max_count / count) ** 0.5)
            for label, count in self.label_counts.items()
        }
        weights: list[float] = []
        for image_id in self.ids:
            labels = [int(obj["detector_label"]) for obj in self.records[image_id]["objects"]]
            weights.append(max((repeat_by_label[label] for label in labels), default=1.0))
        return WeightedRandomSampler(weights=weights, num_samples=len(self.ids), replacement=True)


def collate_detection(batch):
    return tuple(zip(*batch))

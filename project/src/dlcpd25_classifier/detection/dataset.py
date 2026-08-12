"""IP102 Pascal VOC dataset using the frozen DLCPD-25 public class IDs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .mapping import DetectionClassMapping

NAME_PATTERN = re.compile(r"<name>\s*(\d+)\s*</name>")


def _parse_annotation(path: Path) -> tuple[list[list[float]], list[int]]:
    """Parse official VOC XML, including IP087000986's duplicated root document."""
    text = path.read_text(encoding="utf-8")
    try:
        roots = [ElementTree.fromstring(text)]
    except ElementTree.ParseError:
        documents = re.findall(r"<annotation\b.*?</annotation>", text, flags=re.DOTALL)
        if not documents:
            raise
        roots = [ElementTree.fromstring(document) for document in documents]
    boxes: list[list[float]] = []
    labels: list[int] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for root in roots:
        for obj in root.findall("object"):
            label_text = obj.findtext("name")
            box = obj.find("bndbox")
            if label_text is None or box is None:
                raise ValueError(f"incomplete object annotation: {path}")
            coordinates = [float(box.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")]
            if not all(torch.isfinite(torch.tensor(coordinates))) or coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
                raise ValueError(f"invalid bounding box: {path}")
            key = (int(label_text), *coordinates)
            if key in seen:
                continue
            seen.add(key)
            labels.append(int(label_text))
            boxes.append(coordinates)
    if not boxes:
        raise ValueError(f"annotation has no objects: {path}")
    return boxes, labels


class IP102DetectionDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Return detector labels for training plus public DLCPD-25 IDs for auditing."""

    def __init__(
        self,
        voc_root: str | Path,
        split_file: str | Path,
        mapping_path: str | Path,
    ) -> None:
        self.root = Path(voc_root)
        self.mapping = DetectionClassMapping(mapping_path)
        self.image_ids = tuple(
            line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("split contains duplicate image IDs")
        for image_id in self.image_ids:
            if not (self.root / "JPEGImages" / f"{image_id}.jpg").is_file():
                raise FileNotFoundError(f"missing IP102 image: {image_id}")
            if not (self.root / "Annotations" / f"{image_id}.xml").is_file():
                raise FileNotFoundError(f"missing IP102 annotation: {image_id}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = self.image_ids[index]
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = self.root / "Annotations" / f"{image_id}.xml"
        boxes, ip102_ids = _parse_annotation(annotation_path)
        records = [self.mapping.from_ip102(class_id) for class_id in ip102_ids]
        with Image.open(image_path) as source:
            image = F.pil_to_tensor(source.convert("RGB")).float().div(255.0)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor([record.detector_label for record in records], dtype=torch.int64),
            "dlcpd25_class_ids": torch.tensor([record.dlcpd25_class_id for record in records], dtype=torch.int64),
            "ip102_class_ids": torch.tensor(ip102_ids, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image, target

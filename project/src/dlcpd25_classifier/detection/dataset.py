"""IP102 detection Dataset backed by the frozen T0 data contract."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .mapping import DetectionClassMapping

_ANNOTATION_PATTERN = re.compile(r"<annotation\b.*?</annotation>", re.DOTALL)


def _parse_annotation(path: Path) -> tuple[list[list[float]], list[int]]:
    """Parse a raw VOC XML for compatibility, applying the frozen T0 rules."""
    text = path.read_text(encoding="utf-8")
    try:
        roots = [ElementTree.fromstring(text)]
    except ElementTree.ParseError:
        documents = _ANNOTATION_PATTERN.findall(text)
        if not documents:
            raise
        roots = [ElementTree.fromstring(document) for document in documents]

    # IP087000986 contains the same complete annotation document twice.
    unique_roots: list[ElementTree.Element] = []
    signatures: set[bytes] = set()
    for root in roots:
        signature = ElementTree.tostring(root, encoding="utf-8")
        if signature not in signatures:
            signatures.add(signature)
            unique_roots.append(root)

    boxes: list[list[float]] = []
    labels: list[int] = []
    for root in unique_roots:
        size = root.find("size")
        if size is None:
            raise ValueError(f"annotation has no size: {path}")
        width = float(size.findtext("width", "nan"))
        height = float(size.findtext("height", "nan"))
        for obj in root.findall("object"):
            label_text = obj.findtext("name")
            box = obj.find("bndbox")
            if label_text is None or box is None:
                raise ValueError(f"incomplete object annotation: {path}")
            coordinates = [float(box.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")]
            finite = all(math.isfinite(value) for value in coordinates)
            ordered = coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]
            bounded = 0 <= coordinates[0] and coordinates[2] <= width and 0 <= coordinates[1] and coordinates[3] <= height
            if not (finite and ordered and bounded):
                # T0 filters only the invalid box and retains valid peers.
                continue
            labels.append(int(label_text))
            boxes.append(coordinates)
    if not boxes:
        raise ValueError(f"annotation has no valid objects: {path}")
    return boxes, labels


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value}")
    return path


class IP102DetectionDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Load IP102 targets in detector, source, and public DLCPD-25 ID spaces.

    A split beside ``annotations.jsonl`` automatically uses the frozen T0
    annotations. ``annotations_path`` can be given explicitly. Raw VOC parsing
    remains available only as a compatibility path for pre-T0 callers.
    """

    def __init__(
        self,
        voc_root: str | Path,
        split_file: str | Path,
        mapping_path: str | Path,
        annotations_path: str | Path | None = None,
        transforms: Callable[[Tensor, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]] | None = None,
    ) -> None:
        self.root = Path(voc_root)
        self.mapping = DetectionClassMapping(mapping_path)
        self.transforms = transforms
        split_path = Path(split_file)
        self.image_ids = tuple(line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not self.image_ids or len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("split must be nonempty and contain unique image IDs")

        inferred = split_path.parent / "annotations.jsonl"
        contract_path = Path(annotations_path) if annotations_path is not None else (inferred if inferred.is_file() else None)
        self._records: dict[str, dict[str, Any]] | None = None
        if contract_path is not None:
            requested = set(self.image_ids)
            records: dict[str, dict[str, Any]] = {}
            with contract_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    image_id = record.get("image_id")
                    if image_id not in requested:
                        continue
                    if image_id in records:
                        raise ValueError(f"duplicate annotation record for {image_id} at line {line_number}")
                    records[image_id] = record
            missing = sorted(requested - records.keys())
            if missing:
                raise ValueError(f"split IDs missing from frozen annotations: {missing[:3]}")
            self._records = records

        for image_id in self.image_ids:
            if self._records is None:
                image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
                annotation_path = self.root / "Annotations" / f"{image_id}.xml"
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing IP102 image: {image_id}")
                if not annotation_path.is_file():
                    raise FileNotFoundError(f"missing IP102 annotation: {image_id}")
            else:
                record = self._records[image_id]
                relative = _safe_relative_path(str(record["image_path"]), "image_path")
                if not (self.root / relative).is_file():
                    raise FileNotFoundError(f"missing IP102 image: {image_id}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = self.image_ids[index]
        if self._records is None:
            image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
            boxes, ip102_ids = _parse_annotation(self.root / "Annotations" / f"{image_id}.xml")
            stable_index = index
        else:
            record = self._records[image_id]
            image_path = self.root / _safe_relative_path(str(record["image_path"]), "image_path")
            boxes = [obj["bbox"] for obj in record["objects"]]
            ip102_ids = [int(obj["ip102_class_id"]) for obj in record["objects"]]
            stable_index = int(record["formal_index"])

        records = [self.mapping.from_ip102(class_id) for class_id in ip102_ids]
        with Image.open(image_path) as source:
            image = F.pil_to_tensor(source.convert("RGB")).float().div(255.0)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": box_tensor,
            "labels": torch.tensor([record.detector_label for record in records], dtype=torch.int64),
            "dlcpd25_class_ids": torch.tensor([record.dlcpd25_class_id for record in records], dtype=torch.int64),
            "ip102_class_ids": torch.tensor(ip102_ids, dtype=torch.int64),
            "image_id": torch.tensor(stable_index, dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

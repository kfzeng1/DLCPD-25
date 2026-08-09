"""Read the frozen 203-class taxonomy and resolve hierarchical predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassRecord:
    class_id: int
    official_name: str
    local_directory: str
    host_group: str
    host_id: str
    host_zh: str
    category: str
    category_zh: str
    image_count: int


class Taxonomy:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("classes")
        if not isinstance(records, list) or len(records) != 203:
            raise ValueError("taxonomy must contain 203 classes")
        self._classes = tuple(ClassRecord(**record) for record in records)
        if [record.class_id for record in self._classes] != list(range(203)):
            raise ValueError("class IDs must be contiguous from zero")

    @property
    def classes(self) -> tuple[ClassRecord, ...]:
        return self._classes

    def resolve(self, class_id: int) -> ClassRecord:
        if not 0 <= class_id < len(self._classes):
            raise IndexError(f"unknown class ID: {class_id}")
        return self._classes[class_id]

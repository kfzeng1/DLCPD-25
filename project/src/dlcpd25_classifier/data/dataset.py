"""Validated Dataset loader for frozen DLCPD-25 split CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


CSV_FIELDS = ("relative_path", "class_id", "sha256", "duplicate_group_id", "split")
EXIF_TRANSPOSE_METHODS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


@dataclass(frozen=True)
class SplitRecord:
    relative_path: str
    class_id: int
    sha256: str
    duplicate_group_id: str
    split: str


def _apply_valid_exif_orientation(image: Image.Image) -> Image.Image:
    orientation = image.getexif().get(274)
    if type(orientation) is int and orientation in EXIF_TRANSPOSE_METHODS:
        return image.transpose(EXIF_TRANSPOSE_METHODS[orientation])
    return image


class DLCPD25Dataset(Dataset[tuple[Any, int]]):
    """Load one frozen split without rescanning directories or inferring labels."""

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        taxonomy_path: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
        verify_files: bool = True,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split_csv = Path(split_csv).resolve()
        self.taxonomy_path = Path(taxonomy_path).resolve()
        self.transform = transform
        if not self.data_root.is_dir() or self.data_root.is_symlink():
            raise ValueError(f"invalid data root: {self.data_root}")
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        classes = taxonomy.get("classes")
        if not isinstance(classes, list) or [item.get("class_id") for item in classes] != list(range(203)):
            raise ValueError("taxonomy must contain fixed class IDs 0-202")
        self._classes = {int(item["class_id"]): item for item in classes}
        expected_split = self.split_csv.stem
        records: list[SplitRecord] = []
        seen: set[str] = set()
        with self.split_csv.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(f"unexpected split CSV schema: {reader.fieldnames}")
            for row in reader:
                relative_path = str(row["relative_path"])
                pure_path = PurePosixPath(relative_path)
                if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
                    raise ValueError(f"unsafe relative path: {relative_path}")
                if relative_path in seen:
                    raise ValueError(f"duplicate relative path: {relative_path}")
                seen.add(relative_path)
                class_id = int(row["class_id"])
                class_record = self._classes.get(class_id)
                if class_record is None:
                    raise ValueError(f"unknown class_id: {class_id}")
                if pure_path.parts[0] != class_record["local_directory"]:
                    raise ValueError(f"path/taxonomy mismatch for class_id={class_id}: {relative_path}")
                if row["split"] != expected_split:
                    raise ValueError(f"split field differs from filename: {row['split']} != {expected_split}")
                if len(row["sha256"]) != 64 or not row["duplicate_group_id"]:
                    raise ValueError(f"invalid hash or duplicate group: {relative_path}")
                record = SplitRecord(
                    relative_path=relative_path,
                    class_id=class_id,
                    sha256=row["sha256"],
                    duplicate_group_id=row["duplicate_group_id"],
                    split=row["split"],
                )
                if verify_files:
                    source = self._source_path(record)
                    if not source.is_file() or source.is_symlink():
                        raise ValueError(f"split source is missing or symlinked: {relative_path}")
                records.append(record)
        if not records:
            raise ValueError(f"split CSV is empty: {self.split_csv}")
        self.records = tuple(records)

    def _source_path(self, record: SplitRecord) -> Path:
        source = (self.data_root / record.relative_path).resolve()
        try:
            source.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"split path leaves data root: {record.relative_path}") from exc
        return source

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        source = self._source_path(record)
        with Image.open(source) as image:
            image.load()
            rgb = _apply_valid_exif_orientation(image).convert("RGB")
        value = self.transform(rgb) if self.transform is not None else rgb
        return value, record.class_id

    def get_record(self, index: int) -> SplitRecord:
        return self.records[index]

"""Map IP102 detection labels onto the frozen DLCPD-25 class IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionClass:
    ip102_class_id: int
    detector_label: int
    dlcpd25_class_id: int
    ip102_name: str
    dlcpd25_name: str
    match_method: str


class DetectionClassMapping:
    """Translate public DLCPD-25 IDs to contiguous detector labels and back."""

    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload.get("classes")
        if not isinstance(records, list) or len(records) != 97:
            raise ValueError("IP102 detection mapping must contain 97 classes")
        self._classes = tuple(DetectionClass(**record) for record in records)
        detector_labels = [record.detector_label for record in self._classes]
        if sorted(set(detector_labels)) != list(range(1, 97)):
            raise ValueError("detector labels must cover 1..96; 0 is background")
        ip102_ids = [record.ip102_class_id for record in self._classes]
        dlcpd25_ids = [record.dlcpd25_class_id for record in self._classes]
        if len(set(ip102_ids)) != 97 or len(set(dlcpd25_ids)) != 96:
            raise ValueError("expected 97 IP102 labels mapped onto 96 DLCPD-25 classes")
        if not all(0 <= value <= 101 for value in ip102_ids):
            raise ValueError("IP102 class IDs must be within 0..101")
        if not all(0 <= value <= 202 for value in dlcpd25_ids):
            raise ValueError("DLCPD-25 class IDs must be within 0..202")
        self._by_ip102 = {record.ip102_class_id: record for record in self._classes}
        self._by_detector: dict[int, DetectionClass] = {}
        for record in self._classes:
            existing = self._by_detector.setdefault(record.detector_label, record)
            if existing.dlcpd25_class_id != record.dlcpd25_class_id:
                raise ValueError("one detector label maps to multiple DLCPD-25 classes")
        self._by_dlcpd25: dict[int, tuple[DetectionClass, ...]] = {}
        for class_id in sorted(set(dlcpd25_ids)):
            self._by_dlcpd25[class_id] = tuple(
                record for record in self._classes if record.dlcpd25_class_id == class_id
            )

    @property
    def classes(self) -> tuple[DetectionClass, ...]:
        return self._classes

    @property
    def num_detector_classes(self) -> int:
        """Return foreground classes; model num_classes is this value plus background."""
        return len(self._by_detector)

    def from_ip102(self, class_id: int) -> DetectionClass:
        try:
            return self._by_ip102[class_id]
        except KeyError as exc:
            raise KeyError(f"IP102 class {class_id} has no detection mapping") from exc

    def from_detector(self, label: int) -> DetectionClass:
        if label == 0:
            raise KeyError("detector label 0 is background and has no DLCPD-25 class ID")
        try:
            return self._by_detector[label]
        except KeyError as exc:
            raise KeyError(f"unknown detector label: {label}") from exc

    def from_dlcpd25(self, class_id: int) -> tuple[DetectionClass, ...]:
        try:
            return self._by_dlcpd25[class_id]
        except KeyError as exc:
            raise KeyError(f"DLCPD-25 class {class_id} is not detectable by IP102") from exc

    def export_predictions(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Replace detector-private labels with the system-wide DLCPD-25 IDs."""
        labels = prediction.get("labels")
        if labels is None:
            raise KeyError("prediction is missing labels")
        mapped = [self.from_detector(int(label)).dlcpd25_class_id for label in labels]
        return {**prediction, "labels": mapped}

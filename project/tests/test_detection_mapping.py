from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dlcpd25_classifier.detection import DetectionClassMapping

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"


def test_detection_mapping_uses_dlcpd25_public_ids() -> None:
    mapping = DetectionClassMapping(MAPPING)
    assert mapping.num_detector_classes == 96
    assert sorted({record.detector_label for record in mapping.classes}) == list(range(1, 97))
    assert len({record.dlcpd25_class_id for record in mapping.classes}) == 96
    assert mapping.from_ip102(8).dlcpd25_class_id == 199
    assert mapping.from_ip102(23).dlcpd25_class_id == 88
    assert mapping.from_ip102(24).dlcpd25_class_id == 83
    assert mapping.from_ip102(83).dlcpd25_class_id == 14
    assert [record.ip102_class_id for record in mapping.from_dlcpd25(97)] == [50, 51]
    assert mapping.from_ip102(50).detector_label == mapping.from_ip102(51).detector_label


def test_background_and_unannotated_classes_are_rejected() -> None:
    mapping = DetectionClassMapping(MAPPING)
    with pytest.raises(KeyError, match="background"):
        mapping.from_detector(0)
    with pytest.raises(KeyError, match="no detection mapping"):
        mapping.from_ip102(59)
    with pytest.raises(KeyError, match="not detectable"):
        mapping.from_dlcpd25(103)


def test_prediction_export_replaces_private_detector_labels() -> None:
    mapping = DetectionClassMapping(MAPPING)
    exported = mapping.export_predictions(
        {"labels": torch.tensor([mapping.from_ip102(8).detector_label]), "scores": [0.9]}
    )
    assert exported["labels"] == [199]
    assert exported["scores"] == [0.9]


def test_mapping_metadata_is_explicit() -> None:
    payload = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert payload["public_class_id_space"] == "DLCPD-25 class_id 0..202"
    assert payload["detector_label_space"].startswith("1..96")
    assert payload["ip102_source_labels"] == 97
    assert payload["detector_foreground_classes"] == 96
    assert payload["mapped_dlcpd25_detection_classes"] == 96
    assert payload["match_summary"] == {
        "normalized_name": 89,
        "reviewed_alias_or_host": 8,
    }

from __future__ import annotations

from pathlib import Path

import torch
from dlcpd25_classifier.detection import IP102DetectionDataset

ROOT = Path(__file__).resolve().parents[2]
VOC_ROOT = ROOT / "data/raw/ip102/downloads/Detection/VOC2007"
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"


def test_official_detection_sample_returns_both_label_spaces(tmp_path: Path) -> None:
    split = tmp_path / "sample.txt"
    split.write_text("IP000000000\n", encoding="utf-8")
    dataset = IP102DetectionDataset(VOC_ROOT, split, MAPPING)
    image, target = dataset[0]
    assert image.dtype == torch.float32
    assert image.shape == (3, 420, 650)
    assert target["boxes"].shape == (1, 4)
    assert target["ip102_class_ids"].tolist() == [0]
    assert target["dlcpd25_class_ids"].tolist() == [153]
    assert target["labels"].tolist() == [dataset.mapping.from_ip102(0).detector_label]


def test_duplicated_official_xml_is_parsed_once(tmp_path: Path) -> None:
    split = tmp_path / "sample.txt"
    split.write_text("IP087000986\n", encoding="utf-8")
    dataset = IP102DetectionDataset(VOC_ROOT, split, MAPPING)
    _, target = dataset[0]
    assert target["ip102_class_ids"].tolist() == [86]
    assert target["boxes"].shape == (1, 4)


def test_degenerate_box_is_filtered_but_valid_peer_is_retained(tmp_path: Path) -> None:
    split = tmp_path / "sample.txt"
    split.write_text("IP046000898\n", encoding="utf-8")
    dataset = IP102DetectionDataset(VOC_ROOT, split, MAPPING)
    _, target = dataset[0]
    assert target["ip102_class_ids"].tolist() == [45]
    assert target["boxes"].tolist() == [[9.0, 34.0, 185.0, 131.0]]

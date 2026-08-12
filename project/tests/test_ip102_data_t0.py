from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from dlcpd25_classifier.detection import IP102DetectionDataset
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
VOC_ROOT = ROOT / "data/raw/ip102/downloads/Detection/VOC2007"
OUTPUT = ROOT / "artifacts/data/ip102-detection-v1"
SCRIPT = ROOT / "scripts/ip102/build_detection_data_t0.py"

SPEC = importlib.util.spec_from_file_location("build_detection_data_t0", SCRIPT)
assert SPEC and SPEC.loader
T0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T0)


def test_multilabel_split_is_deterministic_exact_and_preserves_eligible_labels() -> None:
    image_ids = [f"item-{index}" for index in range(12)]
    labels = {image_id: {0} for image_id in image_ids}
    labels["item-0"] = {0, 1}
    labels["item-1"] = {0, 1}
    labels["item-2"] = {0, 2}
    labels["item-3"] = {0, 2}
    first = T0.iterative_multilabel_split(image_ids, labels, 0.25, 17)
    second = T0.iterative_multilabel_split(image_ids, labels, 0.25, 17)
    assert first == second
    train, val, strategy = first
    assert len(train) == 9 and len(val) == 3
    assert set(train).isdisjoint(val) and set(train) | set(val) == set(image_ids)
    for label in (0, 1, 2):
        assert any(label in labels[item] for item in train)
        assert any(label in labels[item] for item in val)
    assert strategy["algorithm"] == "deterministic_iterative_multilabel_stratification_v1"


def test_parser_deduplicates_root_and_filters_only_bad_box() -> None:
    mapping_payload = json.loads((ROOT / "metadata/ip102-detection-class-map.json").read_text(encoding="utf-8"))
    mapping = {int(item["ip102_class_id"]): item for item in mapping_payload["classes"]}
    duplicate = T0.parse_annotation(VOC_ROOT / "Annotations/IP087000986.xml", "IP087000986", mapping)
    assert duplicate["xml_document_count"] == 2
    assert duplicate["duplicate_xml_documents_ignored"] == 1
    assert len(duplicate["raw_objects"]) == len(duplicate["objects"]) == 1

    degenerate = T0.parse_annotation(VOC_ROOT / "Annotations/IP046000898.xml", "IP046000898", mapping)
    assert len(degenerate["raw_objects"]) == 2
    assert len(degenerate["objects"]) == 1
    assert degenerate["raw_objects"][0]["filter_reasons"] == ["non_positive_width"]
    assert degenerate["objects"][0]["bbox"] == [9.0, 34.0, 185.0, 131.0]


def test_frozen_t0_contract_and_checksum_verifier() -> None:
    assert T0.verify_checksums(OUTPUT / "checksums.sha256") == 20
    assert len(T0.read_ids(OUTPUT / "train.txt")) == 12142
    assert len(T0.read_ids(OUTPUT / "val.txt")) == 3036
    assert len(T0.read_ids(OUTPUT / "test.txt")) == 3798
    assert sum(1 for _ in (OUTPUT / "annotations.jsonl").open(encoding="utf-8")) == 18976

    audit = json.loads((OUTPUT / "audit-summary.json").read_text(encoding="utf-8"))
    assert audit["boxes"] == {
        "raw_after_duplicate_document_deduplication": 22284,
        "filtered": 1,
        "valid": 22283,
    }

    split_summary = json.loads((OUTPUT / "split-summary.json").read_text(encoding="utf-8"))
    assert split_summary["source_label_coverage"] == {"train": 97, "val": 97, "test": 96}
    assert split_summary["test_missing_ip102_source_labels"] == [61]
    assert split_summary["invariants"]["test_exact_official_order_and_membership"] is True


def test_frozen_dataset_loads_all_splits_and_retains_public_ids() -> None:
    mapping_path = OUTPUT / "class-map.json"
    lengths = {"train": 12142, "val": 3036, "test": 3798}
    for split, expected in lengths.items():
        dataset = IP102DetectionDataset(VOC_ROOT, OUTPUT / f"{split}.txt", mapping_path)
        assert len(dataset) == expected
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            image, target = dataset[index]
            assert image.dtype == torch.float32 and image.ndim == 3 and image.shape[0] == 3
            assert torch.isfinite(image).all()
            assert target["boxes"].shape[0] == target["labels"].shape[0] > 0
            assert torch.isfinite(target["boxes"]).all()
            assert torch.all(target["labels"].ge(1)) and torch.all(target["labels"].le(96))
            assert torch.all(target["dlcpd25_class_ids"].ge(0)) and torch.all(target["dlcpd25_class_ids"].le(202))


def _collate(batch):
    return tuple(zip(*batch))


def test_train_and_val_dataloader_smoke() -> None:
    for split in ("train", "val"):
        dataset = IP102DetectionDataset(VOC_ROOT, OUTPUT / f"{split}.txt", OUTPUT / "class-map.json")
        images, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate)))
        assert len(images) == len(targets) == 2
        assert all(image.dtype == torch.float32 for image in images)
        assert all(target["boxes"].shape[1] == 4 for target in targets)

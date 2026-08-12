#!/usr/bin/env python3
"""Build and independently verify the frozen IP102 detection T0 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOC_ROOT = REPO_ROOT / "data/raw/ip102/downloads/Detection/VOC2007"
DEFAULT_MAPPING = REPO_ROOT / "metadata/ip102-detection-class-map.json"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata/class-taxonomy.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/data/ip102-detection-v1"
TEST_SOURCE = REPO_ROOT / "project/tests/test_ip102_data_t0.py"
DATASET_REGRESSION_TEST_SOURCE = REPO_ROOT / "project/tests/test_detection_dataset.py"
DATASET_SOURCE = REPO_ROOT / "project/src/dlcpd25_classifier/detection/dataset.py"
MAPPING_SOURCE = REPO_ROOT / "project/src/dlcpd25_classifier/detection/mapping.py"
DATASET_VERIFIER_SOURCE = REPO_ROOT / "scripts/ip102/verify_detection_dataset_t0.py"

SCHEMA_VERSION = 1
SEED = 20260812
VAL_RATIO = 0.20
EXPECTED_IMAGES = 18_981
EXPECTED_FORMAL = 18_976
EXPECTED_TRAINVAL = 15_178
EXPECTED_TEST = 3_798
EXPECTED_RAW_BOXES = 22_284
EXPECTED_VALID_BOXES = 22_283
EXPECTED_SOURCE_LABELS = 97
EXPECTED_DETECTOR_LABELS = 96
EXPECTED_EXTRA_IMAGES = 5
KNOWN_DUPLICATED_ROOT = "IP087000986"
KNOWN_DEGENERATE_IMAGE = "IP046000898"
KNOWN_TEST_MISSING_SOURCE_LABEL = 61
ANNOTATION_PATTERN = re.compile(r"<annotation\b.*?</annotation>", re.DOTALL)
OUTPUT_NAMES = (
    "train.txt",
    "val.txt",
    "test.txt",
    "annotations.jsonl",
    "class-map.json",
    "audit-summary.json",
    "split-summary.json",
    "exceptions.json",
    "build-config.json",
    "data-handoff.md",
)


class T0Error(RuntimeError):
    """Raised when the T0 contract cannot be built or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise T0Error(f"refusing to overwrite existing artifact with different bytes: {path}")
    path.write_bytes(content)


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise T0Error(f"duplicate image IDs in {path}")
    return values


def parse_xml_documents(text: str, path: Path) -> tuple[list[ElementTree.Element], int]:
    try:
        roots = [ElementTree.fromstring(text)]
    except ElementTree.ParseError:
        documents = ANNOTATION_PATTERN.findall(text)
        if not documents:
            raise T0Error(f"unparseable XML: {path}")
        roots = [ElementTree.fromstring(document) for document in documents]
    signatures: set[bytes] = set()
    unique: list[ElementTree.Element] = []
    for root in roots:
        signature = ElementTree.tostring(root, encoding="utf-8")
        if signature not in signatures:
            signatures.add(signature)
            unique.append(root)
    return unique, len(roots) - len(unique)


def parse_annotation(path: Path, image_id: str, mapping: dict[int, dict[str, Any]]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    roots, duplicate_documents = parse_xml_documents(text, path)
    sizes: set[tuple[int, int, int]] = set()
    raw_objects: list[dict[str, Any]] = []
    valid_objects: list[dict[str, Any]] = []
    source_index = 0
    for root in roots:
        size = root.find("size")
        if size is None:
            raise T0Error(f"missing annotation size: {path}")
        width = int(size.findtext("width", "-1"))
        height = int(size.findtext("height", "-1"))
        depth = int(size.findtext("depth", "-1"))
        if width <= 0 or height <= 0 or depth <= 0:
            raise T0Error(f"invalid annotation size: {path}")
        sizes.add((width, height, depth))
        for obj in root.findall("object"):
            label_text = obj.findtext("name")
            box = obj.find("bndbox")
            if label_text is None or box is None:
                raise T0Error(f"incomplete object: {path}")
            source_id = int(label_text)
            if source_id not in mapping:
                raise T0Error(f"unmapped source class {source_id}: {path}")
            coordinates = [float(box.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")]
            reasons: list[str] = []
            if not all(math.isfinite(value) for value in coordinates):
                reasons.append("non_finite")
            else:
                if coordinates[0] >= coordinates[2]:
                    reasons.append("non_positive_width")
                if coordinates[1] >= coordinates[3]:
                    reasons.append("non_positive_height")
                if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[2] > width or coordinates[3] > height:
                    reasons.append("out_of_bounds")
            mapped = mapping[source_id]
            raw = {
                "source_object_index": source_index,
                "bbox": coordinates,
                "ip102_class_id": source_id,
                "detector_label": int(mapped["detector_label"]),
                "dlcpd25_class_id": int(mapped["dlcpd25_class_id"]),
                "valid": not reasons,
                "filter_reasons": reasons,
            }
            raw_objects.append(raw)
            if not reasons:
                valid_objects.append({key: value for key, value in raw.items() if key not in {"valid", "filter_reasons"}})
            source_index += 1
    if len(sizes) != 1:
        raise T0Error(f"inconsistent duplicated annotation sizes: {path}")
    if not valid_objects:
        raise T0Error(f"annotation has no valid boxes: {path}")
    width, height, depth = sizes.pop()
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "depth": depth,
        "xml_document_count": len(roots) + duplicate_documents,
        "duplicate_xml_documents_ignored": duplicate_documents,
        "raw_objects": raw_objects,
        "objects": valid_objects,
    }


def iterative_multilabel_split(
    image_ids: Sequence[str], labels_by_id: dict[str, set[int]], val_ratio: float, seed: int
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Deterministic, exact-size iterative stratification on image-level labels."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    target_size = round(len(image_ids) * val_ratio)
    support = Counter(label for image_id in image_ids for label in labels_by_id[image_id])
    desired = {label: min(max(round(count * val_ratio), 1 if count >= 2 else 0), count - 1) for label, count in support.items()}
    rng = random.Random(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    tie_rank = {image_id: rank for rank, image_id in enumerate(shuffled)}
    remaining = set(image_ids)
    selected: list[str] = []
    selected_support: Counter[int] = Counter()

    def eligible(image_id: str) -> bool:
        return all(
            support[label] < 2 or selected_support[label] + 1 < support[label]
            for label in labels_by_id[image_id]
        )

    while len(selected) < target_size:
        deficits = {label: desired[label] - selected_support[label] for label in support}
        needed = [label for label, deficit in deficits.items() if deficit > 0 and any(label in labels_by_id[i] for i in remaining)]
        if not needed:
            break
        label = min(needed, key=lambda item: (sum(item in labels_by_id[i] for i in remaining), support[item], item))
        candidates = [image_id for image_id in remaining if label in labels_by_id[image_id] and eligible(image_id)]
        if not candidates:
            raise T0Error(f"no eligible validation candidate remains for source label {label}")
        chosen = max(
            candidates,
            key=lambda image_id: (
                sum(max(deficits.get(item, 0), 0) / support[item] for item in labels_by_id[image_id]),
                len(labels_by_id[image_id]),
                -tie_rank[image_id],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        selected_support.update(labels_by_id[chosen])

    if len(selected) < target_size:
        fill = sorted(
            (image_id for image_id in remaining if eligible(image_id)),
            key=lambda image_id: (
                sum(abs((selected_support[label] + 1) - support[label] * val_ratio) for label in labels_by_id[image_id]),
                tie_rank[image_id],
            ),
        )
        needed_count = target_size - len(selected)
        if len(fill) < needed_count:
            raise T0Error("cannot reach exact validation size without removing a rare label from train")
        selected.extend(fill[:needed_count])
    val_set = set(selected)
    train = [image_id for image_id in image_ids if image_id not in val_set]
    val = [image_id for image_id in image_ids if image_id in val_set]
    train_support = Counter(label for image_id in train for label in labels_by_id[image_id])
    val_support = Counter(label for image_id in val for label in labels_by_id[image_id])
    missing_rare = [label for label, count in support.items() if count >= 2 and (train_support[label] == 0 or val_support[label] == 0)]
    if missing_rare:
        raise T0Error(f"stratification failed to preserve eligible labels: {missing_rare}")
    return train, val, {
        "algorithm": "deterministic_iterative_multilabel_stratification_v1",
        "seed": seed,
        "val_ratio": val_ratio,
        "target_val_images": target_size,
        "rare_class_policy": "For every trainval source label with at least two images, require at least one image in train and val.",
    }


def per_label_stats(ids: Iterable[str], records: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    image_counts: Counter[int] = Counter()
    box_counts: Counter[int] = Counter()
    for image_id in ids:
        labels = {int(obj["ip102_class_id"]) for obj in records[image_id]["objects"]}
        image_counts.update(labels)
        box_counts.update(int(obj["ip102_class_id"]) for obj in records[image_id]["objects"])
    return {str(label): {"images": image_counts[label], "boxes": box_counts[label]} for label in sorted(image_counts)}


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def audit_raw(voc_root: Path, mapping_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], list[str], list[str]]:
    trainval = read_ids(voc_root / "ImageSets/Main/trainval.txt")
    test = read_ids(voc_root / "ImageSets/Main/test.txt")
    if len(trainval) != EXPECTED_TRAINVAL or len(test) != EXPECTED_TEST or set(trainval) & set(test):
        raise T0Error("official split count or disjointness mismatch")
    formal = trainval + test
    if len(formal) != EXPECTED_FORMAL or len(formal) != len(set(formal)):
        raise T0Error("official formal ID count mismatch")
    image_paths = sorted((voc_root / "JPEGImages").glob("*.jpg"))
    xml_paths = sorted((voc_root / "Annotations").glob("*.xml"))
    if len(image_paths) != EXPECTED_IMAGES or len(xml_paths) != EXPECTED_FORMAL:
        raise T0Error("raw image/XML file count mismatch")
    extra_images = sorted(path.name for path in image_paths if path.stem not in set(formal))
    if len(extra_images) != EXPECTED_EXTRA_IMAGES:
        raise T0Error("extra JPEG count mismatch")
    missing_images = sorted(image_id for image_id in formal if not (voc_root / "JPEGImages" / f"{image_id}.jpg").is_file())
    missing_xml = sorted(image_id for image_id in formal if not (voc_root / "Annotations" / f"{image_id}.xml").is_file())
    if missing_images or missing_xml:
        raise T0Error("formal image/XML pairs are incomplete")

    mapping = {int(record["ip102_class_id"]): record for record in mapping_payload["classes"]}
    records: dict[str, dict[str, Any]] = {}
    image_formats: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    duplicate_roots: list[dict[str, Any]] = []
    filtered_boxes: list[dict[str, Any]] = []
    raw_boxes = valid_boxes = 0
    for formal_index, image_id in enumerate(formal):
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        record = parse_annotation(xml_path, image_id, mapping)
        with Image.open(image_path) as image:
            image.load()
            actual_size = image.size
            image_format = str(image.format)
            image_mode = image.mode
        if actual_size != (record["width"], record["height"]):
            raise T0Error(f"image/XML size mismatch: {image_id}")
        image_formats[image_format] += 1
        image_modes[image_mode] += 1
        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "formal_index": formal_index,
                "official_partition": "trainval" if formal_index < len(trainval) else "test",
                "image_path": f"JPEGImages/{image_id}.jpg",
                "annotation_path": f"Annotations/{image_id}.xml",
                "image_format": image_format,
                "image_mode": image_mode,
                "image_sha256": sha256_file(image_path),
                "annotation_sha256": sha256_file(xml_path),
            }
        )
        raw_boxes += len(record["raw_objects"])
        valid_boxes += len(record["objects"])
        if record["duplicate_xml_documents_ignored"]:
            duplicate_roots.append({"image_id": image_id, "ignored_documents": record["duplicate_xml_documents_ignored"], "action": "deduplicate_identical_document_content"})
        for obj in record["raw_objects"]:
            if not obj["valid"]:
                filtered_boxes.append({"image_id": image_id, **obj, "action": "filter_box_only_keep_image"})
        records[image_id] = record

    source_labels = sorted({int(obj["ip102_class_id"]) for record in records.values() for obj in record["objects"]})
    detector_labels = sorted({int(obj["detector_label"]) for record in records.values() for obj in record["objects"]})
    public_ids = sorted({int(obj["dlcpd25_class_id"]) for record in records.values() for obj in record["objects"]})
    if raw_boxes != EXPECTED_RAW_BOXES or valid_boxes != EXPECTED_VALID_BOXES:
        raise T0Error(f"box count mismatch: raw={raw_boxes}, valid={valid_boxes}")
    if len(source_labels) != EXPECTED_SOURCE_LABELS or detector_labels != list(range(1, EXPECTED_DETECTOR_LABELS + 1)):
        raise T0Error("class mapping coverage mismatch")
    if duplicate_roots != [{"image_id": KNOWN_DUPLICATED_ROOT, "ignored_documents": 1, "action": "deduplicate_identical_document_content"}]:
        raise T0Error(f"unexpected duplicated XML documents: {duplicate_roots}")
    if len(filtered_boxes) != 1 or filtered_boxes[0]["image_id"] != KNOWN_DEGENERATE_IMAGE:
        raise T0Error(f"unexpected invalid boxes: {filtered_boxes}")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "raw": {"jpeg_files": len(image_paths), "xml_files": len(xml_paths), "official_ids": len(formal), "extra_jpeg_files": len(extra_images)},
        "official": {"trainval_images": len(trainval), "test_images": len(test), "overlap": 0},
        "boxes": {"raw_after_duplicate_document_deduplication": raw_boxes, "filtered": raw_boxes - valid_boxes, "valid": valid_boxes},
        "labels": {"ip102_source": len(source_labels), "detector_foreground": len(detector_labels), "dlcpd25_public_mapped": len(public_ids), "detector_label_min": min(detector_labels), "detector_label_max": max(detector_labels), "public_id_min": min(public_ids), "public_id_max": max(public_ids)},
        "images": {"formats": dict(sorted(image_formats.items())), "modes": dict(sorted(image_modes.items())), "decode_failures": 0, "xml_image_dimension_mismatches": 0},
        "per_source_label_all": per_label_stats(formal, records),
    }
    exceptions = {
        "schema_version": SCHEMA_VERSION,
        "duplicated_xml_documents": duplicate_roots,
        "filtered_boxes": filtered_boxes,
        "extra_jpeg_files": extra_images,
        "missing_test_source_labels": [KNOWN_TEST_MISSING_SOURCE_LABEL],
        "policy": {"raw_files_modified": False, "extra_jpegs_deleted": False, "test_support_fabricated": False},
    }
    return records, audit, exceptions, trainval, test


def build(voc_root: Path, mapping_path: Path, taxonomy_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise T0Error(f"output directory already contains files; use --verify-only: {output}")
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    if len(mapping_payload.get("classes", [])) != EXPECTED_SOURCE_LABELS or len(taxonomy.get("classes", [])) != 203:
        raise T0Error("mapping or taxonomy contract has unexpected class count")
    records, audit, exceptions, trainval, test = audit_raw(voc_root, mapping_payload)
    labels_by_id = {image_id: {int(obj["ip102_class_id"]) for obj in records[image_id]["objects"]} for image_id in trainval}
    train, val, strategy = iterative_multilabel_split(trainval, labels_by_id, VAL_RATIO, SEED)
    split_sets = {"train": set(train), "val": set(val), "test": set(test)}
    if split_sets["train"] & split_sets["val"] or (split_sets["train"] | split_sets["val"]) != set(trainval):
        raise T0Error("derived train/val invariant failed")
    if split_sets["test"] != set(test) or split_sets["test"] & (split_sets["train"] | split_sets["val"]):
        raise T0Error("official test invariant failed")
    split_stats = {name: per_label_stats(ids, records) for name, ids in (("train", train), ("val", val), ("test", test))}
    test_source = {int(label) for label in split_stats["test"]}
    all_source = {int(label) for label in audit["per_source_label_all"]}
    missing_test = sorted(all_source - test_source)
    if missing_test != [KNOWN_TEST_MISSING_SOURCE_LABEL]:
        raise T0Error(f"unexpected test source-label gap: {missing_test}")
    split_summary = {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "invariants": {"train_val_nonempty": bool(train and val), "train_val_disjoint": True, "train_val_union_equals_official_trainval": True, "test_exact_official_order_and_membership": test == read_ids(voc_root / "ImageSets/Main/test.txt"), "test_disjoint": True},
        "source_label_coverage": {name: len(stats) for name, stats in split_stats.items()},
        "test_missing_ip102_source_labels": missing_test,
        "test_missing_detector_labels": [int(mapping_payload["classes"][[int(r["ip102_class_id"]) for r in mapping_payload["classes"]].index(KNOWN_TEST_MISSING_SOURCE_LABEL)]["detector_label"])],
        "per_source_label": split_stats,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "stage": "T0",
        "git_baseline": os.environ.get("T0_GIT_BASELINE", "a0791e60654035186d63000f9f18fd43084fb814"),
        "voc_root": repo_relative(voc_root),
        "mapping_source": repo_relative(mapping_path),
        "taxonomy_source": repo_relative(taxonomy_path),
        "official_trainval": "ImageSets/Main/trainval.txt",
        "official_test": "ImageSets/Main/test.txt",
        "split": strategy,
        "cleaning": {"xml_parser": "xml.etree.ElementTree", "duplicated_root_rule": "parse concatenated annotation documents and deduplicate identical document bytes", "invalid_box_rule": "filter only boxes that are non-finite, non-positive-area, or outside declared image bounds", "coordinate_convention": "source VOC coordinates preserved; validity requires 0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height"},
        "official_test_isolation": "Seed, ratio, cleaning, and train/val assignment were selected without consulting official test statistics, samples, visualizations, or metrics. Test is audited only after the split is frozen.",
        "runtime": {"python": platform.python_version(), "pillow": Image.__version__, "platform": platform.platform()},
        "source_sha256": {"official_trainval": sha256_file(voc_root / "ImageSets/Main/trainval.txt"), "official_test": sha256_file(voc_root / "ImageSets/Main/test.txt"), "mapping": sha256_file(mapping_path), "taxonomy": sha256_file(taxonomy_path), "builder": sha256_file(Path(__file__).resolve()), "dataset_verifier": sha256_file(DATASET_VERIFIER_SOURCE), "dataset": sha256_file(DATASET_SOURCE), "mapping_loader": sha256_file(MAPPING_SOURCE), "t0_test": sha256_file(TEST_SOURCE), "dataset_regression_test": sha256_file(DATASET_REGRESSION_TEST_SOURCE)},
    }

    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "train.txt", ("\n".join(train) + "\n").encode("utf-8"))
    write_new(output / "val.txt", ("\n".join(val) + "\n").encode("utf-8"))
    write_new(output / "test.txt", ("\n".join(test) + "\n").encode("utf-8"))
    annotations = b"".join(canonical_json(records[image_id], pretty=False) for image_id in trainval + test)
    write_new(output / "annotations.jsonl", annotations)
    write_new(output / "class-map.json", mapping_path.read_bytes())
    write_new(output / "audit-summary.json", canonical_json(audit))
    write_new(output / "split-summary.json", canonical_json(split_summary))
    write_new(output / "exceptions.json", canonical_json(exceptions))
    write_new(output / "build-config.json", canonical_json(config))
    handoff = f"""# IP102 Detection Data Handoff v1

Status: frozen and accepted by the project lead on 2026-08-12. This is the T0 data contract; T1 has not been executed.

## Inputs and policy

- Raw root: `{repo_relative(voc_root)}/` (read-only)
- Official trainval/test: {len(trainval):,}/{len(test):,}; official test is preserved exactly and was not used for split or cleaning decisions.
- Derived train/val: {len(train):,}/{len(val):,}, deterministic iterative multilabel stratification v1, seed {SEED}, validation ratio {VAL_RATIO:.2f}.
- All paths in annotations are relative to the VOC root. The five extra JPEGs are listed in `exceptions.json` and excluded.

## Annotation and label contract

- `annotations.jsonl` contains all {len(records):,} formal IDs, {EXPECTED_RAW_BOXES:,} traceable raw boxes and {EXPECTED_VALID_BOXES:,} effective boxes.
- `IP087000986.xml` is parsed as duplicated identical documents and counted once. The zero-width box in `IP046000898.xml` is filtered while its valid peer remains.
- IP102 source labels remain auditable; detector labels are contiguous `1..96` (`0` is background); public outputs use frozen DLCPD-25 `class_id 0..202`.
- IP102 classes 50 and 51 share DLCPD-25 public class 97 and one detector label. Official test has no source class 61; no support or AP may be fabricated.

## Consumer rules

Algorithm code must instantiate `IP102DetectionDataset` with this directory's `train.txt` or `val.txt`, this directory's `class-map.json`, and the raw VOC root. Evaluation may use only the unchanged `test.txt` after training configuration is frozen. Do not rescan directories, reinterpret XML, infer IDs, change splits, or expose detector-private labels as public IDs.

Verify with:

```bash
/home/zkf/pytorch-env/bin/python scripts/ip102/build_detection_data_t0.py --verify-only
/home/zkf/pytorch-env/bin/python scripts/ip102/verify_detection_dataset_t0.py --workers 8
sha256sum -c artifacts/data/ip102-detection-v1/checksums.sha256
```
"""
    write_new(output / "data-handoff.md", handoff.encode("utf-8"))
    checksum_paths = [output / name for name in OUTPUT_NAMES] + [Path(__file__).resolve(), DATASET_VERIFIER_SOURCE, DATASET_SOURCE, MAPPING_SOURCE, TEST_SOURCE, DATASET_REGRESSION_TEST_SOURCE, mapping_path, taxonomy_path, voc_root / "ImageSets/Main/trainval.txt", voc_root / "ImageSets/Main/test.txt"]
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in sorted(set(checksum_paths), key=repo_relative)]
    write_new(output / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    return {"train": len(train), "val": len(val), "test": len(test), "raw_boxes": EXPECTED_RAW_BOXES, "valid_boxes": EXPECTED_VALID_BOXES}


def verify_checksums(path: Path) -> int:
    checked = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, name = line.split("  ", 1)
        except ValueError as exc:
            raise T0Error(f"invalid checksum line {line_number}") from exc
        target = Path(name)
        if not target.is_absolute():
            target = REPO_ROOT / target
        if not target.is_file() or sha256_file(target) != expected:
            raise T0Error(f"checksum mismatch: {name}")
        checked += 1
    return checked


def load_annotation_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in result:
                raise T0Error(f"duplicate annotation ID at line {line_number}: {image_id}")
            result[image_id] = record
    return result


def verify(voc_root: Path, output: Path, *, verify_raw_hashes: bool = True) -> dict[str, Any]:
    missing = [name for name in (*OUTPUT_NAMES, "checksums.sha256") if not (output / name).is_file()]
    if missing:
        raise T0Error(f"missing T0 artifacts: {missing}")
    checksum_count = verify_checksums(output / "checksums.sha256")
    train, val, test = (read_ids(output / f"{name}.txt") for name in ("train", "val", "test"))
    official_trainval = read_ids(voc_root / "ImageSets/Main/trainval.txt")
    official_test = read_ids(voc_root / "ImageSets/Main/test.txt")
    if not train or not val or set(train) & set(val) or set(train) | set(val) != set(official_trainval):
        raise T0Error("train/val split invariant failed")
    if test != official_test or set(test) & (set(train) | set(val)):
        raise T0Error("test split invariant failed")
    records = load_annotation_index(output / "annotations.jsonl")
    if len(records) != EXPECTED_FORMAL or set(records) != set(official_trainval) | set(official_test):
        raise T0Error("annotation manifest coverage failed")
    raw_boxes = valid_boxes = 0
    source_labels: set[int] = set()
    detector_labels: set[int] = set()
    public_ids: set[int] = set()
    decoded = 0
    for image_id in official_trainval + official_test:
        record = records[image_id]
        image_path = Path(str(record["image_path"]))
        annotation_path = Path(str(record["annotation_path"]))
        if image_path.is_absolute() or annotation_path.is_absolute() or ".." in image_path.parts or ".." in annotation_path.parts:
            raise T0Error(f"unsafe relative path: {image_id}")
        image_file = voc_root / image_path
        xml_file = voc_root / annotation_path
        if not image_file.is_file() or not xml_file.is_file():
            raise T0Error(f"missing raw pair: {image_id}")
        if verify_raw_hashes and (sha256_file(image_file) != record["image_sha256"] or sha256_file(xml_file) != record["annotation_sha256"]):
            raise T0Error(f"raw source hash mismatch: {image_id}")
        with Image.open(image_file) as image:
            image.load()
            if image.size != (record["width"], record["height"]):
                raise T0Error(f"decoded dimension mismatch: {image_id}")
        decoded += 1
        raw_boxes += len(record["raw_objects"])
        valid_boxes += len(record["objects"])
        for obj in record["objects"]:
            box = obj["bbox"]
            if not all(math.isfinite(float(value)) for value in box) or not (0 <= box[0] < box[2] <= record["width"] and 0 <= box[1] < box[3] <= record["height"]):
                raise T0Error(f"invalid effective box: {image_id}")
            source_labels.add(int(obj["ip102_class_id"])); detector_labels.add(int(obj["detector_label"])); public_ids.add(int(obj["dlcpd25_class_id"]))
    if (raw_boxes, valid_boxes) != (EXPECTED_RAW_BOXES, EXPECTED_VALID_BOXES):
        raise T0Error("verified box totals mismatch")
    if len(source_labels) != 97 or detector_labels != set(range(1, 97)) or not all(0 <= value <= 202 for value in public_ids):
        raise T0Error("verified class spaces mismatch")
    exceptions = json.loads((output / "exceptions.json").read_text(encoding="utf-8"))
    if len(exceptions["extra_jpeg_files"]) != 5 or exceptions["missing_test_source_labels"] != [61]:
        raise T0Error("exception contract mismatch")
    return {"checksums": checksum_count, "decoded_images": decoded, "raw_boxes": raw_boxes, "valid_boxes": valid_boxes, "counts": {"train": len(train), "val": len(val), "test": len(test)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-raw-hashes", action="store_true", help="For focused tests only; formal verification must not use this.")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.voc_root.resolve(), args.output.resolve(), verify_raw_hashes=not args.skip_raw_hashes)
    else:
        result = build(args.voc_root.resolve(), args.mapping.resolve(), args.taxonomy.resolve(), args.output.resolve())
        result["verification"] = verify(args.voc_root.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

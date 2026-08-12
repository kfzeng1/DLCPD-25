#!/usr/bin/env python3
"""Build the audited IP102-to-DLCPD-25 detection class mapping."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from xml.etree import ElementTree

EXPECTED_IP102_IDS = set(range(102)) - {59, 60, 63, 75, 80}
MANUAL_DLCPD25_IDS = {
    8: 199,  # white backed plant hopper -> white-backed planthopper
    23: 88,  # IP102 hierarchy places army worm in the corn group
    24: 83,  # IP102 hierarchy places aphids in the corn group
    25: 59,  # official IP102 typo -> Protaetia brevitarsis
    29: 11,  # bird cherry-oataphid -> Bird cherry-oat aphid
    50: 97,  # legume blister beetle -> blister beetle
    66: 2,  # genus-only IP102 label -> Ampelophaga rubiginosa
    83: 14,  # Bactrocera minax synonym -> Chinese citrus fly
}


def normalize(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).lower()
    value = re.sub(r"[（(][^（）()]*[）)]", " ", value)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def read_ip102_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) != len(names) + 1:
            raise ValueError(f"invalid IP102 class row: {line!r}")
        names.append(parts[1].strip())
    if len(names) != 102:
        raise ValueError(f"expected 102 IP102 names, got {len(names)}")
    return names


def detected_ids(annotation_dir: Path) -> set[int]:
    result: set[int] = set()
    files = sorted(annotation_dir.glob("*.xml"))
    if len(files) != 18_976:
        raise ValueError(f"expected 18976 XML files, got {len(files)}")
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            root = ElementTree.fromstring(text)
            values = [obj.findtext("name") for obj in root.findall("object")]
        except ElementTree.ParseError:
            values = re.findall(r"<name>\s*(\d+)\s*</name>", text)
        result.update(int(value) for value in values if value is not None)
    if result != EXPECTED_IP102_IDS:
        raise ValueError(f"unexpected IP102 detection IDs: {sorted(result)}")
    return result


def build_mapping(names_path: Path, annotations: Path, taxonomy_path: Path) -> dict[str, object]:
    names = read_ip102_names(names_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or len(classes) != 203:
        raise ValueError("DLCPD-25 taxonomy must contain 203 classes")
    pests = [record for record in classes if record.get("category") == "pest"]
    normalized: dict[str, list[dict[str, object]]] = {}
    for record in pests:
        normalized.setdefault(normalize(str(record["official_name"])), []).append(record)

    records: list[dict[str, object]] = []
    direct_matches = 0
    for ip102_id in sorted(detected_ids(annotations)):
        ip102_name = names[ip102_id]
        if ip102_id in MANUAL_DLCPD25_IDS:
            class_id = MANUAL_DLCPD25_IDS[ip102_id]
            method = "reviewed_alias"
        else:
            candidates = normalized.get(normalize(ip102_name), [])
            if len(candidates) != 1:
                raise ValueError(f"IP102 class {ip102_id} has {len(candidates)} direct matches")
            class_id = int(candidates[0]["class_id"])
            method = "normalized_name"
            direct_matches += 1
        target = classes[class_id]
        if target.get("category") != "pest":
            raise ValueError(f"mapped DLCPD-25 class is not a pest: {class_id}")
        records.append(
            {
                "ip102_class_id": ip102_id,
                "dlcpd25_class_id": class_id,
                "ip102_name": ip102_name,
                "dlcpd25_name": target["official_name"],
                "match_method": method,
            }
        )
    public_ids = {record["dlcpd25_class_id"] for record in records}
    if direct_matches != 89 or len(public_ids) != 96:
        raise ValueError("mapping is not the expected 97 detector labels to 96 public classes")
    detector_labels = {
        class_id: label for label, class_id in enumerate(sorted(public_ids), start=1)
    }
    for record in records:
        record["detector_label"] = detector_labels[record["dlcpd25_class_id"]]
    return {
        "schema_version": 1,
        "public_class_id_space": "DLCPD-25 class_id 0..202",
        "detector_label_space": "1..96; 0 is Faster R-CNN background",
        "ip102_source_labels": 97,
        "detector_foreground_classes": 96,
        "mapped_dlcpd25_detection_classes": 96,
        "dlcpd25_classes": 203,
        "missing_ip102_detection_class_ids": [59, 60, 63, 75, 80],
        "match_summary": {"normalized_name": 89, "reviewed_alias_or_host": 8},
        "many_to_one_mapping": {
            "dlcpd25_class_id": 97,
            "ip102_class_ids": [50, 51],
            "reason": "DLCPD-25 merges IP102 legume blister beetle and blister beetle.",
        },
        "classes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("metadata/ip102-detection-class-map.json"))
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    payload = build_mapping(
        repo / "data/raw/ip102/downloads/Classification/classes.txt",
        repo / "data/raw/ip102/downloads/Detection/VOC2007/Annotations",
        repo / "metadata/class-taxonomy.json",
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

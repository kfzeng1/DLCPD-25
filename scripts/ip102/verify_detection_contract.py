#!/usr/bin/env python3
"""Independently verify the frozen IP102 Plan-A detection data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOC_ROOT = REPO_ROOT / "data/raw/ip102/VOC2007"
DEFAULT_CONTRACT = REPO_ROOT / "artifacts/data/ip102"
EXPECTED = {"train": 12_142, "val": 3_036, "test": 3_798}


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate ids in {path}")
    return values


def parse_xml(path: Path) -> list[ET.Element]:
    text = path.read_text(encoding="utf-8")
    try:
        return [ET.fromstring(text)]
    except ET.ParseError:
        import re
        docs = re.findall(r"<annotation\b.*?</annotation>", text, re.DOTALL)
        if not docs:
            raise
        seen: set[bytes] = set()
        roots: list[ET.Element] = []
        for doc in docs:
            root = ET.fromstring(doc)
            signature = ET.tostring(root)
            if signature in seen:
                continue
            seen.add(signature)
            roots.append(root)
        return roots


def verify_split(voc_root: Path, contract: Path, split: str, expected: int) -> dict[str, int]:
    ids = read_ids(contract / f"{split}.txt")
    if len(ids) != expected:
        raise ValueError(f"{split}: expected {expected} ids, got {len(ids)}")
    image_count = box_count = 0
    formats: Counter[str] = Counter()
    for image_id in ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        if not image_path.is_file() or not xml_path.is_file():
            raise ValueError(f"missing file pair: {image_id}")
        with Image.open(image_path) as image:
            image.load()
            formats[str(image.format)] += 1
        for root in parse_xml(xml_path):
            for obj in root.findall("object"):
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError(f"object without bndbox: {image_id}")
                coords = [float(box.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(value) for value in coords):
                    continue
                if coords[0] < coords[2] and coords[1] < coords[3]:
                    box_count += 1
        image_count += 1
    return {"images": image_count, "boxes": box_count, "formats": dict(sorted(formats.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    for split, expected in EXPECTED.items():
        result[split] = verify_split(args.voc_root.resolve(), args.contract.resolve(), split, expected)
    total_images = sum(result[split]["images"] for split in EXPECTED)
    total_boxes = sum(result[split]["boxes"] for split in EXPECTED)
    if total_images != 18_976 or total_boxes != 22_283:
        raise ValueError(f"contract totals mismatch: images={total_images}, boxes={total_boxes}")
    result["totals"] = {"images": total_images, "boxes": total_boxes}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

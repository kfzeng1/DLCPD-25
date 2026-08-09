#!/usr/bin/env python3
"""Build the reviewed DLCPD-25 class taxonomy and browsable symlink views."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = REPO_ROOT / "metadata" / "official-class-names.txt"
DEFAULT_ALIASES = REPO_ROOT / "metadata" / "class-directory-aliases.json"
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "dlcpd25-203"
DEFAULT_JSON = REPO_ROOT / "metadata" / "class-taxonomy.json"
DEFAULT_CSV = REPO_ROOT / "metadata" / "class-taxonomy.csv"
DEFAULT_VIEW = REPO_ROOT / "data" / "views" / "by-category"

CATEGORY_INFO = {
    "pest": ("农业有害生物", "01_pest_农业有害生物"),
    "disease": ("植物病害", "02_disease_植物病害"),
    "healthy": ("健康", "03_healthy_健康"),
    "disorder": ("非生物或生理缺陷", "04_disorder_非生物或生理缺陷"),
    "mixed": ("混合或歧义类别", "05_mixed_混合或歧义"),
}

# These sets use the official directory names, not translated local aliases.
DISEASES = {
    "Bacterial blight（cotton）",
    "Cotton curl virus（cotton）",
    "Fusarium wilt（cotton）",
    "Powdery mildew（cotton）",
    "Target spot（cotton）",
    "apple black rot",
    "apple frogeye spot",
    "apple scab",
    "bell pepper bacterial spot",
    "cedar apple rust",
    "cherry powdery mildew",
    "corn curvularia leaf spot fungus",
    "corn（maize） common rust",
    "corn（maize） northern leaf blight",
    "grape black rot（fungus）",
    "grape esca (black measles)(fungus)",
    "grape leaf blight(fungus)",
    "large wheat crown and root rot",
    "maize dwarf mosaic virus",
    "maize grey leaf spot(cercospora zeae-maydis tehon and daniels)",
    "orange huanglongbing(citrus greening)",
    "peach bacterial spot",
    "pepper scab",
    "potato early blight(fungus)",
    "potato late blight(fungus)",
    "puccinia polysora",
    "rice bacterial leaf blight",
    "rice blast",
    "rice brown spot",
    "rice leaf smut",
    "rice tungro",
    "soybean Crestamento Bacteriano",
    "soybean bacterial blight",
    "soybean brown spot",
    "soybean ferrugen",
    "soybean mossaic virus",
    "soybean powdery mildew",
    "soybean septoria brown spot",
    "soybean southern blight",
    "soybean sudden death syndrone",
    "soybean yellow mosaic",
    "squash powdery mildew",
    "strawberry leaf scorch",
    "tomato bacterial spot",
    "tomato early blight(fungus)",
    "tomato late blight(water mold)",
    "tomato leaf mold(fungus)",
    "tomato mosaic virus",
    "tomato powdery mildew",
    "tomato septoria leaf spot(fungus)",
    "tomato target spot(bacteria)",
    "tomato yellow leaf curl virus",
    "wheat leaf rust",
    "wheat septoria",
    "wheat stem rust",
    "wheat stripe rust",
}

HEALTHY = {
    "Healthy（cotton）",
    "apple healthy",
    "bell pepper healthy",
    "blueberry healthy",
    "cherry healthy",
    "citrus healthy",
    "corn healthy",
    "grape healthy",
    "peach healthy",
    "pepper healthy",
    "potato healthy",
    "raspberry healthy",
    "rice healthy",
    "soybean healthy",
    "strawberry healthy",
    "tomato healthy",
    "wheat healthy",
}

DISORDERS = {
    "Herbicide Growth Damage（cotton）",
    "Leaf Redding（cotton）",
    "Leaf Variegation（cotton）",
}

MIXED = {"garlic pest and diseases"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--no-view", action="store_true")
    return parser.parse_args()


def category_for(name: str) -> str:
    if name in DISEASES:
        return "disease"
    if name in HEALTHY:
        return "healthy"
    if name in DISORDERS:
        return "disorder"
    if name in MIXED:
        return "mixed"
    return "pest"


def load_inputs(classes_path: Path, aliases_path: Path) -> tuple[list[str], dict[str, str]]:
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    if len(classes) != 203 or len(set(classes)) != 203:
        raise ValueError("official class list must contain 203 unique names")
    if not isinstance(aliases, dict) or set(aliases) != set(classes):
        raise ValueError("alias map must match the official class list exactly")
    reviewed = DISEASES | HEALTHY | DISORDERS | MIXED
    unknown = reviewed - set(classes)
    if unknown:
        raise ValueError(f"taxonomy contains unknown official classes: {sorted(unknown)}")
    return classes, aliases


def count_files(path: Path) -> int:
    return sum(item.is_file() for item in path.rglob("*"))


def build_records(data_dir: Path, classes: list[str], aliases: dict[str, str]) -> list[dict[str, object]]:
    records = []
    for class_id, official_name in enumerate(classes):
        local_name = aliases[official_name]
        class_dir = data_dir / local_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"class directory not found: {class_dir}")
        category = category_for(official_name)
        records.append(
            {
                "class_id": class_id,
                "official_name": official_name,
                "local_directory": local_name,
                "category": category,
                "category_zh": CATEGORY_INFO[category][0],
                "image_count": count_files(class_dir),
            }
        )
    return records


def write_metadata(records: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    counts = Counter(str(item["category"]) for item in records)
    payload = {
        "schema_version": 1,
        "class_id_rule": "Zero-based order in metadata/official-class-names.txt.",
        "taxonomy_scope": "Project-level grouping reviewed from official class semantics; not an official DLCPD-25 hierarchy.",
        "categories": [
            {
                "id": key,
                "name_zh": value[0],
                "class_count": counts[key],
            }
            for key, value in CATEGORY_INFO.items()
        ],
        "classes": records,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def build_view(view_root: Path, data_dir: Path, records: list[dict[str, object]]) -> None:
    expected: set[Path] = set()
    for _, (_, directory) in CATEGORY_INFO.items():
        group_dir = view_root / directory
        group_dir.mkdir(parents=True, exist_ok=True)
        for entry in group_dir.iterdir():
            if entry.is_symlink():
                entry.unlink()
            else:
                raise ValueError(f"refusing to replace non-symlink view entry: {entry}")
    for record in records:
        group_dir = view_root / CATEGORY_INFO[str(record["category"])][1]
        link = group_dir / str(record["local_directory"])
        target = os.path.relpath(
            data_dir / str(record["local_directory"]),
            start=group_dir,
        )
        link.symlink_to(target, target_is_directory=True)
        expected.add(link)
    if len(expected) != len(records):
        raise ValueError("taxonomy view did not create one unique link per class")


def main() -> int:
    args = parse_args()
    try:
        classes, aliases = load_inputs(args.classes, args.aliases)
        records = build_records(args.data, classes, aliases)
        write_metadata(records, args.json, args.csv)
        if not args.no_view:
            build_view(args.view, args.data, records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    counts = Counter(str(item["category"]) for item in records)
    print(json.dumps({"classes": len(records), "images": sum(int(item["image_count"]) for item in records), "categories": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

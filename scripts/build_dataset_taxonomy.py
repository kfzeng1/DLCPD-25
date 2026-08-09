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
DEFAULT_VIEW = REPO_ROOT / "data" / "views" / "by-host"

CATEGORY_INFO = {
    "pest": ("农业有害生物", "01_pest_农业有害生物"),
    "disease": ("植物病害", "02_disease_植物病害"),
    "healthy": ("健康", "03_healthy_健康"),
    "disorder": ("非生物或生理缺陷", "04_disorder_非生物或生理缺陷"),
}

HOST_GROUPS = {
    "economic_crop": ("经济作物", "01_economic_crop_经济作物"),
    "food_crop": ("粮食作物", "02_food_crop_粮食作物"),
}

# Host order follows paper Table 1, excluding cucumber, which is absent from the
# current official 203-directory release.
HOST_INFO = {
    "citrus": ("economic_crop", "柑橘", "01_citrus_柑橘"),
    "tomato": ("economic_crop", "番茄", "02_tomato_番茄"),
    "grape": ("economic_crop", "葡萄", "03_grape_葡萄"),
    "apple": ("economic_crop", "苹果", "04_apple_苹果"),
    "soybean": ("economic_crop", "大豆", "05_soybean_大豆"),
    "peach": ("economic_crop", "桃", "06_peach_桃"),
    "mango": ("economic_crop", "芒果", "07_mango_芒果"),
    "alfalfa": ("economic_crop", "苜蓿", "08_alfalfa_苜蓿"),
    "bell_pepper": ("economic_crop", "甜椒", "09_bell_pepper_甜椒"),
    "strawberry": ("economic_crop", "草莓", "10_strawberry_草莓"),
    "cherry": ("economic_crop", "樱桃", "11_cherry_樱桃"),
    "cotton": ("economic_crop", "棉花", "12_cotton_棉花"),
    "squash": ("economic_crop", "南瓜", "13_squash_南瓜"),
    "blueberry": ("economic_crop", "蓝莓", "14_blueberry_蓝莓"),
    "raspberry": ("economic_crop", "树莓", "15_raspberry_树莓"),
    "beet": ("economic_crop", "甜菜", "16_beet_甜菜"),
    "pepper": ("economic_crop", "辣椒", "17_pepper_辣椒"),
    "garlic": ("economic_crop", "大蒜", "18_garlic_大蒜"),
    "corn": ("food_crop", "玉米", "19_corn_玉米"),
    "rice": ("food_crop", "水稻", "20_rice_水稻"),
    "potato": ("food_crop", "马铃薯", "21_potato_马铃薯"),
    "wheat": ("food_crop", "小麦", "22_wheat_小麦"),
}

HOST_ALIASES = {
    "bell pepper": "bell_pepper",
    "citrus": "citrus",
    "citru": "citrus",
    "tomato": "tomato",
    "vitis": "grape",
    "grape": "grape",
    "apple": "apple",
    "soybean": "soybean",
    "peach": "peach",
    "mango": "mango",
    "alfalfa": "alfalfa",
    "strawberry": "strawberry",
    "cherry": "cherry",
    "cotton": "cotton",
    "squash": "squash",
    "blueberry": "blueberry",
    "raspberry": "raspberry",
    "beet": "beet",
    "pepper": "pepper",
    "garlic": "garlic",
    "corn": "corn",
    "maize": "corn",
    "rice": "rice",
    "potato": "potato",
    "wheat": "wheat",
}

HOST_OVERRIDES = {
    "puccinia polysora": "corn",
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
    "garlic pest and diseases",
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
    return "pest"


def host_for(name: str) -> str:
    if name in HOST_OVERRIDES:
        return HOST_OVERRIDES[name]
    normalized = name.lower().replace("（", "(").replace("）", ")")
    checks = (
        lambda alias: f"({alias})" in normalized or f"({alias} " in normalized,
        lambda alias: normalized.startswith(alias + " ") or normalized.startswith(alias + "(") or normalized == alias,
        lambda alias: alias in normalized.replace("(", " ").replace(")", " ").split(),
    )
    for check in checks:
        matches = {host_id for alias, host_id in HOST_ALIASES.items() if check(alias)}
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise ValueError(f"class resolves to multiple hosts: {name!r} -> {sorted(matches)}")
    raise ValueError(f"class does not resolve to a host: {name!r}")


def load_inputs(classes_path: Path, aliases_path: Path) -> tuple[list[str], dict[str, str]]:
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    if len(classes) != 203 or len(set(classes)) != 203:
        raise ValueError("official class list must contain 203 unique names")
    if not isinstance(aliases, dict) or set(aliases) != set(classes):
        raise ValueError("alias map must match the official class list exactly")
    reviewed = DISEASES | HEALTHY | DISORDERS
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
        host_id = host_for(official_name)
        host_group, host_zh, _ = HOST_INFO[host_id]
        records.append(
            {
                "class_id": class_id,
                "official_name": official_name,
                "local_directory": local_name,
                "host_group": host_group,
                "host_id": host_id,
                "host_zh": host_zh,
                "category": category,
                "category_zh": CATEGORY_INFO[category][0],
                "image_count": count_files(class_dir),
            }
        )
    return records


def write_metadata(records: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    counts = Counter(str(item["category"]) for item in records)
    host_class_counts = Counter(str(item["host_id"]) for item in records)
    host_image_counts = Counter()
    for item in records:
        host_image_counts[str(item["host_id"])] += int(item["image_count"])
    payload = {
        "schema_version": 2,
        "class_id_rule": "Zero-based order in metadata/official-class-names.txt.",
        "hierarchy": "host -> label_category -> class_label",
        "taxonomy_scope": "Host hierarchy follows paper Table 1 where compatible with the current official 203-directory release; label categories are project-reviewed attributes.",
        "host_groups": [
            {"id": key, "name_zh": value[0]}
            for key, value in HOST_GROUPS.items()
        ],
        "hosts": [
            {
                "id": host_id,
                "name_zh": values[1],
                "group": values[0],
                "view_directory": values[2],
                "class_count": host_class_counts[host_id],
                "image_count": host_image_counts[host_id],
            }
            for host_id, values in HOST_INFO.items()
        ],
        "categories": [
            {
                "id": key,
                "name_zh": value[0],
                "view_directory": value[1],
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
    for _, (_, _, host_directory) in HOST_INFO.items():
        host_dir = view_root / host_directory
        for _, (_, category_directory) in CATEGORY_INFO.items():
            category_dir = host_dir / category_directory
            category_dir.mkdir(parents=True, exist_ok=True)
            for entry in category_dir.iterdir():
                if entry.is_symlink():
                    entry.unlink()
                else:
                    raise ValueError(f"refusing to replace non-symlink view entry: {entry}")
    for record in records:
        _, _, host_directory = HOST_INFO[str(record["host_id"])]
        category_directory = CATEGORY_INFO[str(record["category"])][1]
        category_dir = view_root / host_directory / category_directory
        link = category_dir / str(record["local_directory"])
        target = os.path.relpath(
            data_dir / str(record["local_directory"]),
            start=category_dir,
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
    print(json.dumps({"classes": len(records), "hosts": len({item['host_id'] for item in records}), "images": sum(int(item["image_count"]) for item in records), "categories": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

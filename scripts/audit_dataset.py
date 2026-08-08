#!/usr/bin/env python3
"""Audit a local DLCPD-25 subset against the official 203 directory names."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = REPO_ROOT / "metadata" / "official-class-names.txt"
DEFAULT_ALIASES = REPO_ROOT / "metadata" / "class-directory-aliases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a DLCPD-25 subset contains all official class directories. "
            "Image counts may differ from the paper."
        )
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "data",
        help="directory containing the class folders",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES,
        help="newline-delimited expected class names",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON report to this path in addition to stdout",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=DEFAULT_ALIASES,
        help="JSON map from official names to accepted local directory names",
    )
    return parser.parse_args()


def load_expected(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"class list not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate entries in class list: {path}")
    return names


def load_aliases(path: Path, expected: list[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"alias map not found: {path}")
    aliases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in aliases.items()
    ):
        raise ValueError(f"alias map must be a string-to-string JSON object: {path}")
    if set(aliases) != set(expected):
        raise ValueError("alias map keys do not match the expected class list")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("alias map contains duplicate local directory names")
    return aliases


def audit(
    data_dir: Path, expected: list[str], aliases: dict[str, str]
) -> dict[str, object]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data directory not found: {data_dir}")

    actual_dirs = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    expected_set = set(expected)
    alias_to_official = {local: official for official, local in aliases.items()}

    canonical_to_dirs: dict[str, list[str]] = {}
    extra: list[str] = []
    for directory_name in actual_dirs:
        if directory_name in expected_set:
            official_name = directory_name
        elif directory_name in alias_to_official:
            official_name = alias_to_official[directory_name]
        else:
            extra.append(directory_name)
            continue
        canonical_to_dirs.setdefault(official_name, []).append(directory_name)

    missing = sorted(expected_set - set(canonical_to_dirs))
    duplicate_representations = {
        official: names
        for official, names in sorted(canonical_to_dirs.items())
        if len(names) > 1
    }

    per_class: dict[str, int] = {}
    extensions: Counter[str] = Counter()
    total_files = 0
    directory_names: dict[str, str] = {}
    for official_name, names in sorted(canonical_to_dirs.items()):
        class_name = names[0]
        class_dir = data_dir / class_name
        count = 0
        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue
            count += 1
            total_files += 1
            extensions[path.suffix.lower() or "<none>"] += 1
        per_class[official_name] = count
        directory_names[official_name] = class_name

    empty_classes = sorted(name for name, count in per_class.items() if count == 0)
    complete = (
        not missing
        and not extra
        and not empty_classes
        and not duplicate_representations
        and len(canonical_to_dirs) == len(expected)
    )
    return {
        "schema_version": 1,
        "data_dir": str(data_dir.resolve()),
        "acceptance_policy": "all_official_classes_present; image_count_may_vary",
        "expected_class_count": len(expected),
        "actual_class_count": len(actual_dirs),
        "class_set_complete": complete,
        "missing_classes": missing,
        "extra_classes": sorted(extra),
        "duplicate_class_representations": duplicate_representations,
        "empty_classes": empty_classes,
        "total_files": total_files,
        "extensions": dict(sorted(extensions.items())),
        "directory_names": directory_names,
        "per_class": per_class,
    }


def main() -> int:
    args = parse_args()
    try:
        expected = load_expected(args.classes)
        report = audit(args.data_dir, expected, load_aliases(args.aliases, expected))
    except (FileNotFoundError, NotADirectoryError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["class_set_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

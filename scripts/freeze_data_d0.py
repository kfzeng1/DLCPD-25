#!/usr/bin/env python3
"""Freeze and verify the D0 data-root and taxonomy-v1 contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "metadata" / "d0-freeze-config-v1.json"


class FreezeError(ValueError):
    """Raised when the current data root does not match the D0 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="D0 freeze configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the configured artifact directory",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def repo_path(value: str, field: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise FreezeError(f"{field} must remain inside the repository: {value}") from exc
    return path


def require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FreezeError(f"{field} must be a JSON object with string keys")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = require_mapping(load_json(path), "config")
    required = {
        "schema_version",
        "freeze_id",
        "taxonomy_version",
        "target_data_version",
        "release_state",
        "data_root",
        "output_dir",
        "sources",
        "expected",
        "category_semantics",
        "inventory_fingerprint",
    }
    missing = sorted(required - set(config))
    if missing:
        raise FreezeError(f"config is missing fields: {missing}")
    if config["schema_version"] != 1:
        raise FreezeError("unsupported D0 config schema_version")
    return config


def load_official_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(names) != len(set(names)):
        raise FreezeError("official class list contains duplicate names")
    return names


def validate_csv(path: Path, classes: list[dict[str, Any]]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = list(classes[0])
        if reader.fieldnames != expected_fields:
            raise FreezeError("taxonomy CSV fields do not match taxonomy JSON")
        rows = list(reader)
    if len(rows) != len(classes):
        raise FreezeError("taxonomy CSV row count does not match taxonomy JSON")
    for row_index, (row, record) in enumerate(zip(rows, classes, strict=True)):
        expected = {key: str(value) for key, value in record.items()}
        if row != expected:
            raise FreezeError(f"taxonomy CSV differs from JSON at row {row_index}")


def validate_metadata(
    config: dict[str, Any],
    official_names: list[str],
    aliases: dict[str, Any],
    taxonomy: dict[str, Any],
    taxonomy_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    expected = require_mapping(config["expected"], "expected")
    expected_classes = int(expected["class_count"])
    if len(official_names) != expected_classes:
        raise FreezeError("official class count does not match the D0 contract")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()):
        raise FreezeError("class aliases must be a string-to-string mapping")
    if set(aliases) != set(official_names) or len(set(aliases.values())) != expected_classes:
        raise FreezeError("class aliases are not a unique mapping of all official classes")

    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or not all(isinstance(item, dict) for item in classes):
        raise FreezeError("taxonomy classes must be an array of objects")
    if len(classes) != expected_classes:
        raise FreezeError("taxonomy class count does not match the D0 contract")

    class_ids = [item.get("class_id") for item in classes]
    expected_ids = list(range(int(expected["class_id_min"]), int(expected["class_id_max"]) + 1))
    if class_ids != expected_ids:
        raise FreezeError("taxonomy class IDs must be ordered, unique, and continuous")
    taxonomy_names = [item.get("official_name") for item in classes]
    if taxonomy_names != official_names:
        raise FreezeError("official class list and taxonomy class order differ")
    if any(item.get("local_directory") != aliases[item["official_name"]] for item in classes):
        raise FreezeError("taxonomy local directories differ from the alias mapping")

    category_counts = Counter(str(item.get("category")) for item in classes)
    expected_categories = {
        str(key): int(value)
        for key, value in require_mapping(
            expected["category_class_counts"], "expected.category_class_counts"
        ).items()
    }
    if dict(category_counts) != expected_categories:
        raise FreezeError("taxonomy category counts do not match the D0 contract")
    semantics = require_mapping(config["category_semantics"], "category_semantics")
    if any(item.get("category_zh") != semantics[item["category"]] for item in classes):
        raise FreezeError("taxonomy category semantics do not match the D0 contract")

    hosts = taxonomy.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != int(expected["host_count"]):
        raise FreezeError("taxonomy host count does not match the D0 contract")
    host_ids = [item.get("id") for item in hosts if isinstance(item, dict)]
    if len(host_ids) != len(set(host_ids)) or set(host_ids) != {
        item.get("host_id") for item in classes
    }:
        raise FreezeError("taxonomy hosts are not unique or do not cover all classes")
    host_class_counts = Counter(str(item.get("host_id")) for item in classes)
    host_image_counts: Counter[str] = Counter()
    for item in classes:
        host_image_counts[str(item.get("host_id"))] += int(item.get("image_count", 0))
    for host in hosts:
        host_id = str(host.get("id"))
        if host.get("class_count") != host_class_counts[host_id]:
            raise FreezeError(f"published class count differs for host: {host_id}")
        if host.get("image_count") != host_image_counts[host_id]:
            raise FreezeError(f"published image count differs for host: {host_id}")

    published_categories = taxonomy.get("categories")
    if not isinstance(published_categories, list):
        raise FreezeError("taxonomy categories must be an array")
    published = {
        str(item.get("id")): (item.get("name_zh"), item.get("class_count"))
        for item in published_categories
        if isinstance(item, dict)
    }
    frozen = {
        category: (semantics[category], count)
        for category, count in expected_categories.items()
    }
    if published != frozen:
        raise FreezeError("published taxonomy category metadata differs from the contract")

    validate_csv(taxonomy_csv, classes)
    return classes, expected_categories


def scan_data_root(
    data_root: Path,
    classes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not data_root.is_dir():
        raise FreezeError(f"data root does not exist: {data_root}")
    if data_root.is_symlink():
        raise FreezeError("data root must not be a symlink")

    expected_directories = {str(item["local_directory"]) for item in classes}
    visible_entries = {entry.name: entry for entry in data_root.iterdir() if not entry.name.startswith(".")}
    missing = sorted(expected_directories - set(visible_entries))
    extra = sorted(set(visible_entries) - expected_directories)
    if missing or extra:
        raise FreezeError(f"data-root class entries differ: missing={missing}, extra={extra}")
    invalid = sorted(
        name
        for name, entry in visible_entries.items()
        if not entry.is_dir() or entry.is_symlink()
    )
    if invalid:
        raise FreezeError(f"class entries must be real directories: {invalid}")

    digest = hashlib.sha256()
    extensions: Counter[str] = Counter()
    per_class: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for record in classes:
        class_dir = data_root / str(record["local_directory"])
        files: list[Path] = []
        for path in class_dir.rglob("*"):
            if path.is_symlink():
                raise FreezeError(f"raw class directory contains a symlink: {path}")
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise FreezeError(f"raw class directory contains a special entry: {path}")
        files.sort(key=lambda path: path.relative_to(data_root).as_posix())
        class_bytes = 0
        for path in files:
            relative_path = path.relative_to(data_root).as_posix()
            size = path.stat().st_size
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\n")
            class_bytes += size
            total_bytes += size
            extensions[path.suffix.lower() or "<none>"] += 1
        file_count = len(files)
        if file_count == 0:
            raise FreezeError(f"empty class directory: {record['local_directory']}")
        if record.get("image_count") != file_count:
            raise FreezeError(
                f"taxonomy count differs for class_id={record['class_id']}: "
                f"taxonomy={record.get('image_count')}, actual={file_count}"
            )
        total_files += file_count
        per_class.append(
            {
                "class_id": record["class_id"],
                "official_name": record["official_name"],
                "local_directory": record["local_directory"],
                "file_count": file_count,
                "total_bytes": class_bytes,
            }
        )
    try:
        display_root = data_root.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_root = data_root.as_posix()
    return {
        "relative_path": display_root,
        "class_directory_count": len(classes),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "extensions": dict(sorted(extensions.items())),
        "inventory_sha256": digest.hexdigest(),
        "inventory_scope": "relative paths and file sizes; file contents are deferred to D2",
        "ignored_hidden_root_entries": sorted(
            entry.name for entry in data_root.iterdir() if entry.name.startswith(".")
        ),
        "per_class": per_class,
    }


def write_idempotent(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FreezeError(f"refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_freeze(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    sources_config = require_mapping(config["sources"], "sources")
    sources = {key: repo_path(str(value), f"sources.{key}") for key, value in sources_config.items()}
    for key, path in sources.items():
        if not path.is_file():
            raise FreezeError(f"source file does not exist: {key}={path}")

    expected = require_mapping(config["expected"], "expected")
    official_names = load_official_names(sources["official_class_names"])
    if sha256_file(sources["official_class_names"]) != expected["official_class_names_sha256"]:
        raise FreezeError("official class-list SHA-256 differs from the D0 contract")
    aliases = require_mapping(load_json(sources["class_directory_aliases"]), "aliases")
    taxonomy = require_mapping(load_json(sources["class_taxonomy_json"]), "taxonomy")
    classes, category_counts = validate_metadata(
        config,
        official_names,
        aliases,
        taxonomy,
        sources["class_taxonomy_csv"],
    )
    data_root = repo_path(str(config["data_root"]), "data_root")
    inventory = scan_data_root(data_root, classes)
    if inventory["total_files"] != int(expected["total_files"]):
        raise FreezeError("data-root file count does not match the D0 contract")

    if output_override is None:
        output_dir = repo_path(str(config["output_dir"]), "output_dir")
    else:
        output_dir = output_override.resolve()
    snapshots = {
        "d0-freeze-config-v1.json": config_path,
        "official-class-names-v1.txt": sources["official_class_names"],
        "class-directory-aliases-v1.json": sources["class_directory_aliases"],
        "class-taxonomy-v1.json": sources["class_taxonomy_json"],
        "class-taxonomy-v1.csv": sources["class_taxonomy_csv"],
    }
    for name, source in snapshots.items():
        write_idempotent(output_dir / name, source.read_bytes())

    source_checksums = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(sources.values()) | {config_path})
    }
    snapshot_checksums = {
        name: sha256_file(output_dir / name) for name in sorted(snapshots)
    }
    report = {
        "schema_version": 1,
        "freeze_id": config["freeze_id"],
        "taxonomy_version": config["taxonomy_version"],
        "target_data_version": config["target_data_version"],
        "release_state": config["release_state"],
        "git_commit": git_commit(),
        "class_id_source": "metadata/class-taxonomy.json classes array",
        "class_id_range": [expected["class_id_min"], expected["class_id_max"]],
        "class_count": len(classes),
        "host_count": len(taxonomy["hosts"]),
        "category_class_counts": category_counts,
        "category_semantics": config["category_semantics"],
        "data_root": inventory,
        "source_sha256": source_checksums,
        "snapshot_sha256": snapshot_checksums,
        "stage_boundaries": {
            "image_decode_audit": "not_performed_d1",
            "per_file_content_sha256": "not_performed_d2",
            "dhash_and_duplicate_groups": "not_performed_d2",
            "train_val_test_split": "not_performed_d3"
        },
    }
    report_path = output_dir / "d0-freeze.json"
    write_idempotent(report_path, canonical_json(report))

    checksum_paths = sorted(set(sources.values()) | {config_path})
    checksum_paths.extend(sorted(output_dir / name for name in snapshots))
    checksum_paths.append(report_path)
    checksum_lines = []
    for path in checksum_paths:
        try:
            display = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display = path.as_posix()
        checksum_lines.append(f"{sha256_file(path)}  {display}")
    write_idempotent(
        output_dir / "checksums.sha256",
        ("\n".join(checksum_lines) + "\n").encode("utf-8"),
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        report = build_freeze(args.config, args.output_dir)
    except (FreezeError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "freeze_id": report["freeze_id"],
        "classes": report["class_count"],
        "files": report["data_root"]["total_files"],
        "hosts": report["host_count"],
        "categories": report["category_class_counts"],
        "inventory_sha256": report["data_root"]["inventory_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

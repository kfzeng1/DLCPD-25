"""Fast data-contract checks run before A1 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
CSV_FIELDS = ("relative_path", "class_id", "sha256", "duplicate_group_id", "split")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_CLASS_IDS = tuple(range(203))
EXPECTED_HOSTS = 22
EXPECTED_CATEGORIES = {"pest", "disease", "healthy", "disorder"}


class PreflightError(ValueError):
    """Raised when frozen data does not satisfy the training contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreflightError(f"JSON root must be an object: {path}")
    return payload


def resolve_repo_path(repo_root: Path, value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PreflightError(f"unsafe {label} path: {value}")
    path = repo_root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"missing or symlinked {label}: {value}")
    return path


def inspect_taxonomy(path: Path, expected_sha256: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PreflightError(
            f"taxonomy SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = load_json(path)
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != len(EXPECTED_CLASS_IDS):
        raise PreflightError("taxonomy must contain exactly 203 classes")
    class_ids = tuple(item.get("class_id") for item in classes if isinstance(item, dict))
    if class_ids != EXPECTED_CLASS_IDS:
        raise PreflightError("taxonomy class IDs must be contiguous and ordered from 0 to 202")
    local_directories = [item.get("local_directory") for item in classes]
    official_names = [item.get("official_name") for item in classes]
    if len(set(local_directories)) != 203 or not all(isinstance(value, str) and value for value in local_directories):
        raise PreflightError("taxonomy local directories must be non-empty and unique")
    if len(set(official_names)) != 203 or not all(isinstance(value, str) and value for value in official_names):
        raise PreflightError("taxonomy official names must be non-empty and unique")
    hosts = {item.get("host_id") for item in classes}
    categories = {item.get("category") for item in classes}
    if len(hosts) != EXPECTED_HOSTS:
        raise PreflightError(f"taxonomy must contain {EXPECTED_HOSTS} hosts, got {len(hosts)}")
    if categories != EXPECTED_CATEGORIES:
        raise PreflightError(f"unexpected taxonomy categories: {sorted(categories)}")
    return {int(item["class_id"]): item for item in classes}, {
        "status": "passed",
        "sha256": actual_sha256,
        "class_count": len(classes),
        "class_id_min": min(class_ids),
        "class_id_max": max(class_ids),
        "host_count": len(hosts),
        "categories": sorted(categories),
    }


def verify_checksum_manifest(repo_root: Path, path: Path) -> dict[str, Any]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            expected, relative_path = raw_line.split("  ", 1)
        except ValueError as exc:
            raise PreflightError(f"invalid checksum line {line_number}: {raw_line}") from exc
        if not SHA256_PATTERN.fullmatch(expected):
            raise PreflightError(f"invalid SHA-256 at checksum line {line_number}")
        if relative_path in seen:
            raise PreflightError(f"duplicate checksum path: {relative_path}")
        seen.add(relative_path)
        target = resolve_repo_path(repo_root, relative_path, "checksum target")
        actual = sha256_file(target)
        if actual != expected:
            raise PreflightError(
                f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        entries.append((relative_path, actual))
    if not entries:
        raise PreflightError("D5 checksum manifest is empty")
    return {
        "status": "passed",
        "manifest": path.relative_to(repo_root).as_posix(),
        "manifest_sha256": sha256_file(path),
        "verified_files": len(entries),
    }


def inspect_splits(
    repo_root: Path,
    data_root: Path,
    split_contracts: dict[str, Any],
    expected_statistics: dict[str, Any],
    required_fields: Iterable[str],
    classes: dict[int, dict[str, Any]],
    verify_source_files: bool = True,
) -> dict[str, Any]:
    if tuple(required_fields) != CSV_FIELDS:
        raise PreflightError(f"unexpected required split fields: {tuple(required_fields)}")
    if not data_root.is_dir() or data_root.is_symlink():
        raise PreflightError(f"invalid data root: {data_root}")

    seen_paths: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    sha_groups: dict[str, tuple[str, str]] = {}
    split_results: dict[str, Any] = {}
    all_groups: set[str] = set()

    for split in SPLITS:
        contract = split_contracts.get(split)
        statistic = expected_statistics.get(split)
        if not isinstance(contract, dict) or not isinstance(statistic, dict):
            raise PreflightError(f"missing contract or statistics for split: {split}")
        split_path = resolve_repo_path(repo_root, str(contract.get("path", "")), f"{split} split")
        expected_hash = contract.get("sha256")
        actual_hash = sha256_file(split_path)
        if expected_hash != actual_hash or statistic.get("sha256") != actual_hash:
            raise PreflightError(f"{split} split SHA-256 does not match the frozen release")
        if contract.get("size_bytes") != split_path.stat().st_size:
            raise PreflightError(f"{split} split size does not match the frozen release")

        class_counts: Counter[int] = Counter()
        row_count = 0
        split_groups: set[str] = set()
        with split_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise PreflightError(f"unexpected {split} CSV schema: {reader.fieldnames}")
            for row_number, row in enumerate(reader, start=2):
                relative_path = row["relative_path"]
                pure_path = PurePosixPath(relative_path)
                if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
                    raise PreflightError(f"unsafe path in {split} row {row_number}: {relative_path}")
                previous_split = seen_paths.setdefault(relative_path, split)
                if previous_split != split or previous_split == split and relative_path in seen_paths and seen_paths[relative_path] != split:
                    raise PreflightError(f"path crosses splits: {relative_path}")
                # setdefault cannot distinguish a duplicate inside the same CSV, so compare row totals below.
                try:
                    class_id = int(row["class_id"])
                except ValueError as exc:
                    raise PreflightError(f"invalid class ID in {split} row {row_number}") from exc
                class_record = classes.get(class_id)
                if class_record is None:
                    raise PreflightError(f"unknown class ID {class_id} in {split} row {row_number}")
                if pure_path.parts[0] != class_record["local_directory"]:
                    raise PreflightError(f"path/taxonomy mismatch in {split} row {row_number}")
                if row["split"] != split:
                    raise PreflightError(f"split field mismatch in {split} row {row_number}")
                file_sha256 = row["sha256"]
                group_id = row["duplicate_group_id"]
                if not SHA256_PATTERN.fullmatch(file_sha256) or not group_id:
                    raise PreflightError(f"invalid hash or group in {split} row {row_number}")
                previous_group_split = group_splits.setdefault(group_id, split)
                if previous_group_split != split:
                    raise PreflightError(f"duplicate group crosses splits: {group_id}")
                previous_sha = sha_groups.setdefault(file_sha256, (group_id, split))
                if previous_sha != (group_id, split):
                    raise PreflightError(f"identical SHA-256 has inconsistent group or split: {file_sha256}")
                if verify_source_files:
                    source = data_root.joinpath(*pure_path.parts)
                    if not source.is_file() or source.is_symlink():
                        raise PreflightError(f"missing or symlinked source image: {relative_path}")
                class_counts[class_id] += 1
                split_groups.add(group_id)
                all_groups.add(group_id)
                row_count += 1

        if row_count != len({path for path, owner in seen_paths.items() if owner == split}):
            raise PreflightError(f"duplicate relative path within {split} split")
        if row_count != statistic.get("count"):
            raise PreflightError(
                f"{split} count mismatch: expected {statistic.get('count')}, got {row_count}"
            )
        coverage = sorted(class_counts)
        if coverage != list(EXPECTED_CLASS_IDS):
            raise PreflightError(f"{split} does not cover all 203 classes")
        split_results[split] = {
            "status": "passed",
            "path": split_path.relative_to(repo_root).as_posix(),
            "sha256": actual_hash,
            "rows": row_count,
            "class_coverage": len(coverage),
            "duplicate_groups": len(split_groups),
            "min_class_samples": min(class_counts.values()),
            "max_class_samples": max(class_counts.values()),
        }

    total_rows = sum(item["rows"] for item in split_results.values())
    return {
        "status": "passed",
        "splits": split_results,
        "total_rows": total_rows,
        "unique_paths": len(seen_paths),
        "unique_duplicate_groups": len(all_groups),
        "unique_content_hashes": len(sha_groups),
        "path_overlap_count": 0,
        "duplicate_group_leakage_count": 0,
        "sha_group_conflict_count": 0,
        "source_files_checked": total_rows if verify_source_files else 0,
    }


def run_preflight(
    repo_root: Path,
    release_path: Path,
    data_root: Path,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    release_path = release_path.resolve()
    data_root = data_root.resolve()
    release = load_json(release_path)
    if release.get("schema_version") != 2 or release.get("stage") != "D5-R1":
        raise PreflightError("training requires the D5-R1 schema_version 2 release")
    if release.get("data_version") != "data-v1":
        raise PreflightError("training requires data-v1")
    if release.get("formal_chain") != ["D0", "D1", "D2-R2", "D3-R2", "D4-R1", "D5-R1"]:
        raise PreflightError("unexpected formal data chain")

    checksum_path = release_path.parent / "checksums.sha256"
    checksum_result = verify_checksum_manifest(repo_root, checksum_path)
    fixed_contract = release.get("fixed_contract")
    statistics = release.get("statistics")
    if not isinstance(fixed_contract, dict) or not isinstance(statistics, dict):
        raise PreflightError("release is missing fixed_contract or statistics")
    taxonomy_contract = fixed_contract.get("taxonomy_snapshot")
    if not isinstance(taxonomy_contract, dict):
        raise PreflightError("release is missing the taxonomy snapshot contract")
    taxonomy_path = resolve_repo_path(
        repo_root, str(taxonomy_contract.get("path", "")), "taxonomy snapshot"
    )
    expected_taxonomy_sha256 = str(release.get("taxonomy_sha256", ""))
    if taxonomy_contract.get("sha256") != expected_taxonomy_sha256:
        raise PreflightError("release taxonomy hashes disagree")
    classes, taxonomy_result = inspect_taxonomy(taxonomy_path, expected_taxonomy_sha256)

    for label in ("manifest", "duplicate_groups"):
        contract = fixed_contract.get(label)
        if not isinstance(contract, dict):
            raise PreflightError(f"release is missing {label} contract")
        path = resolve_repo_path(repo_root, str(contract.get("path", "")), label)
        if path.stat().st_size != contract.get("size_bytes") or sha256_file(path) != contract.get("sha256"):
            raise PreflightError(f"frozen {label} does not match its contract")

    split_result = inspect_splits(
        repo_root=repo_root,
        data_root=data_root,
        split_contracts=fixed_contract.get("splits", {}),
        expected_statistics=statistics.get("splits", {}),
        required_fields=fixed_contract.get("required_fields", ()),
        classes=classes,
        verify_source_files=verify_source_files,
    )
    if split_result["total_rows"] != statistics.get("usable_files"):
        raise PreflightError("split total does not match usable_files")
    if statistics.get("total_files") != statistics.get("usable_files") + statistics.get("bad_files"):
        raise PreflightError("release total_files does not equal usable_files plus bad_files")

    return {
        "schema_version": 1,
        "check": "training-preflight",
        "status": "passed",
        "data_version": release["data_version"],
        "input_release": release_path.relative_to(repo_root).as_posix(),
        "input_release_sha256": sha256_file(release_path),
        "input_release_status": release.get("release_status"),
        "taxonomy": taxonomy_result,
        "d5_checksums": checksum_result,
        "fixed_contract": {
            "status": "passed",
            "manifest_sha256": fixed_contract["manifest"]["sha256"],
            "duplicate_groups_sha256": fixed_contract["duplicate_groups"]["sha256"],
        },
        "split_integrity": split_result,
        "release_statistics": {
            "total_files": statistics["total_files"],
            "usable_files": statistics["usable_files"],
            "bad_files": statistics["bad_files"],
            "long_tail_class_count": len(release.get("long_tail_classes", [])),
        },
        "known_limitations": release.get("known_limitations", []),
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/data/v1/d5-r1/data-v1-release.json")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/dlcpd25-203"))
    parser.add_argument("--skip-source-file-check", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    release_path = args.release if args.release.is_absolute() else repo_root / args.release
    data_root = args.data_root if args.data_root.is_absolute() else repo_root / args.data_root
    try:
        report = run_preflight(
            repo_root,
            release_path,
            data_root,
            verify_source_files=not args.skip_source_file_check,
        )
    except (PreflightError, OSError, json.JSONDecodeError) as exc:
        print(f"training preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "rows": report["split_integrity"]["total_rows"],
        "classes": report["taxonomy"]["class_count"],
        "path_overlap": report["split_integrity"]["path_overlap_count"],
        "group_leakage": report["split_integrity"]["duplicate_group_leakage_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

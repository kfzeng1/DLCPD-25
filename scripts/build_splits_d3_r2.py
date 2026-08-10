#!/usr/bin/env python3
"""Build standalone, deterministic D3-R2 train/val/test splits from D2-R1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D2_R1 = REPO_ROOT / "artifacts" / "data" / "v1" / "d2-r1"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "data" / "v1" / "d3-r2"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
TEST_SOURCE = REPO_ROOT / "project" / "tests" / "test_splits_d3_r2.py"
SEED = 20260809
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
ALGORITHM_VERSION = "sparse-group-stratified-greedy-v1"
IMPLEMENTATION_VERSION = "standalone-d3-r2-v1"
CSV_FIELDS = ("relative_path", "class_id", "sha256", "duplicate_group_id", "split")
EXCLUDED_FIELDS = ("relative_path", "class_id", "sha256", "decode_error_type", "decode_error")


class SplitError(ValueError):
    """Raised when D3-R2 inputs, outputs, or split invariants are invalid."""


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    indices: tuple[int, ...]
    class_counts: dict[int, int]
    size: int
    tie_break: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-r1-dir", type=Path, default=DEFAULT_D2_R1)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise SplitError(f"refusing to overwrite a different artifact: {path}")
        return
    path.write_bytes(payload)


def verify_checksum_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        source = REPO_ROOT / name
        if separator != "  " or not source.is_file() or sha256_file(source) != digest:
            raise SplitError(f"checksum mismatch: {name}")


def load_d2_r1_records(d2_r1_dir: Path) -> list[dict[str, Any]]:
    verify_checksum_file(d2_r1_dir / "checksums.sha256")
    records = []
    with (d2_r1_dir / "manifest-hashed.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
    if len(records) != 221396:
        raise SplitError("D2-R1 manifest row count differs")
    seen_paths = set()
    for record in records:
        path = str(record["relative_path"])
        if path in seen_paths:
            raise SplitError(f"D2-R1 path is duplicated: {path}")
        seen_paths.add(path)
        if "phash64" not in record or "d2_r0_duplicate_group_id" not in record:
            raise SplitError("input is not a D2-R1 manifest")
        if not str(record["duplicate_group_id"]).startswith("dg-r1-"):
            raise SplitError("input contains a non-R1 duplicate group ID")
    if sum(record["decode_status"] == "ok" for record in records) != 221377:
        raise SplitError("D2-R1 usable image count differs")
    return records


def stable_tie(group_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{group_id}".encode("ascii")).hexdigest()


def build_groups(records: list[dict[str, Any]]) -> tuple[list[GroupInfo], dict[int, list[str]]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["decode_status"] == "ok":
            members[str(record["duplicate_group_id"])].append(index)
    groups = []
    class_groups: dict[int, list[str]] = defaultdict(list)
    for group_id, indices in members.items():
        counts = Counter(int(records[index]["class_id"]) for index in indices)
        groups.append(
            GroupInfo(
                group_id=group_id,
                indices=tuple(indices),
                class_counts=dict(counts),
                size=len(indices),
                tie_break=stable_tie(group_id),
            )
        )
        for class_id in counts:
            class_groups[class_id].append(group_id)
    groups.sort(key=lambda group: group.group_id)
    for group_ids in class_groups.values():
        group_ids.sort()
    return groups, class_groups


def required_splits(class_id: int, group_count: int) -> tuple[str, ...]:
    if group_count >= 3:
        return SPLITS
    if group_count == 2:
        return ("train", "val" if class_id % 2 == 0 else "test")
    return ("train",)


def assign_groups(
    groups: list[GroupInfo], class_groups: dict[int, list[str]], class_count: int
) -> tuple[dict[str, str], dict[str, Any]]:
    by_id = {group.group_id: group for group in groups}
    required = {
        class_id: required_splits(class_id, len(class_groups.get(class_id, [])))
        for class_id in range(class_count)
    }
    missing = {(class_id, split) for class_id, splits in required.items() for split in splits}
    assignment: dict[str, str] = {}

    while missing:
        class_id, split = min(
            missing,
            key=lambda item: (
                sum(group_id not in assignment for group_id in class_groups[item[0]]),
                len(class_groups[item[0]]),
                SPLITS.index(item[1]),
                item[0],
            ),
        )
        candidates = []
        for group_id in class_groups[class_id]:
            if group_id in assignment:
                continue
            group = by_id[group_id]
            feasible = True
            for member_class in group.class_counts:
                remaining_requirements = sum(
                    (member_class, target_split) in missing and target_split != split
                    for target_split in required[member_class]
                )
                available_after = sum(
                    candidate not in assignment and candidate != group_id
                    for candidate in class_groups[member_class]
                )
                if available_after < remaining_requirements:
                    feasible = False
                    break
            if not feasible:
                continue
            covered_weight = sum(
                1.0 / max(1, len(class_groups[member_class]))
                for member_class in group.class_counts
                if (member_class, split) in missing
            )
            candidates.append((-covered_weight, group.size, group.tie_break, group_id))
        if not candidates:
            raise SplitError(f"cannot reserve group coverage for class_id={class_id}, split={split}")
        group_id = min(candidates)[3]
        assignment[group_id] = split
        for member_class in by_id[group_id].class_counts:
            missing.discard((member_class, split))

    total_samples = sum(group.size for group in groups)
    class_totals: Counter[int] = Counter()
    for group in groups:
        class_totals.update(group.class_counts)
    assigned_totals: Counter[str] = Counter()
    assigned_classes: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    for group_id, split in assignment.items():
        group = by_id[group_id]
        assigned_totals[split] += group.size
        assigned_classes[split].update(group.class_counts)

    remaining = [group for group in groups if group.group_id not in assignment]
    remaining.sort(
        key=lambda group: (
            -group.size,
            min(len(class_groups[class_id]) for class_id in group.class_counts),
            group.tie_break,
        )
    )
    for group in remaining:
        scores = []
        for split in SPLITS:
            global_target = RATIOS[split] * total_samples
            global_need = (global_target - assigned_totals[split]) / max(global_target, 1.0)
            class_need = 0.0
            weight_total = 0
            for class_id, count in group.class_counts.items():
                target = RATIOS[split] * class_totals[class_id]
                need = (target - assigned_classes[split][class_id]) / max(target, 1.0)
                class_need += need * count
                weight_total += count
            class_need /= max(weight_total, 1)
            score = 0.55 * class_need + 0.45 * global_need
            scores.append((-score, SPLITS.index(split), split))
        split = min(scores)[2]
        assignment[group.group_id] = split
        assigned_totals[split] += group.size
        assigned_classes[split].update(group.class_counts)

    coverage = {
        split: {class_id for class_id, count in assigned_classes[split].items() if count > 0}
        for split in SPLITS
    }
    unsatisfied = [
        (class_id, split)
        for class_id, splits in required.items()
        for split in splits
        if class_id not in coverage[split]
    ]
    if unsatisfied:
        raise SplitError(f"required class coverage is missing: {unsatisfied}")
    strategy = {
        "required_split_policy": {
            ">=3_groups": ["train", "val", "test"],
            "2_groups": "train plus val for even class_id or test for odd class_id",
            "1_group": ["train"],
        },
        "rare_group_classes": [
            {"class_id": class_id, "group_count": len(class_groups.get(class_id, []))}
            for class_id in range(class_count)
            if len(class_groups.get(class_id, [])) < 10
        ],
    }
    return assignment, strategy


def csv_payload(rows: list[dict[str, Any]], split: str) -> bytes:
    with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in rows:
            if record["split"] == split:
                writer.writerow({field: record[field] for field in CSV_FIELDS})
        stream.seek(0)
        return stream.read().encode("utf-8")


def build_d3_r2(d2_r1_dir: Path, taxonomy_path: Path, output_dir: Path) -> dict[str, Any]:
    d2_r1_dir = d2_r1_dir.resolve()
    taxonomy_path = taxonomy_path.resolve()
    output_dir = output_dir.resolve()
    records = load_d2_r1_records(d2_r1_dir)
    taxonomy = load_json(taxonomy_path)
    class_count = len(taxonomy["classes"])
    if class_count != 203:
        raise SplitError("taxonomy class count differs")
    groups, class_groups = build_groups(records)
    assignment, rare_strategy = assign_groups(groups, class_groups, class_count)

    usable = []
    excluded = []
    for record in records:
        if record["decode_status"] == "ok":
            usable.append(
                {
                    "relative_path": record["relative_path"],
                    "class_id": record["class_id"],
                    "sha256": record["sha256"],
                    "duplicate_group_id": record["duplicate_group_id"],
                    "split": assignment[record["duplicate_group_id"]],
                }
            )
        else:
            excluded.append(record)
    if len(usable) != 221377 or len(excluded) != 19:
        raise SplitError("usable or excluded image count differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_bytes_idempotent(output_dir / f"{split}.csv", csv_payload(usable, split))
    group_sizes = Counter(record["duplicate_group_id"] for record in usable)
    assignment_lines = ["duplicate_group_id,split,size\n"]
    for group_id in sorted(assignment):
        assignment_lines.append(f"{group_id},{assignment[group_id]},{group_sizes[group_id]}\n")
    write_bytes_idempotent(
        output_dir / "group-assignments.csv", "".join(assignment_lines).encode("utf-8")
    )
    with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXCLUDED_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in excluded:
            writer.writerow({field: record[field] for field in EXCLUDED_FIELDS})
        stream.seek(0)
        write_bytes_idempotent(
            output_dir / "excluded-bad-images.csv", stream.read().encode("utf-8")
        )

    runtime_sources = [Path(__file__).resolve()]
    traceability_sources = [TEST_SOURCE.resolve()]
    for source in runtime_sources + traceability_sources:
        if not source.is_file():
            raise SplitError(f"required source file is missing: {source}")
    config = {
        "schema_version": 3,
        "stage": "D3-R2",
        "data_version": "data-v1-candidate-r2",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "input_stage": "D2-R1",
        "input_manifest": repo_relative(d2_r1_dir / "manifest-hashed.jsonl"),
        "input_manifest_sha256": sha256_file(d2_r1_dir / "manifest-hashed.jsonl"),
        "input_checksums": repo_relative(d2_r1_dir / "checksums.sha256"),
        "input_checksums_sha256": sha256_file(d2_r1_dir / "checksums.sha256"),
        "taxonomy": repo_relative(taxonomy_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "runtime_dependency_policy": "Python standard library only; no repository-local imports",
        "runtime_sources": [
            {"path": repo_relative(source), "sha256": sha256_file(source)} for source in runtime_sources
        ],
        "test_sources": [
            {"path": repo_relative(source), "sha256": sha256_file(source)}
            for source in traceability_sources
        ],
        "seed": SEED,
        "target_ratios": RATIOS,
        "group_field": "duplicate_group_id",
        "required_group_prefix": "dg-r1-",
        "group_integrity_priority": "group leakage must remain zero even when ratios or class balance deviate",
        "ordering": "D2-R1 manifest order within each split",
        "csv_fields": list(CSV_FIELDS),
        "rare_class_strategy": rare_strategy["required_split_policy"],
    }
    write_bytes_idempotent(output_dir / "d3-r2-config.json", canonical_json(config))

    split_counts = Counter(record["split"] for record in usable)
    split_group_counts = {
        split: len({record["duplicate_group_id"] for record in usable if record["split"] == split})
        for split in SPLITS
    }
    per_class = []
    for class_id in range(class_count):
        class_records = [record for record in usable if int(record["class_id"]) == class_id]
        counts = Counter(record["split"] for record in class_records)
        per_class.append(
            {
                "class_id": class_id,
                "total": len(class_records),
                "group_count": len(class_groups[class_id]),
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            }
        )
    summary = {
        "schema_version": 3,
        "stage": "D3-R2",
        "data_version": "data-v1-candidate-r2",
        "source_d2_stage": "D2-R1",
        "source_manifest_sha256": sha256_file(d2_r1_dir / "manifest-hashed.jsonl"),
        "implementation_version": IMPLEMENTATION_VERSION,
        "usable_files": len(usable),
        "excluded_bad_files": len(excluded),
        "split_counts": dict(split_counts),
        "split_ratios": {split: split_counts[split] / len(usable) for split in SPLITS},
        "split_group_counts": split_group_counts,
        "path_overlap_count": 0,
        "duplicate_group_leakage_count": 0,
        "class_coverage": {split: sum(row[split] > 0 for row in per_class) for split in SPLITS},
        "rare_class_strategy": rare_strategy,
        "per_class": per_class,
        "runtime": {"python": platform.python_version()},
    }
    for split in SPLITS:
        summary[f"{split}_sha256"] = sha256_file(output_dir / f"{split}.csv")
    summary["group_assignments_sha256"] = sha256_file(output_dir / "group-assignments.csv")
    summary["excluded_bad_sha256"] = sha256_file(output_dir / "excluded-bad-images.csv")
    summary["config_sha256"] = sha256_file(output_dir / "d3-r2-config.json")
    write_bytes_idempotent(output_dir / "d3-r2-summary.json", canonical_json(summary))
    checksum_paths = [
        d2_r1_dir / "manifest-hashed.jsonl",
        d2_r1_dir / "d2-r1-summary.json",
        d2_r1_dir / "checksums.sha256",
        taxonomy_path,
        *runtime_sources,
        *traceability_sources,
        output_dir / "train.csv",
        output_dir / "val.csv",
        output_dir / "test.csv",
        output_dir / "group-assignments.csv",
        output_dir / "excluded-bad-images.csv",
        output_dir / "d3-r2-config.json",
        output_dir / "d3-r2-summary.json",
    ]
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in sorted(set(checksum_paths))]
    write_bytes_idempotent(
        output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8")
    )
    verify_d3_r2(d2_r1_dir, output_dir, taxonomy_path)
    return summary


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise SplitError(f"split CSV schema differs: {path}")
        return list(reader)


def read_excluded(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXCLUDED_FIELDS:
            raise SplitError("excluded bad-image CSV schema differs")
        return list(reader)


def verify_d3_r2(d2_r1_dir: Path, output_dir: Path, taxonomy_path: Path) -> dict[str, Any]:
    d2_r1_dir = d2_r1_dir.resolve()
    output_dir = output_dir.resolve()
    taxonomy_path = taxonomy_path.resolve()
    records = load_d2_r1_records(d2_r1_dir)
    expected_usable = {
        str(record["relative_path"]): {
            "class_id": str(record["class_id"]),
            "sha256": str(record["sha256"]),
            "duplicate_group_id": str(record["duplicate_group_id"]),
        }
        for record in records
        if record["decode_status"] == "ok"
    }
    expected_bad = {
        str(record["relative_path"]): {
            "class_id": str(record["class_id"]),
            "sha256": str(record["sha256"]),
            "decode_error_type": str(record["decode_error_type"]),
            "decode_error": str(record["decode_error"]),
        }
        for record in records
        if record["decode_status"] == "bad"
    }
    observed_paths: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    class_coverage: dict[str, set[int]] = {split: set() for split in SPLITS}
    group_sizes: Counter[str] = Counter()
    split_counts = {}
    for split in SPLITS:
        rows = read_split_csv(output_dir / f"{split}.csv")
        split_counts[split] = len(rows)
        for row in rows:
            path = row["relative_path"]
            if row["split"] != split or path.startswith("/") or "data/views/" in path:
                raise SplitError(f"invalid split row in {split}")
            if path in observed_paths:
                raise SplitError(f"path appears more than once: {path}")
            expected = expected_usable.get(path)
            if expected is None or any(row[field] != expected[field] for field in expected):
                raise SplitError(f"split row differs from D2-R1: {path}")
            observed_paths[path] = split
            group_splits[row["duplicate_group_id"]].add(split)
            group_sizes[row["duplicate_group_id"]] += 1
            class_coverage[split].add(int(row["class_id"]))
    if set(observed_paths) != set(expected_usable):
        raise SplitError("split path set differs from D2-R1 usable path set")
    group_leakage = sum(len(values) > 1 for values in group_splits.values())
    if group_leakage:
        raise SplitError(f"duplicate group leakage detected: {group_leakage}")

    taxonomy = load_json(taxonomy_path)
    class_count = len(taxonomy["classes"])
    if any(len(class_coverage[split]) != class_count for split in SPLITS):
        raise SplitError("one or more splits do not cover all taxonomy classes")
    excluded = read_excluded(output_dir / "excluded-bad-images.csv")
    if len(excluded) != 19 or {row["relative_path"] for row in excluded} != set(expected_bad):
        raise SplitError("excluded bad-image set differs from D2-R1")
    for row in excluded:
        expected = expected_bad[row["relative_path"]]
        if any(row[field] != expected[field] for field in expected):
            raise SplitError(f"excluded bad-image row differs: {row['relative_path']}")

    with (output_dir / "group-assignments.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("duplicate_group_id", "split", "size"):
            raise SplitError("group assignment CSV schema differs")
        assignment_rows = list(reader)
    if len(assignment_rows) != len(group_splits):
        raise SplitError("group assignment row count differs")
    for row in assignment_rows:
        group_id = row["duplicate_group_id"]
        if group_id not in group_splits or group_splits[group_id] != {row["split"]}:
            raise SplitError(f"group assignment differs: {group_id}")
        if int(row["size"]) != group_sizes[group_id]:
            raise SplitError(f"group assignment size differs: {group_id}")

    config = load_json(output_dir / "d3-r2-config.json")
    if config["runtime_dependency_policy"] != "Python standard library only; no repository-local imports":
        raise SplitError("D3-R2 runtime dependency policy differs")
    expected_runtime_source = repo_relative(Path(__file__).resolve())
    if config["runtime_sources"] != [
        {"path": expected_runtime_source, "sha256": sha256_file(Path(__file__).resolve())}
    ]:
        raise SplitError("D3-R2 runtime source index differs")
    summary = load_json(output_dir / "d3-r2-summary.json")
    if summary["source_d2_stage"] != "D2-R1" or summary["implementation_version"] != IMPLEMENTATION_VERSION:
        raise SplitError("D3-R2 summary source or implementation differs")
    if summary["source_manifest_sha256"] != sha256_file(d2_r1_dir / "manifest-hashed.jsonl"):
        raise SplitError("D3-R2 source manifest SHA-256 differs")
    if summary["split_counts"] != split_counts:
        raise SplitError("D3-R2 summary split counts differ")
    if summary["path_overlap_count"] != 0 or summary["duplicate_group_leakage_count"] != 0:
        raise SplitError("D3-R2 summary leakage fields are not zero")
    if summary["class_coverage"] != {split: class_count for split in SPLITS}:
        raise SplitError("D3-R2 summary class coverage differs")
    verify_checksum_file(output_dir / "checksums.sha256")
    checksum_text = (output_dir / "checksums.sha256").read_text(encoding="utf-8")
    if "scripts/build_splits_d3.py" in checksum_text or "scripts/build_splits_d3_r1.py" in checksum_text:
        raise SplitError("D3-R2 checksum contains a rejected split-script dependency")
    if repo_relative(Path(__file__).resolve()) not in checksum_text or repo_relative(TEST_SOURCE) not in checksum_text:
        raise SplitError("D3-R2 source or test is missing from checksum")
    return {
        "rows": len(observed_paths),
        "split_counts": split_counts,
        "path_overlap_count": 0,
        "duplicate_group_leakage_count": group_leakage,
        "class_coverage": {split: len(values) for split, values in class_coverage.items()},
        "excluded_bad_files": len(excluded),
        "source_d2_stage": "D2-R1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "runtime_source_count": len(config["runtime_sources"]),
    }


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            result = verify_d3_r2(args.d2_r1_dir, args.output_dir, args.taxonomy)
        else:
            result = build_d3_r2(args.d2_r1_dir, args.taxonomy, args.output_dir)
    except (SplitError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

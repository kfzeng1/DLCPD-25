#!/usr/bin/env python3
"""Build the DLCPD-25 Plan-A image manifest and frozen train/val/test split.

The script treats the classification dataset as the source of truth:
- one row per image file in ``data/raw/dlcpd25``;
- sha256 duplicate groups are never split across train/val/test;
- every class with enough independent duplicate groups receives train/val/test
  coverage (two groups receive train+test, one group receives train only);
- images which cannot be decoded by Pillow are excluded from the splits and
  reported separately.

Outputs are written to ``artifacts/data/dlcpd25``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "dlcpd25"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata" / "dlcpd25" / "class-taxonomy.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "data" / "dlcpd25"
DEFAULT_WORKERS = 4

SCHEMA_VERSION = 1
SEED = 20260817
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CSV_FIELDS = (
    "relative_path",
    "class_id",
    "sha256",
    "size_bytes",
    "width",
    "height",
    "decode_status",
    "decode_error",
    "duplicate_group_id",
    "split",
)


class ManifestError(RuntimeError):
    """Raised when the manifest/split contract is violated."""


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    indices: tuple[int, ...]
    class_counts: dict[int, int]
    size: int
    tie_break: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def audit_image(path: Path, class_id: int) -> dict[str, Any]:
    size = path.stat().st_size
    digest = sha256_file(path)
    record: dict[str, Any] = {
        "relative_path": repo_relative(path),
        "class_id": class_id,
        "sha256": digest,
        "size_bytes": size,
        "width": None,
        "height": None,
        "decode_status": "ok",
        "decode_error": "",
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record["width"], record["height"] = image.size
    except Exception as exc:  # noqa: BLE001 - decode audit must report every file
        record["decode_status"] = "bad"
        record["decode_error"] = f"{type(exc).__name__}: {exc}"[:400]
    return record


def collect_files(data_root: Path, taxonomy: dict[str, Any]) -> tuple[list[tuple[Path, int]], dict[int, str]]:
    classes = taxonomy["classes"]
    local_names = {int(item["class_id"]): str(item["local_directory"]) for item in classes}
    files: list[tuple[Path, int]] = []
    for class_id in range(len(classes)):
        class_dir = data_root / local_names[class_id]
        if not class_dir.is_dir():
            raise ManifestError(f"class directory is missing: {class_dir}")
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            files.append((path, class_id))
    return files, local_names


def run_audit(files: list[tuple[Path, int]], workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(audit_image, path, class_id) for path, class_id in files]
        for future in as_completed(futures):
            records.append(future.result())
            completed += 1
            if completed % 20000 == 0:
                print(f"  {completed}/{len(files)}", file=sys.stderr, flush=True)
    records.sort(key=lambda record: (record["class_id"], record["relative_path"]))
    return records


def stable_tie(group_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{group_id}".encode("ascii")).hexdigest()


def required_splits(image_count: int, group_count: int) -> tuple[str, ...]:
    if image_count == 0:
        return ()
    if image_count < 2 or group_count < 2:
        return ("train",)
    if group_count >= 3:
        return SPLITS
    # Two independent duplicate groups can only cover two splits while
    # keeping duplicates together. Prefer train+test so final evaluation
    # still sees the class.
    return ("train", "test")


def build_groups(records: list[dict[str, Any]]) -> tuple[list[GroupInfo], dict[int, list[str]]]:
    usable_indices = [i for i, record in enumerate(records) if record["decode_status"] == "ok"]
    members: dict[str, list[int]] = defaultdict(list)
    for index in usable_indices:
        members[str(records[index]["sha256"])].append(index)

    groups: list[GroupInfo] = []
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


def assign_splits(
    groups: list[GroupInfo],
    class_groups: dict[int, list[str]],
    class_count: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    by_id = {group.group_id: group for group in groups}
    # Rebuild class image totals from group class counts.
    image_totals: Counter[int] = Counter()
    for group in groups:
        image_totals.update(group.class_counts)

    required = {
        class_id: required_splits(image_totals.get(class_id, 0), len(class_groups.get(class_id, [])))
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
        candidates: list[tuple[float, int, str, str]] = []
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
            raise ManifestError(f"cannot reserve coverage for class_id={class_id}, split={split}")
        group_id = min(candidates)[3]
        assignment[group_id] = split
        for member_class in by_id[group_id].class_counts:
            missing.discard((member_class, split))

    # Fill remaining groups while balancing global and per-class ratios.
    total_samples = sum(group.size for group in groups)
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
                target = RATIOS[split] * image_totals[class_id]
                need = (target - assigned_classes[split][class_id]) / max(target, 1.0)
                class_need += need * count
                weight_total += count
            class_need /= max(weight_total, 1)
            scores.append((-0.55 * class_need - 0.45 * global_need, SPLITS.index(split), split))
        split = min(scores)[2]
        assignment[group.group_id] = split
        assigned_totals[split] += group.size
        assigned_classes[split].update(group.class_counts)

    coverage = {
        split: {class_id for class_id, count in assigned_classes[split].items() if count > 0}
        for split in SPLITS
    }
    unsatisfied = sorted(
        (class_id, split)
        for class_id, splits in required.items()
        for split in splits
        if class_id not in coverage[split]
    )
    if unsatisfied:
        raise ManifestError(f"split coverage is unsatisfied: {unsatisfied[:20]}")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "ratios": RATIOS,
        "algorithm": "sha256-duplicate-group-aware-stratified-greedy-v1",
        "counts": dict(assigned_totals),
        "total_usable_images": sum(assigned_totals.values()),
        "duplicate_groups": len(groups),
        "class_coverage": {split: len(coverage[split]) for split in SPLITS},
        "per_class": {
            str(class_id): {
                "total_images": image_totals.get(class_id, 0),
                "train": assigned_classes["train"].get(class_id, 0),
                "val": assigned_classes["val"].get(class_id, 0),
                "test": assigned_classes["test"].get(class_id, 0),
            }
            for class_id in range(class_count)
        },
    }
    return assignment, summary


def write_manifest(records: list[dict[str, Any]], assignment: dict[str, str], output: Path) -> None:
    for record in records:
        group_id = record["sha256"]
        record["duplicate_group_id"] = group_id
        record["split"] = assignment.get(group_id, "") if record["decode_status"] == "ok" else ""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with gzip.open(output / "manifest.csv.gz", "wb", compresslevel=6) as raw:
        raw.write(manifest_path.read_bytes())
    bad_records = [record for record in records if record["decode_status"] != "ok"]
    with (output / "excluded-images.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(bad_records)


def write_summary(records: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    extensions = Counter(Path(record["relative_path"]).suffix.lower() or "<none>" for record in records)
    total_files = len(records)
    bad = [record for record in records if record["decode_status"] != "ok"]
    config = {
        "schema_version": SCHEMA_VERSION,
        "stage": "DLCPD25-PLAN-A-DATA-V1",
        "data_root": repo_relative(DEFAULT_DATA_ROOT),
        "taxonomy_source": repo_relative(DEFAULT_TAXONOMY),
        "output_dir": repo_relative(output),
        "seed": SEED,
        "ratios": RATIOS,
        "runtime": {
            "python": platform.python_version(),
            "pillow": Image.__version__,
            "platform": platform.platform(),
        },
    }
    (output / "split-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "build-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "total_files": total_files,
        "usable_files": total_files - len(bad),
        "bad_files": len(bad),
        "extensions": dict(sorted(extensions.items())),
        "classes": len(summary["per_class"]),
    }
    (output / "audit-summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run-limit", type=int, default=0, help="audit only the first N files (smoke tests)")
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    if len(taxonomy.get("classes", [])) != 203:
        raise ManifestError("taxonomy must contain 203 classes")

    files, _ = collect_files(args.data_root, taxonomy)
    if args.dry_run_limit:
        files = files[: args.dry_run_limit]
    print(f"auditing {len(files)} files with {args.workers} workers...", file=sys.stderr)
    records = run_audit(files, args.workers)
    groups, class_groups = build_groups(records)
    assignment, summary = assign_splits(groups, class_groups, 203)
    write_manifest(records, assignment, args.output)
    write_summary(records, summary, args.output)
    print(
        json.dumps(
            {
                "files": len(records),
                "usable": summary["total_usable_images"],
                "groups": len(groups),
                "counts": summary["counts"],
            },
            ensure_ascii=False,
        ),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild D2 duplicate groups with pHash confirmation and bounded diameter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import __version__ as scipy_version
from scipy.fft import dctn


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_SCRIPT = REPO_ROOT / "scripts" / "build_duplicates_d2.py"
OLD_SPEC = importlib.util.spec_from_file_location("build_duplicates_d2_r0", OLD_SCRIPT)
if OLD_SPEC is None or OLD_SPEC.loader is None:
    raise RuntimeError(f"cannot load D2-R0 helpers: {OLD_SCRIPT}")
D2_R0 = importlib.util.module_from_spec(OLD_SPEC)
OLD_SPEC.loader.exec_module(D2_R0)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "dlcpd25-203"
DEFAULT_D1 = REPO_ROOT / "artifacts" / "data" / "v1" / "d1"
DEFAULT_D2_R0 = REPO_ROOT / "artifacts" / "data" / "v1" / "d2"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "data" / "v1" / "d2-r1"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
DEFAULT_WORKERS = 6
DHASH_THRESHOLD = 5
PHASH_THRESHOLD = 8
PHASH_VERSION = "phash64-exif-gray32x32-dct-v1"
GROUPING_VERSION = "dhash-candidate-phash-confirmed-complete-link-v1"
REJECTED_GROUP_IDS = ("dg-001396", "dg-004019")
AUDIT_SEED = 20260810
AUDIT_GROUPS_PER_STRATUM = 20
AUDIT_GROUPS_PER_PAGE = 5
AUDIT_MEMBERS_PER_GROUP = 8


class RevisionError(ValueError):
    """Raised when D2-R1 inputs, grouping, or quality gates are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1)
    parser.add_argument("--d2-r0-dir", type=Path, default=DEFAULT_D2_R0)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
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


def verify_checksum_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        source = REPO_ROOT / name
        if separator != "  " or not source.is_file() or sha256_file(source) != digest:
            raise RevisionError(f"checksum mismatch: {name}")


def write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RevisionError(f"refusing to overwrite a different artifact: {path}")
        return
    path.write_bytes(payload)


def install_temp_idempotent(path: Path, temporary: Path) -> None:
    if path.exists():
        if not path.is_file() or sha256_file(path) != sha256_file(temporary):
            raise RevisionError(f"refusing to overwrite a different artifact: {path}")
        temporary.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, path)


def phash64(path: Path) -> tuple[int, str]:
    with Image.open(path) as image:
        orientation = image.getexif().get(274)
        oriented, exif_status = D2_R0.apply_exif_orientation(image, orientation)
        pixels = np.asarray(
            oriented.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    low_frequency = dctn(pixels, type=2, norm="ortho")[:8, :8]
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    value = 0
    for bit in (low_frequency > median).reshape(-1):
        value = (value << 1) | int(bit)
    return value, exif_status


def phash_task(task: tuple[Path, dict[str, Any]]) -> tuple[str | None, str]:
    path, record = task
    if record["decode_status"] != "ok":
        return None, "not_applicable"
    value, exif_status = phash64(path)
    return f"{value:016x}", exif_status


def load_inputs(
    data_root: Path, d1_dir: Path, d2_r0_dir: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    verify_checksum_file(d1_dir / "checksums.sha256")
    verify_checksum_file(d2_r0_dir / "checksums.sha256")
    d1_records = D2_R0.load_d1_records(d1_dir)
    records = []
    with (d2_r0_dir / "manifest-hashed.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
    if len(records) != len(d1_records):
        raise RevisionError("D2-R0 and D1 manifest row counts differ")
    paths = []
    for d1_record, record in zip(d1_records, records, strict=True):
        if d1_record["relative_path"] != record["relative_path"]:
            raise RevisionError("D2-R0 path ordering differs from D1")
        if len(str(record["sha256"])) != 64:
            raise RevisionError("D2-R0 SHA-256 field is invalid")
        if (record["decode_status"] == "ok") != (record["dhash64"] is not None):
            raise RevisionError("D2-R0 dHash presence differs from decode status")
        path = D2_R0.safe_source_path(data_root, str(record["relative_path"]))
        if path.stat().st_size != int(record["file_size_bytes"]):
            raise RevisionError(f"source size differs from D1: {record['relative_path']}")
        record["d2_r0_duplicate_group_id"] = record.pop("duplicate_group_id")
        record.pop("duplicate_group_size", None)
        paths.append(path)
    return records, paths


def _complete_link_clusters(
    fingerprint_members: dict[tuple[int, int], list[int]],
) -> tuple[list[list[int]], dict[str, Any]]:
    fingerprints = sorted(fingerprint_members, key=lambda item: min(fingerprint_members[item]))
    fingerprint_id = {fingerprint: index for index, fingerprint in enumerate(fingerprints)}
    by_dhash: dict[int, list[int]] = defaultdict(list)
    for index, (dhash, _phash) in enumerate(fingerprints):
        by_dhash[dhash].append(index)

    adjacency = [set() for _ in fingerprints]
    compatible_edges: list[tuple[int, int, int, int, int, int]] = []
    candidate_fingerprint_pairs = 0
    rejected_by_phash = 0
    direct_dhash_pairs = 0

    def compare_dhash_buckets(left_dhash: int, right_dhash: int, dhash_distance: int) -> None:
        nonlocal candidate_fingerprint_pairs, rejected_by_phash
        left_ids = by_dhash[left_dhash]
        right_ids = by_dhash[right_dhash]
        pairs = combinations(left_ids, 2) if left_dhash == right_dhash else (
            (left, right) for left in left_ids for right in right_ids
        )
        for left, right in pairs:
            candidate_fingerprint_pairs += 1
            phash_distance = (fingerprints[left][1] ^ fingerprints[right][1]).bit_count()
            if phash_distance > PHASH_THRESHOLD:
                rejected_by_phash += 1
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
            earliest_left = min(fingerprint_members[fingerprints[left]])
            earliest_right = min(fingerprint_members[fingerprints[right]])
            compatible_edges.append(
                (
                    dhash_distance,
                    phash_distance,
                    min(earliest_left, earliest_right),
                    max(earliest_left, earliest_right),
                    left,
                    right,
                )
            )

    for dhash in sorted(by_dhash):
        if len(by_dhash[dhash]) > 1:
            compare_dhash_buckets(dhash, dhash, 0)

    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_dhash_comparisons = 0
    for dhash in sorted(by_dhash):
        candidates: set[int] = set()
        keys = D2_R0.band_keys(dhash)
        for key in keys:
            candidates.update(band_index[key])
        for other in sorted(candidates):
            candidate_dhash_comparisons += 1
            distance = (dhash ^ other).bit_count()
            if distance <= DHASH_THRESHOLD:
                direct_dhash_pairs += 1
                compare_dhash_buckets(other, dhash, distance)
        for key in keys:
            band_index[key].append(dhash)

    parent = list(range(len(fingerprints)))
    cluster_nodes = [{index} for index in range(len(fingerprints))]

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    rejected_complete_link_merges = 0
    accepted_complete_link_merges = 0
    for _dd, _pd, _first, _second, left, right in sorted(compatible_edges):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        left_nodes = cluster_nodes[left_root]
        right_nodes = cluster_nodes[right_root]
        if not all(right_node in adjacency[left_node] for left_node in left_nodes for right_node in right_nodes):
            rejected_complete_link_merges += 1
            continue
        left_earliest = min(min(fingerprint_members[fingerprints[node]]) for node in left_nodes)
        right_earliest = min(min(fingerprint_members[fingerprints[node]]) for node in right_nodes)
        if right_earliest < left_earliest:
            left_root, right_root = right_root, left_root
            left_nodes, right_nodes = right_nodes, left_nodes
        parent[right_root] = left_root
        cluster_nodes[left_root] = left_nodes | right_nodes
        cluster_nodes[right_root] = set()
        accepted_complete_link_merges += 1

    clusters = []
    for root, nodes in enumerate(cluster_nodes):
        if not nodes or find(root) != root:
            continue
        members = []
        for node in nodes:
            members.extend(fingerprint_members[fingerprints[node]])
        clusters.append(sorted(members))
    stats = {
        "unique_fingerprints": len(fingerprints),
        "candidate_dhash_comparisons": candidate_dhash_comparisons,
        "direct_dhash_value_pairs": direct_dhash_pairs,
        "candidate_fingerprint_pairs": candidate_fingerprint_pairs,
        "phash_confirmed_fingerprint_edges": len(compatible_edges),
        "phash_rejected_fingerprint_pairs": rejected_by_phash,
        "complete_link_accepted_merges": accepted_complete_link_merges,
        "complete_link_rejected_merges": rejected_complete_link_merges,
    }
    return clusters, stats


def group_distances(members: list[int], records: list[dict[str, Any]]) -> tuple[int, int]:
    fingerprints = sorted(
        {
            (int(str(records[index]["dhash64"]), 16), int(str(records[index]["phash64"]), 16))
            for index in members
            if records[index]["dhash64"] is not None
        }
    )
    max_dhash = 0
    max_phash = 0
    for left, right in combinations(fingerprints, 2):
        max_dhash = max(max_dhash, (left[0] ^ right[0]).bit_count())
        max_phash = max(max_phash, (left[1] ^ right[1]).bit_count())
    return max_dhash, max_phash


def assign_duplicate_groups(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any], dict[str, list[int]]]:
    sha_members: dict[str, list[int]] = defaultdict(list)
    fingerprint_members: dict[tuple[int, int], list[int]] = defaultdict(list)
    bad_sha_members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        sha = str(record["sha256"])
        sha_members[sha].append(index)
        if record["dhash64"] is None:
            if record.get("phash64") is not None:
                raise RevisionError("bad image unexpectedly has pHash")
            bad_sha_members[sha].append(index)
        else:
            if record.get("phash64") is None:
                raise RevisionError("decodable image is missing pHash")
            fingerprint_members[
                (int(str(record["dhash64"]), 16), int(str(record["phash64"]), 16))
            ].append(index)
    for sha, members in sha_members.items():
        fingerprints = {
            (records[index]["dhash64"], records[index].get("phash64")) for index in members
        }
        if len(fingerprints) != 1:
            raise RevisionError(f"exact SHA-256 records have different perceptual hashes: {sha}")

    decoded_components, grouping_stats = _complete_link_clusters(fingerprint_members)
    components = decoded_components + [sorted(members) for members in bad_sha_members.values()]
    components.sort(key=min)
    group_ids = [""] * len(records)
    component_by_id: dict[str, list[int]] = {}
    max_dhash_diameter = 0
    max_phash_diameter = 0
    for sequence, members in enumerate(components, start=1):
        group_id = f"dg-r1-{sequence:06d}"
        component_by_id[group_id] = members
        dhash_diameter, phash_diameter = group_distances(members, records)
        if dhash_diameter > DHASH_THRESHOLD or phash_diameter > PHASH_THRESHOLD:
            raise RevisionError(
                f"group diameter exceeds threshold: {group_id} d={dhash_diameter} p={phash_diameter}"
            )
        max_dhash_diameter = max(max_dhash_diameter, dhash_diameter)
        max_phash_diameter = max(max_phash_diameter, phash_diameter)
        for member in members:
            if group_ids[member]:
                raise RevisionError("record assigned to more than one duplicate group")
            group_ids[member] = group_id
    if not all(group_ids):
        raise RevisionError("record missing duplicate group")

    exact_groups = [members for members in sha_members.values() if len(members) > 1]
    duplicate_components = [members for members in components if len(members) > 1]
    near_components = [
        members
        for members in duplicate_components
        if len({str(records[index]["sha256"]) for index in members}) > 1
    ]
    cross_class_components = [
        members
        for members in duplicate_components
        if len({int(records[index]["class_id"]) for index in members}) > 1
    ]
    size_histogram = Counter(len(members) for members in components)
    stats = {
        **grouping_stats,
        "total_groups": len(components),
        "singleton_groups": sum(len(members) == 1 for members in components),
        "duplicate_groups": len(duplicate_components),
        "files_in_duplicate_groups": sum(len(members) for members in duplicate_components),
        "exact_sha256_groups": len(exact_groups),
        "files_in_exact_sha256_groups": sum(len(members) for members in exact_groups),
        "near_duplicate_groups": len(near_components),
        "cross_class_duplicate_groups": len(cross_class_components),
        "largest_group_size": max(size_histogram),
        "group_size_histogram": {str(key): value for key, value in sorted(size_histogram.items())},
        "maximum_group_dhash_diameter": max_dhash_diameter,
        "maximum_group_phash_diameter": max_phash_diameter,
    }
    return group_ids, stats, component_by_id


def diverse_members(members: list[int], records: list[dict[str, Any]], limit: int) -> list[int]:
    selected = []
    seen_classes = set()
    for index in members:
        class_id = int(records[index]["class_id"])
        if class_id not in seen_classes:
            selected.append(index)
            seen_classes.add(class_id)
        if len(selected) == limit:
            return selected
    for index in members:
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            break
    return selected


def group_record(
    group_id: str,
    members: list[int],
    records: list[dict[str, Any]],
    member_limit: int | None = None,
) -> dict[str, Any]:
    selected = members if member_limit is None else diverse_members(members, records, member_limit)
    max_dhash, max_phash = group_distances(members, records)
    return {
        "duplicate_group_id": group_id,
        "size": len(members),
        "class_ids": sorted({int(records[index]["class_id"]) for index in members}),
        "sha256_count": len({str(records[index]["sha256"]) for index in members}),
        "fingerprint_count": len(
            {
                (records[index]["dhash64"], records[index].get("phash64"))
                for index in members
                if records[index]["dhash64"] is not None
            }
        ),
        "bad_image_count": sum(records[index]["decode_status"] == "bad" for index in members),
        "maximum_dhash_distance": max_dhash,
        "maximum_phash_distance": max_phash,
        "members": [
            {
                "relative_path": records[index]["relative_path"],
                "class_id": records[index]["class_id"],
                "sha256": records[index]["sha256"],
                "dhash64": records[index]["dhash64"],
                "phash64": records[index].get("phash64"),
            }
            for index in selected
        ],
        "members_truncated": member_limit is not None and len(members) > member_limit,
    }


def build_regression_report(
    records: list[dict[str, Any]], group_ids: list[str], components: dict[str, list[int]]
) -> dict[str, Any]:
    component_metrics = {
        group_id: group_distances(members, records) for group_id, members in components.items()
    }
    regressions = []
    for rejected_group_id in REJECTED_GROUP_IDS:
        members = [
            index
            for index, record in enumerate(records)
            if record["d2_r0_duplicate_group_id"] == rejected_group_id
        ]
        if not members:
            raise RevisionError(f"rejected regression group is missing: {rejected_group_id}")
        replacement_ids = sorted({group_ids[index] for index in members})
        if len(replacement_ids) <= 1:
            raise RevisionError(f"rejected group was not split: {rejected_group_id}")
        regressions.append(
            {
                "d2_r0_duplicate_group_id": rejected_group_id,
                "old_size": len(members),
                "old_class_count": len({int(records[index]["class_id"]) for index in members}),
                "replacement_group_count": len(replacement_ids),
                "largest_replacement_group_size": max(
                    sum(group_ids[index] == group_id for index in members) for group_id in replacement_ids
                ),
                "maximum_replacement_dhash_diameter": max(
                    component_metrics[group_id][0] for group_id in replacement_ids
                ),
                "maximum_replacement_phash_diameter": max(
                    component_metrics[group_id][1] for group_id in replacement_ids
                ),
            }
        )
    return {
        "schema_version": 1,
        "stage": "D2-R1",
        "result": "all_rejected_groups_split_with_bounded_diameter",
        "groups": regressions,
    }


def select_audit_groups(
    components: dict[str, list[int]], records: list[dict[str, Any]]
) -> dict[str, list[tuple[str, list[int]]]]:
    duplicate_groups = [(group_id, members) for group_id, members in components.items() if len(members) > 1]
    largest = sorted(duplicate_groups, key=lambda item: (-len(item[1]), item[0]))[:AUDIT_GROUPS_PER_STRATUM]
    cross_class = sorted(
        (
            item
            for item in duplicate_groups
            if len({int(records[index]["class_id"]) for index in item[1]}) > 1
        ),
        key=lambda item: (
            -len({int(records[index]["class_id"]) for index in item[1]}),
            -len(item[1]),
            item[0],
        ),
    )[:AUDIT_GROUPS_PER_STRATUM]
    generator = random.Random(AUDIT_SEED)
    random_groups = generator.sample(
        sorted(duplicate_groups), min(AUDIT_GROUPS_PER_STRATUM, len(duplicate_groups))
    )
    return {"largest": largest, "cross-class": cross_class, "random": random_groups}


def render_audit_pages(
    output_dir: Path,
    data_root: Path,
    strata: dict[str, list[tuple[str, list[int]]]],
    records: list[dict[str, Any]],
) -> list[Path]:
    output_paths = []
    label_width = 240
    thumb_width = 150
    row_height = 150
    for stratum, groups in strata.items():
        for page_number, start in enumerate(range(0, len(groups), AUDIT_GROUPS_PER_PAGE), start=1):
            page_groups = groups[start : start + AUDIT_GROUPS_PER_PAGE]
            canvas = Image.new(
                "RGB",
                (label_width + AUDIT_MEMBERS_PER_GROUP * thumb_width, len(page_groups) * row_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for row, (group_id, members) in enumerate(page_groups):
                y = row * row_height
                class_count = len({int(records[index]["class_id"]) for index in members})
                max_dhash, max_phash = group_distances(members, records)
                draw.text(
                    (8, y + 8),
                    f"{group_id}\nn={len(members)} classes={class_count}\nd={max_dhash} p={max_phash}",
                    fill="black",
                )
                selected = diverse_members(members, records, AUDIT_MEMBERS_PER_GROUP)
                for column, index in enumerate(selected):
                    x = label_width + column * thumb_width
                    try:
                        with Image.open(data_root / records[index]["relative_path"]) as image:
                            orientation = image.getexif().get(274)
                            oriented, _status = D2_R0.apply_exif_orientation(image, orientation)
                            thumb = ImageOps.fit(
                                oriented.convert("RGB"),
                                (thumb_width - 8, row_height - 28),
                                method=Image.Resampling.LANCZOS,
                            )
                        canvas.paste(thumb, (x + 4, y + 22))
                    except OSError:
                        draw.rectangle((x + 4, y + 22, x + thumb_width - 4, y + row_height - 6), fill="#dddddd")
                    draw.text((x + 4, y + 4), f"c{records[index]['class_id']}", fill="black")
            output_path = output_dir / f"audit-{stratum}-page-{page_number:02d}.jpg"
            fd, temporary_name = tempfile.mkstemp(dir=output_dir, prefix=".audit-", suffix=".jpg")
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                canvas.save(temporary, format="JPEG", quality=90, subsampling=0, optimize=False)
                install_temp_idempotent(output_path, temporary)
            finally:
                temporary.unlink(missing_ok=True)
            output_paths.append(output_path)
    return output_paths


def build_d2_r1(
    data_root: Path,
    d1_dir: Path,
    d2_r0_dir: Path,
    taxonomy_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise RevisionError("workers must be at least 1")
    data_root = data_root.resolve()
    d1_dir = d1_dir.resolve()
    d2_r0_dir = d2_r0_dir.resolve()
    taxonomy_path = taxonomy_path.resolve()
    output_dir = output_dir.resolve()
    records, paths = load_inputs(data_root, d1_dir, d2_r0_dir)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(phash_task, zip(paths, records, strict=True))
        for record, (phash, exif_status) in zip(records, results, strict=True):
            record["phash64"] = phash
            record["phash_exif_status"] = exif_status
            if exif_status != record["dhash_exif_status"]:
                raise RevisionError(f"pHash/dHash EXIF handling differs: {record['relative_path']}")

    group_ids, group_stats, components = assign_duplicate_groups(records)
    group_sizes = {group_id: len(members) for group_id, members in components.items()}
    for record, group_id in zip(records, group_ids, strict=True):
        record["duplicate_group_id"] = group_id
        record["duplicate_group_size"] = group_sizes[group_id]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_fd, manifest_name = tempfile.mkstemp(dir=output_dir, prefix=".manifest-")
    groups_fd, groups_name = tempfile.mkstemp(dir=output_dir, prefix=".groups-")
    os.close(manifest_fd)
    os.close(groups_fd)
    manifest_temp = Path(manifest_name)
    groups_temp = Path(groups_name)
    try:
        with manifest_temp.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        with groups_temp.open("w", encoding="utf-8", newline="\n") as stream:
            for group_id, members in components.items():
                if len(members) > 1:
                    stream.write(
                        json.dumps(
                            group_record(group_id, members, records),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        install_temp_idempotent(output_dir / "manifest-hashed.jsonl", manifest_temp)
        install_temp_idempotent(output_dir / "duplicate-groups.jsonl", groups_temp)
    finally:
        manifest_temp.unlink(missing_ok=True)
        groups_temp.unlink(missing_ok=True)

    regression = build_regression_report(records, group_ids, components)
    write_bytes_idempotent(output_dir / "rejected-groups-regression.json", canonical_json(regression))
    strata = select_audit_groups(components, records)
    audit_samples = {
        "schema_version": 1,
        "stage": "D2-R1",
        "seed": AUDIT_SEED,
        "groups_per_stratum": AUDIT_GROUPS_PER_STRATUM,
        "member_limit_per_group": AUDIT_MEMBERS_PER_GROUP,
        "strata": {
            name: [group_record(group_id, members, records, AUDIT_MEMBERS_PER_GROUP) for group_id, members in groups]
            for name, groups in strata.items()
        },
    }
    write_bytes_idempotent(output_dir / "audit-samples.json", canonical_json(audit_samples))
    audit_pages = render_audit_pages(output_dir, data_root, strata, records)
    config = {
        "schema_version": 2,
        "stage": "D2-R1",
        "data_version": "data-v1-candidate-r1",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "data_root": repo_relative(data_root),
        "input_d1_manifest": repo_relative(d1_dir / "manifest.jsonl"),
        "input_d2_r0_manifest": repo_relative(d2_r0_dir / "manifest-hashed.jsonl"),
        "input_d2_r0_manifest_sha256": sha256_file(d2_r0_dir / "manifest-hashed.jsonl"),
        "taxonomy": repo_relative(taxonomy_path),
        "sha256": {"algorithm": "SHA-256", "source": "verified D2-R0 manifest", "scope": "all files"},
        "dhash": {
            "version": D2_R0.DHASH_VERSION,
            "source": "verified D2-R0 manifest",
            "role": "candidate recall only",
            "threshold_inclusive": DHASH_THRESHOLD,
            "candidate_index": "six exact bands with widths 11,11,11,11,10,10",
        },
        "phash": {
            "version": PHASH_VERSION,
            "scope": "D1 decode_status=ok only",
            "frame": "first frame",
            "exif_orientation": "same deterministic rule as dHash",
            "color": "grayscale",
            "resize": [32, 32],
            "resampling": "Pillow LANCZOS",
            "transform": "SciPy orthonormal DCT-II",
            "low_frequency_block": [8, 8],
            "comparison": "coefficient > median of 63 AC-inclusive positions excluding DC from median",
            "bits": 64,
            "threshold_inclusive": PHASH_THRESHOLD,
        },
        "near_duplicate": {
            "version": GROUPING_VERSION,
            "pair_confirmation": "dHash <= 5 AND pHash <= 8",
            "group_rule": "deterministic greedy complete-link clique partition over confirmed fingerprint edges",
            "diameter_gate": "all decodable member pairs satisfy both thresholds",
            "exact_sha256_rule": "identical SHA-256 remains in one group, including bad images",
            "group_id_rule": "dg-r1-###### ordered by earliest D1 manifest row",
            "unbounded_transitive_closure": False,
        },
        "audit": {
            "strata": ["largest", "cross-class", "random"],
            "groups_per_stratum": AUDIT_GROUPS_PER_STRATUM,
            "random_seed": AUDIT_SEED,
        },
        "workers": workers,
    }
    write_bytes_idempotent(output_dir / "d2-r1-config.json", canonical_json(config))
    summary = {
        "schema_version": 2,
        "stage": "D2-R1",
        "data_version": "data-v1-candidate-r1",
        "total_files": len(records),
        "sha256_files": sum(record["sha256"] is not None for record in records),
        "dhash_files": sum(record["dhash64"] is not None for record in records),
        "phash_files": sum(record["phash64"] is not None for record in records),
        "bad_files_without_perceptual_hashes": sum(
            record["decode_status"] == "bad" and record["dhash64"] is None and record["phash64"] is None
            for record in records
        ),
        "phash_exif_status_counts": dict(
            sorted(Counter(str(record["phash_exif_status"]) for record in records).items())
        ),
        **group_stats,
        "regression_groups_split": len(regression["groups"]),
        "audit_group_counts": {name: len(groups) for name, groups in strata.items()},
        "manifest_sha256": sha256_file(output_dir / "manifest-hashed.jsonl"),
        "duplicate_groups_sha256": sha256_file(output_dir / "duplicate-groups.jsonl"),
        "config_sha256": sha256_file(output_dir / "d2-r1-config.json"),
        "regression_sha256": sha256_file(output_dir / "rejected-groups-regression.json"),
        "audit_samples_sha256": sha256_file(output_dir / "audit-samples.json"),
        "audit_page_sha256": {path.name: sha256_file(path) for path in audit_pages},
        "runtime": {
            "python": platform.python_version(),
            "pillow": Image.__version__,
            "numpy": np.__version__,
            "scipy": scipy_version,
        },
    }
    write_bytes_idempotent(output_dir / "d2-r1-summary.json", canonical_json(summary))
    checksum_paths = [
        d1_dir / "manifest.jsonl",
        d1_dir / "d1-summary.json",
        d1_dir / "checksums.sha256",
        d2_r0_dir / "manifest-hashed.jsonl",
        d2_r0_dir / "checksums.sha256",
        taxonomy_path,
        Path(__file__).resolve(),
        output_dir / "manifest-hashed.jsonl",
        output_dir / "duplicate-groups.jsonl",
        output_dir / "audit-samples.json",
        output_dir / "rejected-groups-regression.json",
        output_dir / "d2-r1-config.json",
        output_dir / "d2-r1-summary.json",
        *audit_pages,
    ]
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in sorted(set(checksum_paths))]
    write_bytes_idempotent(output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    verify_d2_r1(d1_dir, d2_r0_dir, output_dir)
    return summary


def verify_d2_r1(d1_dir: Path, d2_r0_dir: Path, output_dir: Path) -> dict[str, Any]:
    verify_checksum_file(d1_dir / "checksums.sha256")
    verify_checksum_file(d2_r0_dir / "checksums.sha256")
    d1_records = D2_R0.load_d1_records(d1_dir)
    records = []
    with (output_dir / "manifest-hashed.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
    if len(records) != len(d1_records):
        raise RevisionError("D2-R1 manifest row count differs from D1")
    seen = set()
    sha_groups: dict[str, set[str]] = defaultdict(set)
    for d1_record, record in zip(d1_records, records, strict=True):
        path = record["relative_path"]
        if path != d1_record["relative_path"] or path in seen:
            raise RevisionError("D2-R1 paths differ from D1 or are duplicated")
        seen.add(path)
        if len(str(record["sha256"])) != 64:
            raise RevisionError("invalid SHA-256 field")
        expected_perceptual = record["decode_status"] == "ok"
        if expected_perceptual != (record["dhash64"] is not None and record["phash64"] is not None):
            raise RevisionError("perceptual hash presence differs from decode status")
        if record["dhash64"] is not None and (
            len(str(record["dhash64"])) != 16 or len(str(record["phash64"])) != 16
        ):
            raise RevisionError("invalid perceptual hash field")
        sha_groups[str(record["sha256"])].add(str(record["duplicate_group_id"]))
    if any(len(group_ids) != 1 for group_ids in sha_groups.values()):
        raise RevisionError("an exact SHA-256 group crosses duplicate groups")
    rebuilt_ids, rebuilt_stats, rebuilt_components = assign_duplicate_groups(records)
    if rebuilt_ids != [record["duplicate_group_id"] for record in records]:
        raise RevisionError("stored D2-R1 group IDs are not reproducible")
    if rebuilt_stats["maximum_group_dhash_diameter"] > DHASH_THRESHOLD:
        raise RevisionError("stored group exceeds dHash diameter")
    if rebuilt_stats["maximum_group_phash_diameter"] > PHASH_THRESHOLD:
        raise RevisionError("stored group exceeds pHash diameter")
    regression = build_regression_report(records, rebuilt_ids, rebuilt_components)
    stored_regression = load_json(output_dir / "rejected-groups-regression.json")
    if regression != stored_regression:
        raise RevisionError("rejected-group regression report differs")
    summary = load_json(output_dir / "d2-r1-summary.json")
    for key in (
        "total_groups",
        "singleton_groups",
        "duplicate_groups",
        "exact_sha256_groups",
        "near_duplicate_groups",
        "cross_class_duplicate_groups",
        "maximum_group_dhash_diameter",
        "maximum_group_phash_diameter",
    ):
        if summary[key] != rebuilt_stats[key]:
            raise RevisionError(f"D2-R1 summary differs for {key}")
    if summary["total_files"] != 221396 or summary["dhash_files"] != 221377 or summary["phash_files"] != 221377:
        raise RevisionError("D2-R1 file totals differ")
    verify_checksum_file(output_dir / "checksums.sha256")
    return {
        "rows": len(records),
        "sha256_files": len(records),
        "dhash_files": sum(record["dhash64"] is not None for record in records),
        "phash_files": sum(record["phash64"] is not None for record in records),
        "duplicate_groups": rebuilt_stats["duplicate_groups"],
        "cross_class_duplicate_groups": rebuilt_stats["cross_class_duplicate_groups"],
        "maximum_group_dhash_diameter": rebuilt_stats["maximum_group_dhash_diameter"],
        "maximum_group_phash_diameter": rebuilt_stats["maximum_group_phash_diameter"],
        "regression_groups_split": len(regression["groups"]),
    }


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            result = verify_d2_r1(
                args.d1_dir.resolve(), args.d2_r0_dir.resolve(), args.output_dir.resolve()
            )
        else:
            result = build_d2_r1(
                args.data_root,
                args.d1_dir,
                args.d2_r0_dir,
                args.taxonomy,
                args.output_dir,
                args.workers,
            )
    except (RevisionError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

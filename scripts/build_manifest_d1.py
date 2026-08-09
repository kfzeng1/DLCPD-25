#!/usr/bin/env python3
"""Build the D1 image manifest and run a full Pillow decode audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from posixpath import normpath
from typing import Any, Iterator

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D0_REPORT = REPO_ROOT / "artifacts" / "data" / "v1" / "d0" / "d0-freeze.json"
DEFAULT_D0_CHECKSUMS = REPO_ROOT / "artifacts" / "data" / "v1" / "d0" / "checksums.sha256"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
DEFAULT_ALIASES = REPO_ROOT / "metadata" / "class-directory-aliases.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "dlcpd25-203"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "data" / "v1" / "d1"
DEFAULT_WORKERS = 6

MANIFEST_SCHEMA_VERSION = 1
SAMPLE_SEED = 20260809
SAMPLE_COUNT_PER_CLASS = 1
FORMAT_EXTENSIONS = {
    "BMP": {".bmp"},
    "GIF": {".gif"},
    "JPEG": {".jfif", ".jpeg", ".jpg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
}


class ManifestError(ValueError):
    """Raised when the D1 input contract or generated records are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--d0-report", type=Path, default=DEFAULT_D0_REPORT)
    parser.add_argument("--d0-checksums", type=Path, default=DEFAULT_D0_CHECKSUMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="parallel Pillow decoder workers (default: 6)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing D1 artifacts without decoding images again",
    )
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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON: {path}: {exc}") from exc


def require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{field} must be a JSON object")
    return value


def verify_d0_inputs(
    d0_report_path: Path,
    d0_checksums_path: Path,
    taxonomy_path: Path,
    aliases_path: Path,
    data_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    d0_report = require_mapping(load_json(d0_report_path), "d0 report")
    if d0_report.get("freeze_id") != "d0-taxonomy-v1":
        raise ManifestError("D0 report is not d0-taxonomy-v1")
    if d0_report.get("target_data_version") != "data-v1":
        raise ManifestError("D0 report is not for data-v1")
    if d0_report.get("data_root", {}).get("relative_path") != repo_relative(data_root):
        raise ManifestError("D0 report data root does not match D1 data root")
    if d0_report.get("class_count") != 203 or d0_report.get("host_count") != 22:
        raise ManifestError("D0 report class or host count is invalid")
    if d0_report.get("data_root", {}).get("total_files") != 221396:
        raise ManifestError("D0 report file count is invalid")
    for path in (d0_report_path, d0_checksums_path, taxonomy_path, aliases_path, data_root):
        if not path.exists():
            raise ManifestError(f"D1 input does not exist: {path}")

    # Verify the D0 checksum list before using its taxonomy contract. The list
    # deliberately excludes itself, so this check is stable after generation.
    checksum_lines = d0_checksums_path.read_text(encoding="utf-8").splitlines()
    if not checksum_lines:
        raise ManifestError("D0 checksum list is empty")
    for line in checksum_lines:
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise ManifestError(f"invalid D0 checksum line: {line!r}")
        source = REPO_ROOT / name
        if not source.is_file() or sha256_file(source) != digest:
            raise ManifestError(f"D0 checksum mismatch: {name}")

    taxonomy = require_mapping(load_json(taxonomy_path), "taxonomy")
    aliases = require_mapping(load_json(aliases_path), "aliases")
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or len(classes) != 203:
        raise ManifestError("taxonomy must contain 203 classes")
    if [item.get("class_id") for item in classes] != list(range(203)):
        raise ManifestError("taxonomy class IDs are not fixed at 0-202")
    if set(aliases) != {item.get("official_name") for item in classes}:
        raise ManifestError("aliases do not cover the taxonomy classes")
    return d0_report, taxonomy, classes


def iter_class_files(data_root: Path, record: dict[str, Any]) -> Iterator[Path]:
    class_dir = data_root / str(record["local_directory"])
    if not class_dir.is_dir() or class_dir.is_symlink():
        raise ManifestError(f"class directory is missing or symlinked: {class_dir}")
    files = []
    for path in class_dir.rglob("*"):
        if path.is_symlink():
            raise ManifestError(f"raw data contains an unexpected symlink: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ManifestError(f"raw data contains a special entry: {path}")
    for path in sorted(files, key=lambda item: item.relative_to(data_root).as_posix()):
        yield path


def decode_image(path: Path) -> dict[str, Any]:
    """Verify and fully load all frames without changing the source file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                frame_count = int(getattr(image, "n_frames", 1))
                image.verify()
            with Image.open(path) as image:
                for frame_index in range(frame_count):
                    if frame_index:
                        image.seek(frame_index)
                    image.load()
                mode = image.mode
                channels = len(image.getbands())
        if width <= 0 or height <= 0:
            raise ValueError("image has non-positive dimensions")
        return {
            "decode_status": "ok",
            "decode_error_type": None,
            "decode_error": None,
            "format": image_format,
            "width": width,
            "height": height,
            "mode": mode,
            "channels": channels,
            "frame_count": frame_count,
        }
    except Exception as exc:  # Pillow exposes multiple decoder-specific errors.
        raw_message = str(exc).replace(str(path), "<source>")
        message = " ".join(raw_message.split())[:500] or "decoder returned no message"
        return {
            "decode_status": "bad",
            "decode_error_type": type(exc).__name__,
            "decode_error": message,
            "format": None,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "frame_count": None,
        }


def make_record(
    path: Path,
    data_root: Path,
    class_record: dict[str, Any],
    decoded: dict[str, Any],
) -> dict[str, Any]:
    relative_path = path.relative_to(data_root).as_posix()
    record = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "relative_path": relative_path,
        "class_id": class_record["class_id"],
        "official_name": class_record["official_name"],
        "local_directory": class_record["local_directory"],
        "host_group": class_record["host_group"],
        "host_id": class_record["host_id"],
        "host_zh": class_record["host_zh"],
        "category": class_record["category"],
        "category_zh": class_record["category_zh"],
        "extension": path.suffix.lower() or "<none>",
        "file_size_bytes": path.stat().st_size,
    }
    record.update(decoded)
    return record


def write_temp_idempotent(output_path: Path, payload_path: Path) -> None:
    if output_path.exists():
        if not output_path.is_file() or sha256_file(output_path) != sha256_file(payload_path):
            raise ManifestError(f"refusing to overwrite a different artifact: {output_path}")
        payload_path.unlink()
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(payload_path, output_path)


def write_bytes_idempotent(output_path: Path, payload: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".d1-", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
    try:
        write_temp_idempotent(output_path, temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_manifest_files(
    output_dir: Path,
    data_root: Path,
    classes: list[dict[str, Any]],
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if workers < 1:
        raise ManifestError("workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_fd, manifest_name = tempfile.mkstemp(dir=output_dir, prefix=".manifest-")
    bad_fd, bad_name = tempfile.mkstemp(dir=output_dir, prefix=".bad-")
    os.close(manifest_fd)
    os.close(bad_fd)
    manifest_temp = Path(manifest_name)
    bad_temp = Path(bad_name)
    class_stats: list[dict[str, Any]] = []
    successful_paths: dict[int, list[str]] = {}
    seen_paths: set[str] = set()
    extension_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    extension_format_mismatches = 0
    total_files = 0
    bad_files = 0
    width_values: list[int] = []
    height_values: list[int] = []
    try:
        with manifest_temp.open("w", encoding="utf-8", newline="\n") as manifest_stream, bad_temp.open(
            "w", encoding="utf-8", newline="\n"
        ) as bad_stream:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for class_record in classes:
                    stats = {
                        "class_id": class_record["class_id"],
                        "official_name": class_record["official_name"],
                        "local_directory": class_record["local_directory"],
                        "total_files": 0,
                        "ok_files": 0,
                        "bad_files": 0,
                        "formats": Counter(),
                    }
                    successful_paths[int(class_record["class_id"])] = []
                    paths = list(iter_class_files(data_root, class_record))
                    for path, decoded in zip(paths, executor.map(decode_image, paths), strict=True):
                        record = make_record(path, data_root, class_record, decoded)
                        relative_path = record["relative_path"]
                        if (
                            not isinstance(relative_path, str)
                            or relative_path.startswith("/")
                            or normpath(relative_path) != relative_path
                            or relative_path.startswith("../")
                            or relative_path.startswith("data/views/")
                            or relative_path in seen_paths
                        ):
                            raise ManifestError(f"manifest path is invalid or duplicated: {relative_path}")
                        seen_paths.add(relative_path)
                        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        manifest_stream.write(line + "\n")
                        stats["total_files"] += 1
                        total_files += 1
                        extension_counts[record["extension"]] += 1
                        if record["decode_status"] == "ok":
                            stats["ok_files"] += 1
                            successful_paths[int(class_record["class_id"])].append(relative_path)
                            stats["formats"][str(record["format"])] += 1
                            format_counts[str(record["format"])] += 1
                            mode_counts[str(record["mode"])] += 1
                            channel_counts[str(record["channels"])] += 1
                            accepted_extensions = FORMAT_EXTENSIONS.get(str(record["format"]), set())
                            if record["extension"] not in accepted_extensions:
                                extension_format_mismatches += 1
                            width_values.append(int(record["width"]))
                            height_values.append(int(record["height"]))
                        else:
                            stats["bad_files"] += 1
                            bad_files += 1
                            error_type_counts[str(record["decode_error_type"])] += 1
                            bad_stream.write(line + "\n")
                    stats["formats"] = dict(sorted(stats["formats"].items()))
                    if stats["total_files"] != int(class_record["image_count"]):
                        raise ManifestError(
                            f"manifest count differs for class_id={class_record['class_id']}: "
                            f"expected={class_record['image_count']}, actual={stats['total_files']}"
                        )
                    if stats["ok_files"] == 0:
                        raise ManifestError(f"class has no successfully decoded image: {class_record['class_id']}")
                    class_stats.append(stats)
        write_temp_idempotent(output_dir / "manifest.jsonl", manifest_temp)
        write_temp_idempotent(output_dir / "bad-images.jsonl", bad_temp)
    finally:
        manifest_temp.unlink(missing_ok=True)
        bad_temp.unlink(missing_ok=True)

    rng = random.Random(SAMPLE_SEED)
    samples = []
    for class_record in classes:
        candidates = successful_paths[int(class_record["class_id"])]
        selected = sorted(rng.sample(candidates, min(SAMPLE_COUNT_PER_CLASS, len(candidates))))
        samples.append(
            {
                "class_id": class_record["class_id"],
                "official_name": class_record["official_name"],
                "relative_paths": selected,
            }
        )
    summary = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "data_version": "data-v1",
        "d0_freeze_id": "d0-taxonomy-v1",
        "data_root": repo_relative(data_root),
        "ordering": "taxonomy class_id, then relative POSIX path",
        "total_files": total_files,
        "ok_files": total_files - bad_files,
        "bad_files": bad_files,
        "unique_relative_paths": len(seen_paths),
        "paths_unique": len(seen_paths) == total_files,
        "extension_counts": dict(sorted(extension_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "decode_error_types": dict(sorted(error_type_counts.items())),
        "extension_format_mismatch_files": extension_format_mismatches,
        "width_min": min(width_values),
        "width_max": max(width_values),
        "height_min": min(height_values),
        "height_max": max(height_values),
        "class_stats": class_stats,
        "random_sample_seed": SAMPLE_SEED,
        "sample_count_per_class": SAMPLE_COUNT_PER_CLASS,
        "sample_paths": samples,
    }
    return summary, samples


def verify_artifacts(
    output_dir: Path,
    taxonomy_path: Path,
    d0_report_path: Path,
    d0_checksums_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    required = {
        "manifest": output_dir / "manifest.jsonl",
        "bad": output_dir / "bad-images.jsonl",
        "config": output_dir / "d1-config.json",
        "sample": output_dir / "sampled-successful-images.json",
        "summary": output_dir / "d1-summary.json",
        "checksums": output_dir / "checksums.sha256",
    }
    if any(not path.is_file() for path in required.values()):
        raise ManifestError("D1 artifact directory is incomplete")
    config = require_mapping(load_json(required["config"]), "D1 config")
    summary = require_mapping(load_json(required["summary"]), "D1 summary")
    taxonomy = require_mapping(load_json(taxonomy_path), "taxonomy")
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or len(classes) != 203:
        raise ManifestError("taxonomy is not the 203-class taxonomy")
    expected_counts = {int(item["class_id"]): int(item["image_count"]) for item in classes}
    expected_classes = {int(item["class_id"]): item for item in classes}
    expected_fields = set(config["record_fields"])
    seen: set[str] = set()
    bad_paths: list[str] = []
    counts: Counter[int] = Counter()
    ok_counts: Counter[int] = Counter()
    bad_counts: Counter[int] = Counter()
    path_class_ids: dict[str, int] = {}
    manifest_rows = 0
    with required["manifest"].open(encoding="utf-8") as stream:
        for line in stream:
            record = require_mapping(json.loads(line), "manifest record")
            if set(record) != expected_fields:
                raise ManifestError("manifest record fields differ from D1 config")
            path = str(record["relative_path"])
            if (
                path in seen
                or path.startswith("/")
                or normpath(path) != path
                or path.startswith("../")
                or path.startswith("data/views/")
            ):
                raise ManifestError(f"manifest path is duplicated or outside raw data: {path}")
            seen.add(path)
            class_id = int(record["class_id"])
            expected_class = expected_classes.get(class_id)
            if expected_class is None:
                raise ManifestError(f"manifest contains unknown class_id: {class_id}")
            for field in (
                "official_name",
                "local_directory",
                "host_group",
                "host_id",
                "host_zh",
                "category",
                "category_zh",
            ):
                if record[field] != expected_class[field]:
                    raise ManifestError(f"manifest taxonomy mismatch for class_id={class_id}, field={field}")
            path_class_ids[path] = class_id
            counts[class_id] += 1
            if record["decode_status"] == "ok":
                ok_counts[class_id] += 1
            elif record["decode_status"] == "bad":
                bad_counts[class_id] += 1
                bad_paths.append(path)
            else:
                raise ManifestError(f"unknown decode status: {record['decode_status']}")
            manifest_rows += 1
    with required["bad"].open(encoding="utf-8") as stream:
        listed_bad = []
        for line in stream:
            record = require_mapping(json.loads(line), "bad record")
            listed_bad.append(str(record["relative_path"]))
            if record["decode_status"] != "bad":
                raise ManifestError("bad image list contains a non-bad record")
    if listed_bad != bad_paths:
        raise ManifestError("bad image list does not match bad manifest rows")
    if manifest_rows != int(summary["total_files"]) or manifest_rows != sum(expected_counts.values()):
        raise ManifestError("manifest row count does not match summary or taxonomy")
    if len(seen) != manifest_rows or int(summary["unique_relative_paths"]) != len(seen):
        raise ManifestError("manifest relative paths are not unique")
    if any(counts[class_id] != expected for class_id, expected in expected_counts.items()):
        raise ManifestError("manifest per-class counts do not match taxonomy")
    if any(ok_counts[class_id] == 0 for class_id in expected_counts):
        raise ManifestError("at least one class has no successfully decoded image")
    if len(bad_paths) != int(summary["bad_files"]):
        raise ManifestError("bad image count does not match summary")
    sample_payload = require_mapping(load_json(required["sample"]), "sample payload")
    sample_classes = sample_payload.get("samples")
    if not isinstance(sample_classes, list) or len(sample_classes) != 203:
        raise ManifestError("random sample does not cover all classes")
    bad_path_set = set(bad_paths)
    for sample in sample_classes:
        sample_class_id = int(sample.get("class_id"))
        sample_paths = sample.get("relative_paths", [])
        if not isinstance(sample_paths, list) or len(sample_paths) != SAMPLE_COUNT_PER_CLASS:
            raise ManifestError(f"random sample count differs for class_id={sample_class_id}")
        for path in sample_paths:
            if path not in seen or path in bad_path_set or path_class_ids[path] != sample_class_id:
                raise ManifestError(f"sample is not a successfully decoded manifest path: {path}")
    checksum_lines = required["checksums"].read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        digest, separator, name = line.partition("  ")
        if separator != "  ":
            raise ManifestError(f"invalid D1 checksum line: {line!r}")
        path = REPO_ROOT / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ManifestError(f"D1 checksum mismatch: {name}")
    if sha256_file(required["manifest"]) != summary["manifest_sha256"]:
        raise ManifestError("manifest SHA-256 differs from summary")
    if sha256_file(required["bad"]) != summary["bad_images_sha256"]:
        raise ManifestError("bad image SHA-256 differs from summary")
    return {
        "manifest_rows": manifest_rows,
        "bad_files": len(bad_paths),
        "unique_relative_paths": len(seen),
        "classes_with_success": len(ok_counts),
    }


def build_d1(
    data_root: Path,
    taxonomy_path: Path,
    aliases_path: Path,
    d0_report_path: Path,
    d0_checksums_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    taxonomy_path = taxonomy_path.resolve()
    aliases_path = aliases_path.resolve()
    d0_report_path = d0_report_path.resolve()
    d0_checksums_path = d0_checksums_path.resolve()
    output_dir = output_dir.resolve()
    d0_report, taxonomy, classes = verify_d0_inputs(
        d0_report_path, d0_checksums_path, taxonomy_path, aliases_path, data_root
    )
    summary, samples = write_manifest_files(output_dir, data_root, classes, workers)
    sample_payload = {
        "schema_version": 1,
        "seed": SAMPLE_SEED,
        "count_per_class": SAMPLE_COUNT_PER_CLASS,
        "samples": samples,
    }
    d1_config = {
        "schema_version": 1,
        "stage": "D1",
        "data_version": "data-v1",
        "d0_freeze_id": d0_report["freeze_id"],
        "d0_report_sha256": sha256_file(d0_report_path),
        "data_root": repo_relative(data_root),
        "taxonomy": repo_relative(taxonomy_path),
        "aliases": repo_relative(aliases_path),
        "manifest": "manifest.jsonl",
        "bad_images": "bad-images.jsonl",
        "decode_policy": {
            "backend": "Pillow",
            "verify_then_load": True,
            "load_all_frames": True,
            "decompression_bomb_warning": "error",
            "source_mutation": False,
        },
        "record_fields": [
            "schema_version",
            "relative_path",
            "class_id",
            "official_name",
            "local_directory",
            "host_group",
            "host_id",
            "host_zh",
            "category",
            "category_zh",
            "extension",
            "file_size_bytes",
            "decode_status",
            "decode_error_type",
            "decode_error",
            "format",
            "width",
            "height",
            "mode",
            "channels",
            "frame_count",
        ],
        "ordering": "taxonomy class_id, then relative POSIX path",
        "random_sample_seed": SAMPLE_SEED,
        "sample_count_per_class": SAMPLE_COUNT_PER_CLASS,
        "workers": workers,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    write_bytes_idempotent(output_dir / "d1-config.json", canonical_json(d1_config))
    write_bytes_idempotent(output_dir / "sampled-successful-images.json", canonical_json(sample_payload))
    summary["runtime"] = {
        "python": platform.python_version(),
        "pillow": Image.__version__,
    }
    summary["manifest_sha256"] = sha256_file(output_dir / "manifest.jsonl")
    summary["bad_images_sha256"] = sha256_file(output_dir / "bad-images.jsonl")
    summary["d1_config_sha256"] = sha256_file(output_dir / "d1-config.json")
    summary["sample_sha256"] = sha256_file(output_dir / "sampled-successful-images.json")
    write_bytes_idempotent(output_dir / "d1-summary.json", canonical_json(summary))

    checksum_paths = [
        d0_report_path,
        d0_checksums_path,
        taxonomy_path,
        aliases_path,
        Path(__file__).resolve(),
        output_dir / "manifest.jsonl",
        output_dir / "bad-images.jsonl",
        output_dir / "d1-config.json",
        output_dir / "sampled-successful-images.json",
        output_dir / "d1-summary.json",
    ]
    checksum_lines = []
    for path in sorted(set(checksum_paths)):
        checksum_lines.append(f"{sha256_file(path)}  {repo_relative(path)}")
    write_bytes_idempotent(output_dir / "checksums.sha256", ("\n".join(checksum_lines) + "\n").encode("utf-8"))
    verify_artifacts(output_dir, taxonomy_path, d0_report_path, d0_checksums_path)
    return summary


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            result = verify_artifacts(args.output_dir, args.taxonomy, args.d0_report, args.d0_checksums)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        summary = build_d1(
            args.data_root,
            args.taxonomy,
            args.aliases,
            args.d0_report,
            args.d0_checksums,
            args.output_dir,
            args.workers,
        )
    except (ManifestError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "data_version": summary["data_version"],
                "total_files": summary["total_files"],
                "ok_files": summary["ok_files"],
                "bad_files": summary["bad_files"],
                "manifest_sha256": summary["manifest_sha256"],
                "bad_images_sha256": summary["bad_images_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
